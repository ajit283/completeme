use std::{
    collections::HashMap, // Added for HashMap
    env,
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write, stdout},
    path::{Path, PathBuf},
};

use async_openai::{
    Client,
    config::OpenAIConfig, // Ensure OpenAIConfig is imported
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
};
use futures::StreamExt;
use serde::Deserialize;

const DELIMITER: &str = "===";

// --- Structs for TOML Configuration (as defined above) ---
#[derive(Deserialize, Debug, Default)]
struct EndpointsTomlConfig {
    default_openai_endpoint: Option<String>,
    openai_endpoints: Option<HashMap<String, OpenAIEndpointToml>>,
}

// In your TOML Structs section:
#[derive(Deserialize, Debug, Clone)]
struct OpenAIEndpointToml {
    api_key: Option<String>,
    api_base: Option<String>,
    default_model: Option<String>, // <-- New field
}
// --- End of TOML Structs ---

// --- load_toml_config function (as defined above) ---
fn load_toml_config(path: &Path) -> Result<EndpointsTomlConfig, Box<dyn Error>> {
    if !path.exists() {
        return Ok(EndpointsTomlConfig::default());
    }
    let content = fs::read_to_string(path)?;
    let config: EndpointsTomlConfig = toml::from_str(&content)?;
    Ok(config)
}
// --- End of load_toml_config ---

fn find_latest_md_file() -> Option<PathBuf> {
    let entries = fs::read_dir(".").ok()?;
    entries
        .filter_map(|entry| entry.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
        .max_by_key(|e| e.metadata().ok()?.modified().ok())
        .map(|e| e.path())
}

fn parse_chat(file: &Path) -> Result<(Vec<ChatCompletionRequestMessage>, bool), Box<dyn Error>> {
    let file_open_result = File::open(file);
    if file_open_result.is_err() {
        // If file doesn't exist, treat as empty chat with no pending message
        return Ok((vec![], false));
    }
    let file = file_open_result?;
    let reader = BufReader::new(file);
    let mut messages = vec![];
    let mut current = String::new();
    let mut is_user = true;
    let mut has_pending_user_msg = false;

    let mut lines_iter = reader.lines();

    while let Some(line_result) = lines_iter.next() {
        let line = line_result?;
        if line.trim() == DELIMITER {
            if !current.trim().is_empty() {
                let msg = if is_user {
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(current.trim_end_matches('\n'))
                        .build()?
                        .into()
                } else {
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(current.trim_end_matches('\n'))
                        .build()?
                        .into()
                };
                messages.push(msg);
                current.clear();
                is_user = !is_user;
                has_pending_user_msg = false;
            }
        } else {
            current.push_str(&line);
            current.push('\n');
        }
    }

    if !current.trim().is_empty() {
        let msg = ChatCompletionRequestUserMessageArgs::default()
            .content(current.trim_end_matches('\n'))
            .build()?
            .into();
        messages.push(msg);
        // If is_user is true here, it means the last unterminated block is a user message.
        // The original logic implied any unterminated content is a pending user message.
        has_pending_user_msg = is_user; // Only pending if it's a user's turn and content exists
    }
    Ok((messages, has_pending_user_msg))
}

fn open_chat_file(path: &Path) -> Result<BufWriter<File>, Box<dyn Error>> {
    let file = OpenOptions::new().append(true).create(true).open(path)?;
    Ok(BufWriter::new(file))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // --- Parse CLI for --endpoint argument ---
    let args: Vec<String> = env::args().collect();
    let mut chosen_endpoint_name_from_cli: Option<String> = None;
    let mut i = 1; // Skip program name
    while i < args.len() {
        if args[i] == "--endpoint" {
            if i + 1 < args.len() {
                chosen_endpoint_name_from_cli = Some(args[i + 1].clone());
                break;
            } else {
                eprintln!("Warning: --endpoint flag provided without a value. It will be ignored.");
                // No break here, allow loop to finish in case of other args later (though not used now)
            }
        }
        i += 1;
    }
    // --- End of CLI Parsing ---

    let chat_file_path_arg = args
        .iter()
        .find(|arg| Path::new(arg).extension().map_or(false, |ext| ext == "md"));

    let chat_file = chat_file_path_arg
        .map(PathBuf::from)
        .or_else(find_latest_md_file)
        .ok_or_else::<Box<dyn Error>, _>(|| "No Markdown file provided or found".into())?;

    println!("Using chat file: {}", chat_file.display());

    let (messages, has_pending_user_msg) = parse_chat(&chat_file)?;

    // --- Load and Select OpenAI Endpoint Configuration ---
    let toml_path = Path::new("endpoints.toml");
    let toml_config = load_toml_config(toml_path).unwrap_or_else(|err| {
        eprintln!(
            "Warning: Could not load or parse '{}': {}. Will use default OpenAI config.",
            toml_path.display(),
            err
        );
        EndpointsTomlConfig::default()
    });

    let mut selected_endpoint_details: Option<OpenAIEndpointToml> = None;
    let mut config_source_message: String =
        "Defaults (e.g., OPENAI_API_KEY and default OpenAI base)".to_string();

    // Determine which endpoint name to try and load
    let endpoint_name_to_load: Option<String> =
        chosen_endpoint_name_from_cli.or(toml_config.default_openai_endpoint); // Use clone if String, not needed for Option<String> if taken

    if let Some(name) = endpoint_name_to_load {
        // An endpoint name is specified (either by CLI or TOML default)
        if let Some(endpoints_map) = &toml_config.openai_endpoints {
            if let Some(details) = endpoints_map.get(&name) {
                selected_endpoint_details = Some(details.clone());
                config_source_message = format!("endpoint '{}' from TOML configuration", name);
            } else {
                eprintln!(
                    "Warning: Endpoint '{}' (specified by CLI or as TOML default) not found in [openai_endpoints] table. Falling back.",
                    name
                );
            }
        } else {
            eprintln!(
                "Warning: Endpoint '{}' was specified, but no [openai_endpoints] table found in '{}'. Falling back.",
                name,
                toml_path.display()
            );
        }
    } else {
        // No specific endpoint name given (neither CLI nor TOML default).
        // Fallback to the "first" available endpoint in the TOML if any exist.
        if let Some(endpoints_map) = &toml_config.openai_endpoints {
            if !endpoints_map.is_empty() {
                // .iter().next() gets an arbitrary entry from HashMap. Order is not guaranteed.
                if let Some((first_name, first_details)) = endpoints_map.iter().next() {
                    selected_endpoint_details = Some(first_details.clone());
                    config_source_message = format!(
                        "the first available endpoint '{}' from TOML (no specific endpoint chosen)",
                        first_name
                    );
                }
            }
            // If endpoints_map is empty, selected_endpoint_details remains None.
        }
        // If toml_config.openai_endpoints itself is None, selected_endpoint_details remains None.
    }

    // Configure the OpenAI client
    let mut openai_client_config = OpenAIConfig::new(); // Start with defaults (OPENAI_API_KEY, default base)

    let mut model_to_use = "gpt-4o".to_string(); // Hardcoded fallback model if no other source provides one

    if let Some(details) = selected_endpoint_details {
        // If an endpoint was selected from TOML, apply its specific configurations
        if let Some(api_key_from_toml) = details.api_key {
            openai_client_config = openai_client_config.with_api_key(api_key_from_toml);
        }
        if let Some(api_base_from_toml) = details.api_base {
            openai_client_config = openai_client_config.with_api_base(api_base_from_toml);
        }

        if let Some(default_model_from_toml) = details.default_model {
            model_to_use = default_model_from_toml;
        }
    }

    println!("OpenAI client configured using: {}.", config_source_message);

    let client = Client::with_config(openai_client_config);
    // --- End of OpenAI Endpoint Configuration ---

    if messages.is_empty() && !has_pending_user_msg {
        // If no history and no new message
        eprintln!(
            "No messages to send (chat file is empty or ends with a delimiter after assistant's message). Exiting."
        );
        return Ok(());
    }
    // If messages is empty BUT has_pending_user_msg, it means the file only contained a new user message.
    // This is a valid scenario to send to the API.
    //
    //

    let mut base_args_builder = CreateChatCompletionRequestArgs::default();
    let request = base_args_builder
        .model(&model_to_use) // Or your preferred model
        .messages(messages)
        .build()?; // `messages` can be empty if only `has_pending_user_msg` is true initially.
    // However, `parse_chat` ensures `messages` contains the pending user message.

    let mut stream = client.chat().create_stream(request).await?;
    let mut file_writer = open_chat_file(&chat_file)?;
    let mut stdout_writer = stdout().lock();

    if has_pending_user_msg {
        // Ensure the user's part is properly delimited before assistant's response
        // If the file is not empty, ensure we are on a new line before writing delimiter
        let metadata = fs::metadata(&chat_file)?;
        if metadata.len() > 0 {
            // Check if file is not empty
            // Check if file ends with a newline, if not, add one.
            // This is complex to check robustly without reading the end of file.
            // A simpler approach: always add a newline IF there was content from user,
            // assuming the content itself doesn't end with multiple newlines.
            // The content in message is trimmed of its last \n by parse_chat.
            // writeln!(file_writer)?; // Add a newline for separation before delimiter
        }
        writeln!(file_writer, "{DELIMITER}")?;
        file_writer.flush()?;
    }

    let mut assistant_response_started = false;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(resp) => {
                for choice in &resp.choices {
                    if let Some(content) = &choice.delta.content {
                        if !content.is_empty() {
                            // Check if content is not empty
                            assistant_response_started = true;
                        }
                        write!(stdout_writer, "{}", content)?;
                        write!(file_writer, "{}", content)?;
                    }
                }
                stdout_writer.flush()?;
                file_writer.flush()?;
            }
            Err(e) => {
                eprintln!("\nError during stream: {}", e);
                writeln!(file_writer, "\n[Error: {}]", e)?;
                file_writer.flush()?;
                break;
            }
        }
    }

    if assistant_response_started {
        writeln!(stdout_writer)?;
        writeln!(file_writer)?;
        writeln!(file_writer, "{DELIMITER}")?;
        file_writer.flush()?;
    } else {
        println!("\nAssistant did not provide a content response.");
        // If the user message was pending and assistant didn't respond,
        // the file will have user_message\n---\n (no second delimiter)
        // This seems fine.
    }

    Ok(())
}
