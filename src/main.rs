use std::{
    env,
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write, stdout},
    path::{Path, PathBuf},
};

use async_openai::{
    types::{
        ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use futures::StreamExt;

const DELIMITER: &str = "---";

fn find_latest_md_file() -> Option<PathBuf> {
    let entries = fs::read_dir(".").ok()?;
    entries
        .filter_map(|entry| entry.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
        .max_by_key(|e| e.metadata().ok()?.modified().ok())
        .map(|e| e.path())
}

/// Parses a chat file into messages and returns whether the last message is an unterminated user message
fn parse_chat(file: &Path) -> Result<(Vec<ChatCompletionRequestMessage>, bool), Box<dyn Error>> {
    let file = File::open(file)?;
    let reader = BufReader::new(file);
    let mut messages = vec![];
    let mut current = String::new();
    let mut is_user = true;
    let mut has_pending = false;

    for line in reader.lines() {
        let line = line?;
        if line.trim() == DELIMITER {
            if !current.trim().is_empty() {
                let msg = if is_user {
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(current.trim())
                        .build()?.into()
                } else {
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(current.trim())
                        .build()?.into()
                };
                messages.push(msg);
                current.clear();
                is_user = !is_user;
            }
        } else {
            current.push_str(&line);
            current.push('\n');
        }
    }

    if !current.trim().is_empty() {
        has_pending = true;
        let msg = ChatCompletionRequestUserMessageArgs::default()
            .content(current.trim())
            .build()?.into();
        messages.push(msg);
    }

    Ok((messages, has_pending))
}

fn open_chat_file(path: &Path) -> Result<BufWriter<File>, Box<dyn Error>> {
    let file = OpenOptions::new().append(true).open(path)?;
    Ok(BufWriter::new(file))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let chat_file = env::args().nth(1).map(PathBuf::from)
        .or_else(find_latest_md_file)
        .ok_or("No Markdown file provided or found")?;

    let (messages, has_pending_user_msg) = parse_chat(&chat_file)?;

    let client = Client::new();
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4") // or "gpt-3.5-turbo"
        .max_tokens(512u32)
        .messages(messages)
        .build()?;

    let mut stream = client.chat().create_stream(request).await?;
    let mut file_writer = open_chat_file(&chat_file)?;
    let mut stdout_writer = stdout().lock();

    // If the user just typed a new message (not followed by delimiter), write a newline
    if has_pending_user_msg {
    writeln!(file_writer, "{DELIMITER}")?;
    file_writer.flush()?;
}

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(resp) => {
                for choice in &resp.choices {
                    if let Some(content) = &choice.delta.content {
                        write!(stdout_writer, "{}", content)?;
                        write!(file_writer, "{}", content)?;
                    }
                }
                stdout_writer.flush()?;
                file_writer.flush()?;
            }
            Err(e) => {
                writeln!(file_writer, "\n[Error: {}]", e)?;
                break;
            }
        }
    }

    // Append final delimiter
    writeln!(file_writer, "\n{DELIMITER}")?;
    file_writer.flush()?;

    Ok(())
}
