//! LLM request/response JSONL logger.
//!
//! Every call to `LlmClient::create_message` / `create_message_stream`
//! appends one JSONL record to `.codewhale/{session_id}.jsonl` under the
//! current working directory.  Always enabled.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use chrono::Utc;
use futures_util::StreamExt;
use serde::Serialize;
use uuid::Uuid;

use crate::llm_client::StreamEventBox;
use crate::models::{
    ContentBlock, ContentBlockStart, Delta, MessageRequest, MessageResponse, StreamEvent, Usage,
};

// === JSONL Record ===

#[derive(Serialize)]
struct LlmLogRecord {
    timestamp: String,
    request_id: String,
    session_id: Option<String>,
    kind: Option<String>,
    turn_number: Option<u64>,
    provider: &'static str,
    model: String,
    mode: &'static str,
    stop_reason: Option<String>,
    request: MessageRequest,
    response: Option<MessageResponse>,
    duration_ms: u64,
    error: Option<String>,
}

/// Extract a string field from `MessageRequest.metadata`.
fn extract_meta_str(request: &MessageRequest, key: &str) -> Option<String> {
    request
        .metadata
        .as_ref()
        .and_then(|m| m.get(key))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extract a u64 field from `MessageRequest.metadata`.
fn extract_meta_u64(request: &MessageRequest, key: &str) -> Option<u64> {
    request
        .metadata
        .as_ref()
        .and_then(|m| m.get(key))
        .and_then(|v| v.as_u64())
}

/// Build the JSONL file path from the session_id: `{cwd}/.codewhale/{session_id}.jsonl`.
/// Falls back to `_no_session.jsonl` when no session_id is present.
fn log_path_for(session_id: Option<&str>) -> PathBuf {
    let name = match session_id {
        Some(id) if !id.is_empty() => format!("{id}.jsonl"),
        _ => "_no_session.jsonl".to_string(),
    };
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".codewhale")
        .join(name)
}

// === File I/O ===

async fn write_record(path: &PathBuf, record: &LlmLogRecord) {
    let result = async {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        use tokio::io::AsyncWriteExt;
        let line = serde_json::to_string(record)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        Ok::<(), anyhow::Error>(())
    }
    .await;
    if let Err(err) = result {
        crate::logging::warn(format!("LLM log write failed: {err}"));
    }
}

// === Non-streaming entry point ===

/// Log a completed non-streaming request/response pair.
pub async fn log_non_streaming(
    provider: &'static str,
    model: &str,
    request: MessageRequest,
    result: &anyhow::Result<MessageResponse>,
    start: Instant,
) {
    let session_id = extract_meta_str(&request, "session_id");
    let path = log_path_for(session_id.as_deref());
    let record = LlmLogRecord {
        timestamp: Utc::now().to_rfc3339(),
        request_id: Uuid::new_v4().to_string(),
        session_id,
        kind: extract_meta_str(&request, "kind"),
        turn_number: extract_meta_u64(&request, "turn_number"),
        provider,
        model: model.to_string(),
        mode: "non_streaming",
        stop_reason: result.as_ref().ok().and_then(|r| r.stop_reason.clone()),
        request,
        response: result.as_ref().ok().cloned(),
        duration_ms: start.elapsed().as_millis() as u64,
        error: result.as_ref().err().map(|e| e.to_string()),
    };
    write_record(&path, &record).await;
}

/// Log a streaming call that failed before producing a stream.
pub async fn log_stream_error(
    provider: &'static str,
    model: &str,
    request: MessageRequest,
    error: &anyhow::Error,
    start: Instant,
) {
    let session_id = extract_meta_str(&request, "session_id");
    let path = log_path_for(session_id.as_deref());
    let record = LlmLogRecord {
        timestamp: Utc::now().to_rfc3339(),
        request_id: Uuid::new_v4().to_string(),
        session_id,
        kind: extract_meta_str(&request, "kind"),
        turn_number: extract_meta_u64(&request, "turn_number"),
        provider,
        model: model.to_string(),
        mode: "streaming",
        stop_reason: None,
        request,
        response: None,
        duration_ms: start.elapsed().as_millis() as u64,
        error: Some(error.to_string()),
    };
    write_record(&path, &record).await;
}

// === Stream accumulator ===

/// Accumulates `StreamEvent`s into a reconstructed `MessageResponse`.
struct StreamAccumulator {
    response_id: String,
    model: String,
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    usage: Option<Usage>,
    /// Accumulated partial JSON strings per content-block index (for tool_use InputJsonDelta).
    partial_json: HashMap<u32, String>,
}

use std::collections::HashMap;

impl StreamAccumulator {
    fn new(model: String) -> Self {
        Self {
            response_id: String::new(),
            model,
            content: Vec::new(),
            stop_reason: None,
            usage: None,
            partial_json: HashMap::new(),
        }
    }

    fn push_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::MessageStart { message } => {
                self.response_id = message.id.clone();
                if !message.model.is_empty() {
                    self.model = message.model.clone();
                }
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let idx = *index as usize;
                if idx >= self.content.len() {
                    self.content.resize_with(idx + 1, || {
                        ContentBlock::Text {
                            text: String::new(),
                            cache_control: None,
                        }
                    });
                }
                match content_block {
                    ContentBlockStart::Text { text } => {
                        self.content[idx] = ContentBlock::Text {
                            text: text.clone(),
                            cache_control: None,
                        };
                    }
                    ContentBlockStart::Thinking { thinking } => {
                        self.content[idx] = ContentBlock::Thinking {
                            thinking: thinking.clone(),
                        };
                    }
                    ContentBlockStart::ToolUse {
                        id, name, input, ..
                    } => {
                        let is_complete = input.is_object();
                        if !is_complete {
                            self.partial_json.insert(*index, String::new());
                        }
                        self.content[idx] = ContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                            caller: None,
                        };
                    }
                    ContentBlockStart::ServerToolUse { id, name, input } => {
                        self.content[idx] = ContentBlock::ServerToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        };
                    }
                }
            }
            StreamEvent::ContentBlockDelta { index, delta } => {
                let idx = *index as usize;
                if idx >= self.content.len() {
                    return;
                }
                match delta {
                    Delta::TextDelta { text } => {
                        if let ContentBlock::Text {
                            text: existing, ..
                        } = &mut self.content[idx]
                        {
                            existing.push_str(text);
                        }
                    }
                    Delta::ThinkingDelta { thinking } => {
                        if let ContentBlock::Thinking {
                            thinking: existing,
                        } = &mut self.content[idx]
                        {
                            existing.push_str(thinking);
                        }
                    }
                    Delta::InputJsonDelta { partial_json } => {
                        self.partial_json
                            .entry(*index)
                            .or_default()
                            .push_str(partial_json);
                    }
                }
            }
            StreamEvent::MessageDelta { delta, usage } => {
                if let Some(reason) = &delta.stop_reason {
                    self.stop_reason = Some(reason.clone());
                }
                if let Some(usage) = usage {
                    self.usage = Some(usage.clone());
                }
            }
            StreamEvent::ContentBlockStop { .. }
            | StreamEvent::MessageStop
            | StreamEvent::Ping => {}
        }
    }

    fn into_response(mut self) -> MessageResponse {
        for (&index, json_str) in &self.partial_json {
            let idx = index as usize;
            if idx >= self.content.len() {
                continue;
            }
            if let ContentBlock::ToolUse {
                id, name, input, ..
            } = &self.content[idx]
            {
                let parsed = serde_json::from_str::<serde_json::Value>(json_str);
                self.content[idx] = ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: parsed.unwrap_or_else(|_| input.clone()),
                    caller: None,
                };
            }
        }
        MessageResponse {
            id: self.response_id,
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: self.content,
            model: self.model,
            stop_reason: self.stop_reason,
            stop_sequence: None,
            container: None,
            usage: self.usage.unwrap_or_default(),
        }
    }
}

// === Stream log guard ===

/// Ensures a log record is written even if the stream is dropped before
/// `MessageStop` (error, cancellation, panic).
struct StreamLogGuard {
    accumulator: Arc<Mutex<StreamAccumulator>>,
    provider: &'static str,
    model: String,
    request: MessageRequest,
    request_id: String,
    timestamp: String,
    start: Instant,
    completed: AtomicBool,
}

impl Drop for StreamLogGuard {
    fn drop(&mut self) {
        if self.completed.load(Ordering::SeqCst) {
            return;
        }
        let session_id = extract_meta_str(&self.request, "session_id");
        let path = log_path_for(session_id.as_deref());
        let acc = std::mem::replace(
            &mut *self
                .accumulator
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
            StreamAccumulator::new(String::new()),
        );
        let response = acc.into_response();
        let has_content = !response.content.is_empty() || response.stop_reason.is_some();
        let stop_reason = response.stop_reason.clone();
        let record = LlmLogRecord {
            timestamp: self.timestamp.clone(),
            request_id: self.request_id.clone(),
            session_id,
            kind: extract_meta_str(&self.request, "kind"),
            turn_number: extract_meta_u64(&self.request, "turn_number"),
            provider: self.provider,
            model: self.model.clone(),
            mode: "streaming",
            stop_reason,
            request: self.request.clone(),
            response: if has_content { Some(response) } else { None },
            duration_ms: self.start.elapsed().as_millis() as u64,
            error: Some("stream incomplete".to_string()),
        };
        tokio::spawn(async move { write_record(&path, &record).await });
    }
}

// === Stream wrapper ===

/// Wrap a `StreamEventBox` so that every event is accumulated for logging.
/// When `MessageStop` is observed the complete record is written to
/// `.codewhale/{session_id}.jsonl` under the current working directory.
/// If the stream is dropped early, `StreamLogGuard` writes a partial record.
///
/// When logging is disabled, returns the original stream unchanged.
pub fn wrap_stream(
    provider: &'static str,
    model: String,
    request: MessageRequest,
    stream: StreamEventBox,
) -> StreamEventBox {
    let accumulator = Arc::new(Mutex::new(StreamAccumulator::new(model.clone())));
    let request_id = Uuid::new_v4().to_string();
    let timestamp = Utc::now().to_rfc3339();
    let start = Instant::now();

    let guard = Arc::new(StreamLogGuard {
        accumulator: accumulator.clone(),
        provider,
        model: model.clone(),
        request: request.clone(),
        request_id: request_id.clone(),
        timestamp: timestamp.clone(),
        start,
        completed: AtomicBool::new(false),
    });

    let acc = accumulator;
    let wrapped = stream.inspect(move |item| {
        if let Ok(event) = item {
            let mut acc = acc.lock().unwrap_or_else(|e| e.into_inner());
            acc.push_event(event);

            if matches!(event, StreamEvent::MessageStop) {
                guard.completed.store(true, Ordering::SeqCst);
                let consumed = std::mem::replace(
                    &mut *acc,
                    StreamAccumulator::new(String::new()),
                );
                let response = consumed.into_response();
                let stop_reason = response.stop_reason.clone();
                drop(acc);

                let session_id = extract_meta_str(&request, "session_id");
                let path = log_path_for(session_id.as_deref());
                let record = LlmLogRecord {
                    timestamp: timestamp.clone(),
                    request_id: request_id.clone(),
                    session_id,
                    kind: extract_meta_str(&request, "kind"),
                    turn_number: extract_meta_u64(&request, "turn_number"),
                    provider,
                    model: model.clone(),
                    mode: "streaming",
                    stop_reason,
                    request: request.clone(),
                    response: Some(response),
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: None,
                };
                tokio::spawn(async move { write_record(&path, &record).await });
            }
        }
    });

    Box::pin(wrapped)
}
