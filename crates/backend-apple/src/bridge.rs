//! Persistent subprocess bridge to `python/mlx_bridge.py`.
//!
//! One Python worker is spawned lazily on first use and kept alive for the
//! life of the server, so MLX models stay **hot** in the worker's memory
//! (unified memory on Apple silicon) instead of being reloaded per call.
//!
//! Protocol: newline-delimited JSON over the child's stdin/stdout.
//!
//! * unary ops (`ping`, `embed`, `describe`): one request line in, exactly
//!   one reply line out — `{"status":"ok","result":...}` or
//!   `{"status":"error","message":...}`.
//! * streaming ops (`generate`): one request line in, zero or more
//!   `{"status":"chunk",...}` lines, terminated by `{"status":"done",...}`
//!   or `{"status":"error",...}`.
//!
//! The worker handles one request at a time; the Rust side serializes
//! callers through an async mutex. The Python side owns all MLX/Metal
//! specifics so this crate stays pure Rust and compiles on every platform
//! (Linux CI type-checks the arch-apple wiring; at runtime off-macOS the
//! spawn simply fails with a clear `Unavailable`).

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use inferstream_backend::BackendError;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Where to find the Python worker.
#[derive(Debug, Clone)]
pub struct MlxWorkerConfig {
    /// Python interpreter inside the MLX venv (see `scripts/setup-mlx.sh`).
    pub python: PathBuf,
    /// Path to `python/mlx_bridge.py`.
    pub script: PathBuf,
}

impl MlxWorkerConfig {
    /// Resolve from the environment with repo-relative defaults:
    /// `INFERSTREAM_MLX_PYTHON` (default `.venv/bin/python`) and
    /// `INFERSTREAM_MLX_BRIDGE` (default `python/mlx_bridge.py`).
    pub fn from_env() -> Self {
        let python = std::env::var("INFERSTREAM_MLX_PYTHON")
            .unwrap_or_else(|_| ".venv/bin/python".to_string());
        let script = std::env::var("INFERSTREAM_MLX_BRIDGE")
            .unwrap_or_else(|_| "python/mlx_bridge.py".to_string());
        Self {
            python: PathBuf::from(python),
            script: PathBuf::from(script),
        }
    }
}

struct Session {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

#[derive(Default)]
struct WorkerState {
    session: Option<Session>,
}

/// Handle to the persistent MLX worker. Cheap to clone via `Arc`; share one
/// per process so every model lives in the same hot Python worker.
pub struct MlxWorker {
    config: MlxWorkerConfig,
    state: Arc<Mutex<WorkerState>>,
}

/// One parsed reply line from the worker.
#[derive(Debug)]
pub enum BridgeReply {
    Ok(Value),
    Chunk(Value),
    Done(Value),
}

impl MlxWorker {
    pub fn new(config: MlxWorkerConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(WorkerState::default())),
        }
    }

    pub fn config(&self) -> &MlxWorkerConfig {
        &self.config
    }

    fn spawn(&self) -> Result<Session, BackendError> {
        if !self.config.python.exists() {
            return Err(BackendError::Unavailable(format!(
                "python interpreter not found at {} — run scripts/setup-mlx.sh (or set \
                 INFERSTREAM_MLX_PYTHON)",
                self.config.python.display()
            )));
        }
        if !self.config.script.exists() {
            return Err(BackendError::Unavailable(format!(
                "mlx bridge script not found at {} (set INFERSTREAM_MLX_BRIDGE)",
                self.config.script.display()
            )));
        }
        let mut child = Command::new(&self.config.python)
            .arg("-u")
            .arg(&self.config.script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Worker logs (model download progress, tracebacks) go straight
            // to the server's stderr.
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                BackendError::Unavailable(format!(
                    "failed to spawn mlx bridge ({} {}): {e}",
                    self.config.python.display(),
                    self.config.script.display()
                ))
            })?;
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped")).lines();
        info!(
            python = %self.config.python.display(),
            script = %self.config.script.display(),
            "spawned persistent mlx bridge worker"
        );
        Ok(Session {
            child,
            stdin,
            stdout,
        })
    }

    async fn ensure_session<'a>(
        &self,
        state: &'a mut WorkerState,
    ) -> Result<&'a mut Session, BackendError> {
        // Drop a dead session so the next call respawns instead of writing
        // into a broken pipe forever.
        if let Some(session) = state.session.as_mut() {
            if let Ok(Some(status)) = session.child.try_wait() {
                warn!(%status, "mlx bridge worker exited; respawning");
                state.session = None;
            }
        }
        if state.session.is_none() {
            state.session = Some(self.spawn()?);
        }
        Ok(state.session.as_mut().expect("session just ensured"))
    }

    async fn send(session: &mut Session, request: &Value) -> Result<(), BackendError> {
        let mut line = serde_json::to_vec(request)
            .map_err(|e| BackendError::Internal(format!("failed to encode bridge request: {e}")))?;
        line.push(b'\n');
        session
            .stdin
            .write_all(&line)
            .await
            .map_err(|e| BackendError::Unavailable(format!("mlx bridge stdin closed: {e}")))?;
        session
            .stdin
            .flush()
            .await
            .map_err(|e| BackendError::Unavailable(format!("mlx bridge stdin closed: {e}")))?;
        Ok(())
    }

    fn parse_reply(line: &str) -> Result<BridgeReply, BackendError> {
        let value: Value = serde_json::from_str(line)
            .map_err(|e| BackendError::Internal(format!("malformed bridge reply: {e}")))?;
        match value.get("status").and_then(Value::as_str) {
            Some("ok") => Ok(BridgeReply::Ok(
                value.get("result").cloned().unwrap_or(Value::Null),
            )),
            Some("chunk") => Ok(BridgeReply::Chunk(value)),
            Some("done") => Ok(BridgeReply::Done(
                value.get("result").cloned().unwrap_or(Value::Null),
            )),
            Some("error") => {
                let message = value
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown bridge error");
                Err(BackendError::Internal(format!("mlx bridge: {message}")))
            }
            other => Err(BackendError::Internal(format!(
                "bridge reply with unknown status {other:?}"
            ))),
        }
    }

    async fn read_reply(session: &mut Session) -> Result<BridgeReply, BackendError> {
        let line = session
            .stdout
            .next_line()
            .await
            .map_err(|e| BackendError::Unavailable(format!("mlx bridge stdout error: {e}")))?
            .ok_or_else(|| {
                BackendError::Unavailable(
                    "mlx bridge worker exited mid-request (see server stderr for the \
                     Python traceback)"
                        .into(),
                )
            })?;
        Self::parse_reply(&line)
    }

    /// Unary request/reply. Serialized with all other callers; the worker is
    /// single-threaded on purpose (one Metal command queue, no interleaving).
    pub async fn call(&self, request: Value) -> Result<Value, BackendError> {
        let mut state = self.state.lock().await;
        let result = async {
            let session = self.ensure_session(&mut state).await?;
            Self::send(session, &request).await?;
            match Self::read_reply(session).await? {
                BridgeReply::Ok(result) => Ok(result),
                other => Err(BackendError::Internal(format!(
                    "expected a unary reply, bridge sent {other:?}"
                ))),
            }
        }
        .await;
        if matches!(result, Err(BackendError::Unavailable(_))) {
            // Transport-level failure: drop the session so the next call
            // starts a fresh worker.
            state.session = None;
        }
        result
    }

    /// Streaming request: send once, then receive `chunk` values until the
    /// worker sends `done` (yielded last as [`BridgeReply::Done`]). The
    /// worker lock is held for the whole stream — generation is exclusive.
    pub async fn call_stream(
        &self,
        request: Value,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<BridgeReply, BackendError>>, BackendError> {
        let mut state = Arc::clone(&self.state).lock_owned().await;
        // Fail fast (spawn/write errors) before returning a stream.
        self.ensure_session(&mut state).await?;
        let session = state.session.as_mut().expect("session ensured");
        Self::send(session, &request).await?;

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        tokio::spawn(async move {
            // `state` (the owned lock guard) moves in here and is released
            // when the stream finishes. If the receiver drops early we keep
            // draining until `done` so the session stays usable.
            let mut client_gone = false;
            loop {
                let reply = {
                    let session = state.session.as_mut().expect("session held");
                    Self::read_reply(session).await
                };
                match reply {
                    Ok(BridgeReply::Chunk(value)) => {
                        if !client_gone && tx.send(Ok(BridgeReply::Chunk(value))).await.is_err() {
                            client_gone = true;
                        }
                    }
                    Ok(BridgeReply::Done(value)) => {
                        if !client_gone {
                            let _ = tx.send(Ok(BridgeReply::Done(value))).await;
                        }
                        return;
                    }
                    Ok(BridgeReply::Ok(value)) => {
                        if !client_gone {
                            let _ = tx
                                .send(Err(BackendError::Internal(format!(
                                    "expected chunk/done, bridge sent ok: {value}"
                                ))))
                                .await;
                        }
                        return;
                    }
                    Err(error) => {
                        if matches!(error, BackendError::Unavailable(_)) {
                            state.session = None;
                        }
                        if !client_gone {
                            let _ = tx.send(Err(error)).await;
                        }
                        return;
                    }
                }
            }
        });
        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_reply_variants() {
        assert!(matches!(
            MlxWorker::parse_reply(r#"{"status":"ok","result":{"x":1}}"#),
            Ok(BridgeReply::Ok(_))
        ));
        assert!(matches!(
            MlxWorker::parse_reply(r#"{"status":"chunk","token":"hi"}"#),
            Ok(BridgeReply::Chunk(_))
        ));
        assert!(matches!(
            MlxWorker::parse_reply(r#"{"status":"done","result":{}}"#),
            Ok(BridgeReply::Done(_))
        ));
        assert!(matches!(
            MlxWorker::parse_reply(r#"{"status":"error","message":"boom"}"#),
            Err(BackendError::Internal(_))
        ));
        assert!(matches!(
            MlxWorker::parse_reply("not json"),
            Err(BackendError::Internal(_))
        ));
        assert!(matches!(
            MlxWorker::parse_reply(r#"{"status":"wat"}"#),
            Err(BackendError::Internal(_))
        ));
    }

    #[tokio::test]
    async fn missing_python_reports_unavailable() {
        let worker = MlxWorker::new(MlxWorkerConfig {
            python: PathBuf::from("/definitely/not/python"),
            script: PathBuf::from("/definitely/not/bridge.py"),
        });
        let result = worker.call(serde_json::json!({"op": "ping"})).await;
        assert!(matches!(result, Err(BackendError::Unavailable(_))));
    }

    #[test]
    fn worker_config_from_env_defaults() {
        // Only assert the defaults when the vars are unset in this process.
        if std::env::var("INFERSTREAM_MLX_PYTHON").is_err()
            && std::env::var("INFERSTREAM_MLX_BRIDGE").is_err()
        {
            let config = MlxWorkerConfig::from_env();
            assert_eq!(config.python, PathBuf::from(".venv/bin/python"));
            assert_eq!(config.script, PathBuf::from("python/mlx_bridge.py"));
        }
    }
}
