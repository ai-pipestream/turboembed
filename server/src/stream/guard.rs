//! What a stream is told when one of its workers dies (issue #2516). A
//! worker that panics drops its channel sender as it unwinds, and a
//! response channel whose senders are all gone ends the reply with OK:
//! the client would read a clean end with documents unanswered, or a
//! progressive answer with no `Complete` and no receipt. Each worker
//! is therefore watched here, and a panic in it ends the reply
//! INTERNAL instead.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tonic::Status;

use crate::stream::Dead;

/// The reply channel of a stream: its answers, or the status it ends with.
pub(crate) type Reply<T> = mpsc::Sender<std::result::Result<T, Status>>;

/// The status a stream ends with after `what` panicked. It names the
/// worker and never the panic's text, which can quote the document
/// (issue #1908); the text is on stderr, where the panic hook wrote it.
pub(crate) fn panicked(what: &str) -> Status {
    Status::internal(format!("{what} panicked: the stream ended early and what it had not answered is lost"))
}

/// The status a stream ends with when one of its batch tasks did not
/// return: a panic is named as the batch task's and never quotes its
/// payload, which tokio's `JoinError` display would (#1908); a task
/// cancelled at shutdown says so.
pub(crate) fn batch_failed(e: &tokio::task::JoinError) -> Status {
    if e.is_panic() { panicked("a batch task") } else { Status::internal("a batch task was cancelled") }
}

/// Runs a blocking stream task. A panic in it is sent on `reply` as
/// INTERNAL and answers `None`; `reply` is dropped on the way out, so
/// the stream has ended when this returns.
pub(crate) fn reply_on_panic<T, R>(reply: Reply<T>, what: &str, task: impl FnOnce() -> R) -> Option<R> {
    match catch_unwind(AssertUnwindSafe(task)) {
        Ok(r) => Some(r),
        Err(_) => {
            let _ = reply.blocking_send(Err(panicked(what)));
            None
        }
    }
}

/// Watches a spawned stream worker that owns the reply: when it ends
/// by a panic the stream is marked dead, so its reader stops admitting,
/// and the reply ends INTERNAL. The watcher's own sender keeps the
/// reply open only until the worker has ended.
pub(crate) fn watch<T: Send + 'static>(worker: JoinHandle<()>, what: &'static str, reply: Reply<T>, dead: Arc<Dead>) {
    tokio::spawn(async move {
        if worker.await.is_err_and(|e| e.is_panic()) {
            dead.set();
            let _ = reply.send(Err(panicked(what))).await;
        }
    });
}

/// The engine's input ended with no [`Envelope::End`](super::Envelope):
/// the reader stopped without closing the input, so documents the client
/// sent or was about to send are unanswered. A stream already dead is
/// ending for its own reason and says nothing more.
pub(crate) async fn reader_lost<T>(reply: &Reply<T>, dead: &Dead) {
    if !dead.is_set() {
        let lost = "the stream's reader stopped before the input closed: documents it had not handed over are lost";
        let _ = reply.send(Err(Status::internal(lost))).await;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tonic::Code;

    use super::*;
    use crate::deadline::Deadline;
    use crate::metrics::Metrics;
    use crate::proto::result::{AnalyzeOrder, AnalyzeStreamResponse};
    use crate::stream::engine::{self, Config};
    use crate::stream::{Envelope, Shared};

    fn engine() -> (mpsc::Sender<Envelope>, mpsc::Receiver<std::result::Result<AnalyzeStreamResponse, Status>>) {
        let config = Config {
            order: AnalyzeOrder::Input,
            window: 4,
            linger: Duration::from_millis(1),
            deadline: Deadline::none(),
        };
        let (work, work_rx) = mpsc::channel(4);
        let (answers, answers_rx) = mpsc::channel(4);
        engine::start(Shared::with_metrics(Metrics::new()), config, work_rx, answers, Dead::new());
        (work, answers_rx)
    }

    /// A reader that died dropped its sender without closing the input.
    #[tokio::test]
    async fn a_reader_that_stops_without_closing_the_input_ends_the_stream_internal() {
        let (work, mut answers) = engine();
        drop(work);
        let status = answers.recv().await.expect("a status, not a clean end").unwrap_err();
        assert_eq!(status.code(), Code::Internal, "{status:?}");
        assert!(status.message().contains("reader"), "{status:?}");
        assert!(answers.recv().await.is_none());
    }

    #[tokio::test]
    async fn a_half_closed_input_ends_the_stream_clean() {
        let (work, mut answers) = engine();
        work.send(Envelope::End).await.ok().unwrap();
        drop(work);
        assert!(answers.recv().await.is_none());
    }

    #[tokio::test]
    async fn a_scheduler_that_panics_ends_the_stream_internal_and_marks_it_dead() {
        let (reply, mut answers) = mpsc::channel::<std::result::Result<AnalyzeStreamResponse, Status>>(4);
        let dead = Dead::new();
        let owned = reply.clone();
        let worker = tokio::spawn(async move {
            let _owned = owned;
            panic!("an index out of range, quoting the document");
        });
        watch(worker, "the stream's scheduler", reply, dead.clone());
        let status = answers.recv().await.expect("a status, not a clean end").unwrap_err();
        assert_eq!(status.code(), Code::Internal, "{status:?}");
        assert!(status.message().contains("the stream's scheduler panicked"), "{status:?}");
        assert!(!status.message().contains("quoting the document"), "{status:?}");
        assert!(dead.is_set());
        assert!(answers.recv().await.is_none());
    }

    /// A batch task that panics ends its stream naming the task, never
    /// with the panic's payload, which the JoinError's own text carries.
    #[tokio::test]
    async fn a_batch_task_that_panics_is_named_without_its_payload() {
        let e = tokio::task::spawn_blocking(|| panic!("quoting the document text")).await.unwrap_err();
        assert!(e.to_string().contains("quoting the document text"), "the JoinError carries it: {e}");
        let status = batch_failed(&e);
        assert_eq!(status.code(), Code::Internal, "{status:?}");
        assert!(status.message().contains("a batch task panicked"), "{status:?}");
        assert!(!status.message().contains("quoting the document"), "{status:?}");
    }

    #[tokio::test]
    async fn a_worker_that_ends_well_adds_nothing_to_its_stream() {
        let (reply, mut answers) = mpsc::channel::<std::result::Result<AnalyzeStreamResponse, Status>>(4);
        let dead = Dead::new();
        watch(tokio::spawn(async {}), "the stream's scheduler", reply, dead.clone());
        assert!(answers.recv().await.is_none());
        assert!(!dead.is_set());
    }

    #[tokio::test]
    async fn a_blocking_task_that_panics_ends_its_reply_internal() {
        let (reply, mut answers) = mpsc::channel::<std::result::Result<u32, Status>>(4);
        let done = tokio::task::spawn_blocking(move || {
            let owned = reply.clone();
            reply_on_panic(reply, "the progressive analysis task", move || {
                let _ = owned.blocking_send(Ok(1));
                panic!("an export helper gave up, quoting the document");
            })
        });
        assert_eq!(answers.recv().await.unwrap().unwrap(), 1);
        let status = answers.recv().await.expect("a status, not a clean end").unwrap_err();
        assert_eq!(status.code(), Code::Internal, "{status:?}");
        assert!(status.message().contains("the progressive analysis task panicked"), "{status:?}");
        assert!(!status.message().contains("quoting the document"), "{status:?}");
        assert!(answers.recv().await.is_none());
        assert_eq!(done.await.unwrap(), None::<()>);
    }

    #[tokio::test]
    async fn a_blocking_task_that_returns_hands_back_its_result() {
        let (reply, mut answers) = mpsc::channel::<std::result::Result<u32, Status>>(4);
        let done = tokio::task::spawn_blocking(move || reply_on_panic(reply, "the task", || 7));
        assert_eq!(done.await.unwrap(), Some(7));
        assert!(answers.recv().await.is_none());
    }
}
