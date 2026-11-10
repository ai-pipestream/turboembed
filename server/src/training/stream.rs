//! StreamingTraining: one stream for the trainings a client feeds
//! (docs/grpc-training.md, Streamed training). The start names the job: a
//! corpus job (corpus.rs) or a neural component job (neural.rs). Either
//! is checked, takes the one training slot the other trainings use, reads
//! the client's upload to its end, and streams progress for each stage,
//! then its own last message.

use std::pin::Pin;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::Stream;
use tonic::{Status, Streaming};

use super::{TrainingService, corpus, invalid, neural, plain, runtime};
use crate::api::{Failure, TURBO_E_CAPACITY, TURBO_E_INTERNAL};
use crate::proto::training::streaming_training_request::Part;
use crate::proto::training::streaming_training_start::Job;
use crate::proto::training::streaming_training_update::Update;
use crate::proto::training::{
    StreamingStage, StreamingTrainingProgress, StreamingTrainingRequest, StreamingTrainingUpdate,
};

pub(crate) type UpdateStream = Pin<Box<dyn Stream<Item = Result<StreamingTrainingUpdate, Status>> + Send>>;

pub(super) type Tx = mpsc::Sender<Result<StreamingTrainingUpdate, Status>>;

pub(super) fn update(u: Update) -> StreamingTrainingUpdate {
    StreamingTrainingUpdate { update: Some(u) }
}

/// A stage's progress: `completed` of `total` items, 0 where a stage has
/// no total.
pub(super) fn progress(stage: StreamingStage, completed: u64, total: u64) -> StreamingTrainingUpdate {
    update(Update::Progress(StreamingTrainingProgress {
        stage: stage as i32,
        completed_items: completed,
        total_items: total,
        ..Default::default()
    }))
}

pub(super) fn cancelled() -> Status {
    runtime("the client cancelled the training")
}

pub(super) fn internal(e: impl std::fmt::Display) -> Status {
    plain(Failure::new(TURBO_E_INTERNAL, 0, format!("the training's task failed: {e}")))
}

/// The upload's next part, None at its end. The wait ends when the answer
/// is gone or when `idle` passes with no message: a client that drops the
/// answer while it holds its upload open is not seen by the server, so the
/// slot is held at most that long. A second start is refused.
pub(super) async fn next(
    requests: &mut Streaming<StreamingTrainingRequest>,
    tx: &Tx,
    idle: Duration,
) -> Result<Option<Part>, Status> {
    loop {
        let m = tokio::select! {
            m = tokio::time::timeout(idle, requests.message()) => m.map_err(|_| {
                let secs = idle.as_secs_f64();
                plain(Failure::new(TURBO_E_CAPACITY, 0, format!("upload: nothing came for {secs} s; it is dropped")))
            })??,
            () = tx.closed() => return Err(cancelled()),
        };
        match m {
            None => return Ok(None),
            Some(StreamingTrainingRequest { part: Some(Part::Start(_)) }) => {
                return Err(invalid(0, "a second start: only the first message is the start"));
            }
            Some(StreamingTrainingRequest { part: Some(part) }) => return Ok(Some(part)),
            Some(StreamingTrainingRequest { part: None }) => {}
        }
    }
}

/// Reads the start and hands the stream to its job. Every refusal before
/// the first message is the call's own status.
pub(super) async fn start(
    service: &TrainingService,
    mut requests: Streaming<StreamingTrainingRequest>,
) -> Result<UpdateStream, Status> {
    let first = requests.message().await?.ok_or_else(|| invalid(0, "the stream ended before its start"))?;
    let Some(Part::Start(start)) = first.part else {
        return Err(invalid(0, "the first message must be the start"));
    };
    match start.job {
        Some(Job::Corpus(c)) => corpus::start(service, c, requests).await,
        Some(Job::Component(c)) => neural::start(c, requests, service.training.clone(), service.upload_idle).await,
        None => Err(invalid(0, "start: name a corpus job or a component job")),
    }
}
