//! A StreamingTraining component job: a neural component trained from the
//! samples the client streams (docs/grpc-training.md, Streamed training).
//! The settings are checked, then the job takes the training slot, reads
//! every sample, trains with each epoch streamed as progress, and sends
//! the model file back in chunks. The server keeps nothing; a client that
//! goes away stops the training at its next epoch.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Status, Streaming};
use turbo_train_components::depparse::{
    Activation, FeatureLayout, FeedforwardSettings, Tagset, train_feedforward_conllu_observed,
};

use super::component::{MAX_SAMPLE_BYTES, MODEL_CHUNK_BYTES};
use super::stream::{Tx, UpdateStream, cancelled, internal, next, progress, update};
use super::train::Slot;
use super::{invalid, plain};
use crate::api::{Failure, TURBO_E_CAPACITY};
use crate::proto::training::streaming_training_request::Part;
use crate::proto::training::streaming_training_update::Update;
use crate::proto::training::{
    ConlluTagset, DependencyActivation, DependencyFeatureLayout, NeuralComponentKind, NeuralComponentTraining,
    StreamingStage, StreamingTrainingProgress, StreamingTrainingRequest, TrainedNeuralComponent,
};

/// The most epochs one job runs.
const MAX_EPOCHS: u32 = 10_000;

/// A job with its checks passed.
struct Plan {
    kind: NeuralComponentKind,
    tagset: Tagset,
    settings: FeedforwardSettings,
}

/// The job's checks, with the trainer's default wherever a field is 0 or
/// unset. A refusal names the job's field.
fn plan(t: NeuralComponentTraining) -> Result<Plan, Status> {
    let kind = NeuralComponentKind::try_from(t.component).unwrap_or(NeuralComponentKind::Unspecified);
    if kind == NeuralComponentKind::Unspecified {
        return Err(invalid(1, "component: name the component to train"));
    }
    let tagset = match ConlluTagset::try_from(t.tagset) {
        Ok(ConlluTagset::Upos) => Tagset::U,
        Ok(ConlluTagset::Xpos) => Tagset::X,
        Err(_) => return Err(invalid(2, format!("tagset {}: not a ConlluTagset", t.tagset))),
    };
    let layout = match DependencyFeatureLayout::try_from(t.layout) {
        Ok(DependencyFeatureLayout::ChenManning18) => FeatureLayout::ChenManning18,
        Ok(DependencyFeatureLayout::Compact14) => FeatureLayout::OpenNlp14,
        Err(_) => return Err(invalid(12, format!("layout {}: not a DependencyFeatureLayout", t.layout))),
    };
    let activation = match DependencyActivation::try_from(t.activation) {
        Ok(DependencyActivation::Cube) => Activation::Cube,
        Ok(DependencyActivation::Relu) => Activation::Relu,
        Err(_) => return Err(invalid(13, format!("activation {}: not a DependencyActivation", t.activation))),
    };
    if t.epochs > MAX_EPOCHS {
        return Err(invalid(5, format!("epochs {} is over {MAX_EPOCHS}", t.epochs)));
    }
    let d = FeedforwardSettings::default();
    let or = |n: u32, default: usize| if n == 0 { default } else { n as usize };
    let settings = FeedforwardSettings {
        embedding_size: or(t.embedding_size, d.embedding_size),
        hidden_size: or(t.hidden_size, d.hidden_size),
        epochs: or(t.epochs, d.epochs),
        batch_size: or(t.batch_size, d.batch_size),
        learning_rate: if t.learning_rate == 0.0 { d.learning_rate } else { t.learning_rate },
        l2: t.l2.unwrap_or(d.l2),
        dropout: t.dropout.unwrap_or(d.dropout),
        word_cutoff: t.word_cutoff.map_or(d.word_cutoff, |n| n as usize),
        seed: t.seed.unwrap_or(d.seed),
        layout,
        activation,
    };
    // The trainer's own bounds on the widths, the rate, the penalty and
    // the dropout, before anything is read.
    settings.check().map_err(|e| invalid(0, format!("settings: {e}")))?;
    Ok(Plan { kind, tagset, settings })
}

/// Checks the job, takes the slot, then reads and trains.
pub(super) async fn start(
    t: NeuralComponentTraining,
    mut requests: Streaming<StreamingTrainingRequest>,
    training: Arc<AtomicBool>,
    idle: Duration,
) -> Result<UpdateStream, Status> {
    let plan = plan(t)?;
    let slot = Slot::take(training)?;
    let (tx, rx) = mpsc::channel(16);
    let _ = tx.send(Ok(progress(StreamingStage::Admitted, 0, 0))).await;
    tokio::spawn(async move {
        let out = match receive(&mut requests, &tx, idle).await {
            Ok(samples) => {
                let t = tx.clone();
                tokio::task::spawn_blocking(move || train(&plan, &samples, &t))
                    .await
                    .unwrap_or_else(|e| Err(internal(e)))
            }
            Err(e) => Err(e),
        };
        // The slot is free before the model is sent, so a client that has
        // the model can start the next training.
        drop(slot);
        match out {
            Ok((model, trained)) => {
                for chunk in model.chunks(MODEL_CHUNK_BYTES) {
                    if tx.send(Ok(update(Update::Model(chunk.to_vec())))).await.is_err() {
                        return;
                    }
                }
                let _ = tx.send(Ok(update(Update::Component(trained)))).await;
            }
            Err(e) => {
                let _ = tx.send(Err(e)).await;
            }
        }
    });
    Ok(Box::pin(ReceiverStream::new(rx)))
}

/// Every sample byte after the start, up to [`MAX_SAMPLE_BYTES`].
async fn receive(
    requests: &mut Streaming<StreamingTrainingRequest>,
    tx: &Tx,
    idle: Duration,
) -> Result<Vec<u8>, Status> {
    let mut samples = Vec::new();
    while let Some(part) = next(requests, tx, idle).await? {
        let Part::Samples(b) = part else {
            return Err(invalid(0, "a component job takes samples, not documents"));
        };
        if samples.len() + b.len() > MAX_SAMPLE_BYTES {
            return Err(plain(Failure::new(
                TURBO_E_CAPACITY,
                0,
                format!("samples: more than {MAX_SAMPLE_BYTES} bytes"),
            )));
        }
        samples.extend_from_slice(&b);
    }
    Ok(samples)
}

/// The CoNLL-U sentences of `text`: each run of lines with no blank line
/// in it, as the text has it.
fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut at, mut open) = (0, None);
    for line in text.split_inclusive('\n') {
        match (line.trim().is_empty(), open) {
            (false, None) => open = Some(at),
            (true, Some(start)) => {
                out.push(text[start..at].trim_end());
                open = None;
            }
            _ => {}
        }
        at += line.len();
    }
    if let Some(start) = open {
        out.push(text[start..].trim_end());
    }
    out
}

/// Reads the samples, trains and packages the model.
fn train(plan: &Plan, samples: &[u8], tx: &Tx) -> Result<(Vec<u8>, TrainedNeuralComponent), Status> {
    let text = std::str::from_utf8(samples)
        .map_err(|e| invalid(0, format!("samples: not UTF-8 at byte {}", e.valid_up_to())))?;
    let sentences = sentences(text);
    if sentences.is_empty() {
        return Err(invalid(0, "samples: none were read"));
    }
    let _ = tx.blocking_send(Ok(progress(StreamingStage::Read, sentences.len() as u64, 0)));
    let model = train_feedforward_conllu_observed(&sentences, plan.tagset, &plan.settings, &mut |e| {
        let p = StreamingTrainingProgress {
            stage: StreamingStage::Epoch as i32,
            completed_items: e.epoch as u64,
            total_items: e.epochs as u64,
            loss: e.loss,
            ..Default::default()
        };
        // A client that has gone is seen here, and the training stops.
        tx.blocking_send(Ok(update(Update::Progress(p)))).is_ok()
    });
    if tx.is_closed() {
        return Err(cancelled());
    }
    let model = model
        .and_then(|m| m.to_bytes().map_err(|e| e.to_string()))
        .map_err(|e| invalid(0, format!("training: {e}")))?;
    let _ = tx.blocking_send(Ok(progress(StreamingStage::Packaged, 0, 0)));
    let trained = TrainedNeuralComponent {
        component: plan.kind as i32,
        samples: sentences.len() as u64,
        size: model.len() as u64,
        sha256: turbo_maxent::sha256(&model).to_hex(),
    };
    Ok((model, trained))
}

#[cfg(test)]
mod tests {
    use super::sentences;

    #[test]
    fn sentences_are_the_runs_between_blank_lines() {
        assert_eq!(sentences(""), Vec::<&str>::new());
        assert_eq!(sentences("\n \n\r\n"), Vec::<&str>::new());
        assert_eq!(sentences("a\nb\n\nc"), ["a\nb", "c"]);
        assert_eq!(sentences("\n\n# x\r\n1\tu\r\n\r\n \t\n2\tv\n\n\n"), ["# x\r\n1\tu", "2\tv"]);
    }
}
