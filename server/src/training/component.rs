//! TrainComponent: a classical component trained from samples the client
//! streams in their native format (docs/grpc-training.md, Component
//! training). The start is checked, then the call takes the one training
//! slot TrainStaticModel uses, reads every sample, trains with each
//! iteration streamed as progress, and sends the model file back in
//! chunks. The server keeps nothing; a client that goes away stops the
//! training at its next iteration.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Status, Streaming};
use turbo_formats::error::FormatError;
use turbo_formats::native::{
    ChunkReader, DocumentReader, LanguageReader, LemmaReader, NameReader, SentenceReader, TokenReader, WordTagReader,
};
use turbo_sequence::namefind::NameCodec;
use turbo_train::{
    Algorithm, IterDeltaAccuracyUnderTolerance, Key, LogLikelihoodThresholdBreached, ModelPackage, ProgressMonitor,
    StopCriteria, TrainingConfiguration, TrainingMeasure, TrainingParameters,
};
use turbo_train_components as tc;

use super::{invalid, plain, runtime};
use crate::api::{Failure, TURBO_E_BUSY, TURBO_E_CAPACITY, TURBO_E_INTERNAL};
use crate::proto::training::train_component_request::Part;
use crate::proto::training::train_component_update::Update;
use crate::proto::training::{
    ComponentKind, ComponentStage, TrainComponentProgress, TrainComponentRequest, TrainComponentStart,
    TrainComponentUpdate, TrainedComponent,
};

pub(crate) type ComponentStream = Pin<Box<dyn Stream<Item = Result<TrainComponentUpdate, Status>> + Send>>;

type Tx = mpsc::Sender<Result<TrainComponentUpdate, Status>>;

/// The most sample bytes one training reads; they are held in memory.
pub(crate) const MAX_SAMPLE_BYTES: usize = 256 << 20;

/// The model file is sent in chunks of at most this many bytes.
pub(crate) const MODEL_CHUNK_BYTES: usize = 1 << 20;

/// The longest language code a start may name, in bytes.
const MAX_LANGUAGE: usize = 64;

/// The training slot, given back when dropped.
struct Slot(Arc<AtomicBool>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// A start with its checks passed.
struct Plan {
    kind: ComponentKind,
    language: String,
    params: TrainingParameters,
    name_type: Option<String>,
}

fn progress(stage: ComponentStage) -> TrainComponentProgress {
    TrainComponentProgress { stage: stage as i32, ..Default::default() }
}

fn update(u: Update) -> TrainComponentUpdate {
    TrainComponentUpdate { update: Some(u) }
}

fn plan(s: TrainComponentStart) -> Result<Plan, Status> {
    let kind = ComponentKind::try_from(s.component).unwrap_or(ComponentKind::Unspecified);
    if kind == ComponentKind::Unspecified {
        return Err(invalid(1, "component: name the component to train"));
    }
    if s.language.len() > MAX_LANGUAGE || s.language.chars().any(char::is_control) {
        return Err(invalid(2, format!("language: at most {MAX_LANGUAGE} bytes, no control characters")));
    }
    if s.language.is_empty() && kind != ComponentKind::LanguageDetector {
        return Err(invalid(2, "language: the model records its language; name it"));
    }
    let mut params = TrainingParameters::new();
    for (k, v) in &s.parameters {
        params.put_str(k, v);
    }
    // The name finder puts its own defaults in (namefind.rs).
    if kind != ComponentKind::NameFinder {
        for (k, v) in TrainingParameters::defaults().iter() {
            params.put_if_absent(k, v.clone());
        }
    }
    let name_type = (!s.name_type.is_empty()).then_some(s.name_type);
    Ok(Plan { kind, language: s.language, params, name_type })
}

/// Checks the start, takes the slot, then reads and trains.
pub(super) async fn start(
    mut requests: Streaming<TrainComponentRequest>,
    training: Arc<AtomicBool>,
) -> Result<ComponentStream, Status> {
    let first = requests.message().await?.ok_or_else(|| invalid(0, "the stream ended before its start"))?;
    let Some(Part::Start(s)) = first.part else {
        return Err(invalid(0, "the first message must be the start"));
    };
    let plan = plan(s)?;
    if training.swap(true, Ordering::AcqRel) {
        return Err(plain(Failure::new(TURBO_E_BUSY, 0, "another training is running; try again when it ends")));
    }
    let slot = Slot(training);
    let (tx, rx) = mpsc::channel(16);
    let _ = tx.send(Ok(update(Update::Progress(progress(ComponentStage::Admitted))))).await;
    tokio::spawn(async move {
        let out = match receive(&mut requests).await {
            Ok(samples) => {
                let t = tx.clone();
                match tokio::task::spawn_blocking(move || train(&plan, samples, &t)).await {
                    Ok(out) => out,
                    Err(e) => Err(plain(Failure::new(TURBO_E_INTERNAL, 0, format!("the training's task failed: {e}")))),
                }
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
                let _ = tx.send(Ok(update(Update::Trained(trained)))).await;
            }
            Err(e) => {
                let _ = tx.send(Err(e)).await;
            }
        }
    });
    Ok(Box::pin(ReceiverStream::new(rx)))
}

/// Every sample byte after the start, up to [`MAX_SAMPLE_BYTES`].
async fn receive(requests: &mut Streaming<TrainComponentRequest>) -> Result<Vec<u8>, Status> {
    let mut samples = Vec::new();
    while let Some(m) = requests.message().await? {
        match m.part {
            Some(Part::Samples(b)) => {
                if samples.len() + b.len() > MAX_SAMPLE_BYTES {
                    return Err(plain(Failure::new(
                        TURBO_E_CAPACITY,
                        2,
                        format!("samples: more than {MAX_SAMPLE_BYTES} bytes"),
                    )));
                }
                samples.extend_from_slice(&b);
            }
            Some(Part::Start(_)) => return Err(invalid(1, "a second start: only the first message is the start")),
            None => {}
        }
    }
    Ok(samples)
}

/// Each finished iteration as progress.
struct Monitor {
    tx: Tx,
    finished: bool,
}

impl ProgressMonitor for Monitor {
    fn finished_iteration(&mut self, iteration: i32, correct: i32, total: i32, measure: TrainingMeasure, value: f64) {
        let p = TrainComponentProgress {
            iteration: iteration.max(0) as u32,
            correct: correct.max(0) as u64,
            events: total.max(0) as u64,
            measure: measure.name().into(),
            value,
            ..progress(ComponentStage::Iteration)
        };
        let _ = self.tx.blocking_send(Ok(update(Update::Progress(p))));
    }

    fn finished_training(&mut self, _: i32, _: Option<&dyn StopCriteria>) {
        self.finished = true;
    }

    fn is_training_finished(&self) -> bool {
        self.finished
    }

    fn display(&mut self, _: bool) {}
}

/// The trainer's own stop criterion, and a stop when the client has gone.
struct Cancel {
    tx: Tx,
    inner: Option<Box<dyn StopCriteria>>,
}

impl StopCriteria for Cancel {
    fn test(&self, value: f64) -> bool {
        self.tx.is_closed() || self.inner.as_ref().is_some_and(|s| s.test(value))
    }

    fn message_if_satisfied(&self) -> String {
        match &self.inner {
            Some(s) if !self.tx.is_closed() => s.message_if_satisfied(),
            _ => "the client cancelled the training".into(),
        }
    }
}

/// The criterion the parameters' algorithm stops with when none is given:
/// GIS's log likelihood threshold, the perceptron's accuracy tolerance.
fn default_stop(params: &TrainingParameters) -> Result<Option<Box<dyn StopCriteria>>, Status> {
    let bad = |e: turbo_train::Error| invalid(3, format!("parameters: {e}"));
    let algorithm = match params.get_str(Key::ALGORITHM).map_err(bad)? {
        None => Algorithm::Maxent,
        Some(name) => Algorithm::from_name(name).map_err(bad)?,
    };
    Ok(match algorithm {
        Algorithm::Maxent => Some(Box::new(LogLikelihoodThresholdBreached::from_parameters(params).map_err(bad)?)),
        Algorithm::Perceptron => Some(Box::new(IterDeltaAccuracyUnderTolerance::from_parameters(params).map_err(bad)?)),
        _ => None,
    })
}

/// The samples of `text` as `R` reads them; a refused line refuses them all.
fn read<S, R: Iterator<Item = Result<S, FormatError>>>(reader: Result<R, FormatError>) -> Result<Vec<S>, Status> {
    let samples =
        reader.and_then(|r| r.collect::<Result<Vec<_>, _>>()).map_err(|e| invalid(2, format!("samples: {e}")))?;
    if samples.is_empty() {
        return Err(invalid(2, "samples: none were read"));
    }
    Ok(samples)
}

/// Reads the samples, trains and packages the model.
fn train(plan: &Plan, samples: Vec<u8>, tx: &Tx) -> Result<(Vec<u8>, TrainedComponent), Status> {
    let text = std::str::from_utf8(&samples)
        .map_err(|e| invalid(2, format!("samples: not UTF-8 at byte {}", e.valid_up_to())))?;
    let config = &mut TrainingConfiguration::new(
        Some(Box::new(Monitor { tx: tx.clone(), finished: false })),
        Some(Box::new(Cancel { tx: tx.clone(), inner: default_stop(&plan.params)? })),
    );
    let read_progress = |n: usize| {
        let p = TrainComponentProgress { samples: n as u64, ..progress(ComponentStage::Read) };
        let _ = tx.blocking_send(Ok(update(Update::Progress(p))));
        n
    };
    let (lang, params) = (plan.language.as_str(), &plan.params);
    let stamp = tc::Stamp::at(SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64));
    let failed = |e: tc::Error| invalid(0, format!("training: {e}"));
    let (count, package): (usize, ModelPackage) = match plan.kind {
        ComponentKind::SentenceDetector => {
            let s = read(SentenceReader::new(text))?;
            let f = tc::SentenceDetectorFactory::new(lang, true, None, None).map_err(failed)?;
            (read_progress(s.len()), tc::train_sentence_model(&s, &f, params, config, stamp).map_err(failed)?.package)
        }
        ComponentKind::Tokenizer => {
            let s = read(TokenReader::new(text))?;
            let f = tc::TokenizerFactory::new(lang, None, false, None).map_err(failed)?;
            (read_progress(s.len()), tc::train_tokenizer_model(&s, &f, params, config, stamp).map_err(failed)?.package)
        }
        ComponentKind::PosTagger => {
            let s = read(WordTagReader::new(text))?;
            let f = tc::PosTaggerFactory::new(lang, None, None).map_err(failed)?;
            (read_progress(s.len()), tc::train_pos_model(&s, &f, params, config, stamp).map_err(failed)?.package)
        }
        ComponentKind::Lemmatizer => {
            let s = read(LemmaReader::new(text))?;
            let f = tc::LemmatizerFactory::new(lang);
            let trained = tc::train_lemmatizer_model(&s, &f, params, config, stamp);
            (read_progress(s.len()), trained.map_err(failed)?.package)
        }
        ComponentKind::Chunker => {
            let s = read(ChunkReader::new(text))?;
            let f = tc::ChunkerFactory::new(lang);
            (read_progress(s.len()), tc::train_chunker_model(&s, &f, params, config, stamp).map_err(failed)?.package)
        }
        ComponentKind::NameFinder => {
            let s = read(NameReader::new(text))?;
            let f = tc::TokenNameFinderFactory::new(lang, None, NameCodec::Bio).map_err(failed)?;
            let kind = plan.name_type.as_deref();
            let trained = tc::train_name_model(&s, &f, kind, params, config, stamp);
            (read_progress(s.len()), trained.map_err(failed)?.package)
        }
        ComponentKind::DocumentCategorizer => {
            let s = read(DocumentReader::new(text))?;
            let f = turbo_doccat::DoccatFactory::new();
            (read_progress(s.len()), tc::train_doccat(lang, &s, params, &f, config, stamp).map_err(failed)?.package)
        }
        ComponentKind::LanguageDetector => {
            let s = read(LanguageReader::new(text))?;
            let f = tc::LanguageDetectorFactory::new();
            (read_progress(s.len()), tc::train_language_model(&s, &f, params, config, stamp).map_err(failed)?.package)
        }
        ComponentKind::Unspecified => unreachable!("refused by plan"),
    };
    if tx.is_closed() {
        return Err(runtime("the client cancelled the training"));
    }
    let model = package.to_bytes();
    let _ = tx.blocking_send(Ok(update(Update::Progress(progress(ComponentStage::Packaged)))));
    let trained = TrainedComponent {
        component: plan.kind as i32,
        samples: count as u64,
        size: model.len() as u64,
        sha256: turbo_maxent::sha256(&model).to_hex(),
    };
    Ok((model, trained))
}
