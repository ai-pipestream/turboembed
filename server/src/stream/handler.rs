//! One AnalyzeStream call (issue #722): reads the header, refuses what
//! it cannot mean the way the capability listing advertises, and wires
//! the reader and the engine to the response stream. The response
//! channel is bounded by the window: a client that does not read stops
//! the engine, then the queue, then the reader, so memory stays under
//! the advertised caps all the way down.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use tokio::sync::{Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};
use tonic::{Response, Status, Streaming};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_ENUM, names};
use crate::kserve::Reply;
use crate::offsets::OffsetMode;
use crate::proto::result::analyze_stream_request::Kind;
use crate::proto::result::{AnalyzeOptions, AnalyzeOrder, AnalyzeOutput, AnalyzeStreamRequest, AnalyzeStreamResponse};
use crate::sink::SinkForwarder;
use crate::state::State;
use crate::status::to_status;
use crate::stream::engine::{self, Config};
use crate::stream::reader;
use crate::stream::{
    DEFAULT_BATCH_LINGER, MAX_BATCH_LINGER, MAX_INFLIGHT_ITEMS_PER_STREAM, MAX_QUEUED_BYTES_PER_STREAM,
    MAX_QUEUED_ITEMS_PER_STREAM, Shared,
};

/// The response stream of one AnalyzeStream call.
pub(crate) type AnswerStream = Pin<Box<dyn Stream<Item = std::result::Result<AnalyzeStreamResponse, Status>> + Send>>;

fn invalid(message: impl Into<String>) -> Status {
    to_status(&Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message), &[])
}

fn bad_enum(name: &str, value: u32, table: &[(u32, &str)]) -> Failure {
    let all: Vec<&str> = table.iter().map(|(_, n)| *n).collect();
    Failure::new(TURBO_E_INVALID_ENUM, 0, format!("{name} {value}: not one of {}", all.join(", "),))
}

/// Validated per-document options, output kind, and wire offset mode.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DocOptions {
    pub(crate) embed: turbo::turbo_embed_options,
    pub(crate) output: AnalyzeOutput,
    pub(crate) offsets: OffsetMode,
    pub(crate) sparse_top_k: u32,
    pub(crate) sparse_floor: f32,
    pub(crate) token_stride: u32,
}

/// Validates the output kind, offset mode, and options of one
/// `AnalyzeDocument`.
pub(crate) fn doc_options(output: i32, offsets: i32, o: Option<&AnalyzeOptions>) -> Result<DocOptions, Failure> {
    let output = AnalyzeOutput::try_from(output).map_err(|_| {
        Failure::new(
            TURBO_E_INVALID_ENUM,
            0,
            format!(
                "output {output}: not one of ANALYZE_OUTPUT_VECTORS, ANALYZE_OUTPUT_SPARSE_TERMS, ANALYZE_OUTPUT_TOKEN_VECTORS"
            ),
        )
    })?;
    let offsets = OffsetMode::from_proto(offsets)?;
    let embed = options(o)?;
    let raw = o.cloned().unwrap_or_default();
    if !raw.sparse_floor.is_finite() || raw.sparse_floor < 0.0 {
        return Err(Failure::new(
            TURBO_E_INVALID_ARGUMENT,
            0,
            format!("sparse_floor {}: must be a finite number >= 0", raw.sparse_floor),
        ));
    }
    Ok(DocOptions {
        embed,
        output,
        offsets,
        sparse_top_k: raw.sparse_top_k,
        sparse_floor: raw.sparse_floor,
        token_stride: raw.token_stride,
    })
}

/// The document's options as one turbo_embed_options, each field 0 or a
/// constant the header names, exactly what the single-document call's
/// parameters carry. An unknown value is the document's own refusal.
pub(crate) fn options(o: Option<&AnalyzeOptions>) -> Result<turbo::turbo_embed_options, Failure> {
    let o = o.cloned().unwrap_or_default();
    let pick = |name: &str, value: u32, table: &[(u32, &str)]| -> Result<u32, Failure> {
        if value == 0 || table.iter().any(|(v, _)| *v == value) { Ok(value) } else { Err(bad_enum(name, value, table)) }
    };
    Ok(turbo::turbo_embed_options {
        struct_size: size_of::<turbo::turbo_embed_options>() as u32,
        truncate: pick("truncate", o.truncate, names::TRUNCATE)?,
        max_tokens: o.max_tokens,
        prompt_role: pick("prompt_role", o.prompt_role, names::PROMPT)?,
        normalize: pick("normalize", o.normalize, names::NORMALIZE)?,
        pooling: pick("pooling", o.pooling, names::POOLING)?,
        output_dim: o.output_dim,
    })
}

/// The header's order and flow-control knobs, capped by what
/// ServerLimits.stream advertises.
fn config(header: crate::proto::result::AnalyzeStreamHeader) -> Result<Config, Status> {
    let order = AnalyzeOrder::try_from(header.order)
        .map_err(|_| invalid(format!("order {}: not ANALYZE_ORDER_INPUT or ANALYZE_ORDER_COMPLETION", header.order)))?;
    let asked_inflight =
        crate::stream::capped("max_inflight", header.max_inflight, MAX_INFLIGHT_ITEMS_PER_STREAM as u64)?;
    let asked_linger =
        crate::stream::capped("batch_linger_ms", header.batch_linger_ms, MAX_BATCH_LINGER.as_millis() as u64)?;
    Ok(Config {
        order,
        window: if asked_inflight == 0 { MAX_INFLIGHT_ITEMS_PER_STREAM } else { asked_inflight as usize },
        linger: if asked_linger == 0 { DEFAULT_BATCH_LINGER } else { Duration::from_millis(asked_linger) },
    })
}

/// Serves one AnalyzeStream call: reads the header, then hands the
/// rest to the reader and the engine, whose answers fill the bounded
/// response channel the client reads.
pub(crate) async fn start(
    state: Arc<State>,
    mut inbound: Streaming<AnalyzeStreamRequest>,
    sink: Option<Arc<SinkForwarder>>,
) -> Reply<AnswerStream> {
    let header = match inbound.message().await {
        Err(e) => return Err(e),
        Ok(None) => return Err(invalid("the stream's first message must be the header")),
        Ok(Some(crate::proto::result::AnalyzeStreamRequest { kind: Some(Kind::Header(h)) })) => h,
        Ok(Some(_)) => return Err(invalid("the stream's first message must be the header")),
    };
    let config = config(header)?;
    let window = config.window;
    let shared: Arc<Shared> = state.stream.clone();
    let stream_items = Arc::new(Semaphore::new(MAX_QUEUED_ITEMS_PER_STREAM));
    let stream_bytes = Arc::new(Semaphore::new(MAX_QUEUED_BYTES_PER_STREAM));
    let (work_tx, work_rx) = mpsc::channel(MAX_QUEUED_ITEMS_PER_STREAM);
    let (answers_tx, answers_rx) = mpsc::channel(window);
    let dead = Arc::new(AtomicBool::new(false));
    tokio::spawn(reader::run(
        state.clone(),
        shared.clone(),
        stream_items,
        stream_bytes,
        inbound,
        work_tx,
        dead.clone(),
    ));
    engine::start(shared, config, work_rx, answers_tx, dead);
    let answers = ReceiverStream::new(answers_rx);
    match sink {
        // Each answer is offered to the sink as the client is handed it.
        Some(sink) => Ok(Response::new(Box::pin(answers.map(move |a| {
            if let Ok(answer) = &a {
                sink.offer(answer);
            }
            a
        })))),
        None => Ok(Response::new(Box::pin(answers))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_default_to_the_model() {
        let o = options(None).unwrap();
        assert_eq!((o.truncate, o.max_tokens, o.prompt_role, o.normalize, o.pooling, o.output_dim), (0, 0, 0, 0, 0, 0));
        let o = options(Some(&AnalyzeOptions { truncate: 3, prompt_role: 2, ..Default::default() })).unwrap();
        assert_eq!((o.truncate, o.prompt_role), (3, 2));
    }

    #[test]
    fn an_unknown_option_value_is_the_documents_own_refusal() {
        let e = options(Some(&AnalyzeOptions { truncate: 9, ..Default::default() })).unwrap_err();
        assert_eq!(e.code, TURBO_E_INVALID_ENUM);
        assert!(e.message.contains("truncate"), "{e:?}");
        let e = options(Some(&AnalyzeOptions { pooling: 9, ..Default::default() })).unwrap_err();
        assert!(e.message.contains("pooling"), "{e:?}");
    }

    #[test]
    fn the_header_knobs_fall_back_to_the_defaults() {
        let c = config(crate::proto::result::AnalyzeStreamHeader { order: 0, max_inflight: 0, batch_linger_ms: 0 })
            .unwrap();
        assert_eq!(c.window, MAX_INFLIGHT_ITEMS_PER_STREAM);
        assert_eq!(c.linger, DEFAULT_BATCH_LINGER);
        let c = config(crate::proto::result::AnalyzeStreamHeader { order: 1, max_inflight: 1, batch_linger_ms: 5 })
            .unwrap();
        assert_eq!((c.window, c.linger), (1, Duration::from_millis(5)));
        let e = config(crate::proto::result::AnalyzeStreamHeader { order: 2, max_inflight: 0, batch_linger_ms: 0 })
            .unwrap_err();
        assert!(e.message().contains("order"), "{e:?}");
        let e = config(crate::proto::result::AnalyzeStreamHeader {
            order: 0,
            max_inflight: (MAX_INFLIGHT_ITEMS_PER_STREAM + 1) as u32,
            batch_linger_ms: 0,
        })
        .unwrap_err();
        assert!(e.message().contains("max_inflight"), "{e:?}");
    }
}
