//! The receiving side of TurboSinkService: each item checked, the
//! accepted ones handed to the receiver's callback in order, and one
//! summary when the sender closes the stream.

use std::fmt;
use std::sync::Arc;

use tonic::{Request, Response, Status, Streaming};
use turbo_analysis::export::{DocumentGraph, import};

use crate::kserve::Reply;
use crate::proto::sink::turbo_sink_service_server::TurboSinkService;
use crate::proto::sink::{DocumentSinkItem, DocumentSinkSummary};
use crate::results::MAX_GRAPH_BYTES;

/// The most rejection reasons a summary carries.
pub const MAX_SUMMARY_ERRORS: usize = 100;

type OnItem = dyn Fn(DocumentSinkItem) + Send + Sync;

/// A sink served in this server: `on_item` is called with every accepted
/// item, in the order its stream sent it.
#[derive(Clone)]
pub struct SinkReceiver {
    on_item: Arc<OnItem>,
}

impl fmt::Debug for SinkReceiver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SinkReceiver")
    }
}

impl SinkReceiver {
    pub fn new(on_item: impl Fn(DocumentSinkItem) + Send + Sync + 'static) -> SinkReceiver {
        SinkReceiver { on_item: Arc::new(on_item) }
    }
}

/// Why `item` is not accepted after an item of sequence `last`, if it is
/// not: a sequence not over the last, an item carrying nothing, or graph
/// bytes that do not import.
fn check(item: &DocumentSinkItem, last: Option<u64>) -> Result<(), String> {
    if last.is_some_and(|l| item.sequence <= l) {
        return Err(format!("sequence {} is not over the last, {}", item.sequence, last.unwrap_or(0)));
    }
    if item.graph.is_empty() && item.analyzed.is_none() {
        return Err("carries neither a graph nor an answer".into());
    }
    if !item.graph.is_empty() {
        if item.graph.len() as u64 > MAX_GRAPH_BYTES {
            return Err(format!("graph of {} bytes is over max_graph_bytes {MAX_GRAPH_BYTES}", item.graph.len()));
        }
        let graph = DocumentGraph::from_bytes(&item.graph).map_err(|e| format!("document graph: {e}"))?;
        import(&graph, crate::results::document::codecs()).map_err(|e| format!("document graph import: {e}"))?;
    }
    Ok(())
}

#[tonic::async_trait]
impl TurboSinkService for SinkReceiver {
    async fn stream_documents(&self, req: Request<Streaming<DocumentSinkItem>>) -> Reply<DocumentSinkSummary> {
        let mut inbound = req.into_inner();
        let mut summary = DocumentSinkSummary::default();
        let mut last = None;
        while let Some(item) = inbound.message().await? {
            summary.received_count += 1;
            // Importing a graph walks the whole document; keep the async
            // runtime free of it.
            let (item, checked) = tokio::task::spawn_blocking(move || {
                let r = check(&item, last);
                (item, r)
            })
            .await
            .map_err(|e| Status::internal(format!("the sink's check failed: {e}")))?;
            match checked {
                Ok(()) => {
                    last = Some(item.sequence);
                    summary.accepted_count += 1;
                    (self.on_item)(item);
                }
                Err(why) => {
                    summary.rejected_count += 1;
                    if summary.errors.len() < MAX_SUMMARY_ERRORS {
                        summary.errors.push(format!("item {}: {why}", item.sequence));
                    }
                }
            }
        }
        Ok(Response::new(summary))
    }
}
