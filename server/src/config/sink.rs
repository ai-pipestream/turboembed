//! The sink's flags and their variables (docs/grpc-sink.md): the endpoint,
//! the queue bounds, forwarding AnalyzeDocumentProgressive graphs, and the
//! typed projection. Every one but the endpoint needs the endpoint.

use super::refused;
use crate::proto::result::OffsetMode;
use crate::results::disclosure::Disclosure;
use crate::sink::{DEFAULT_SINK_QUEUE_BYTES, DEFAULT_SINK_QUEUE_ITEMS, SinkConfig, SinkTyped};

pub const ENV_SINK_ENDPOINT: &str = "TURBO_KSERVE_SINK_ENDPOINT";
pub const ENV_SINK_QUEUE_ITEMS: &str = "TURBO_KSERVE_SINK_QUEUE_ITEMS";
pub const ENV_SINK_QUEUE_BYTES: &str = "TURBO_KSERVE_SINK_QUEUE_BYTES";
pub const ENV_SINK_DOCUMENTS: &str = "TURBO_KSERVE_SINK_DOCUMENTS";
pub const ENV_SINK_TYPED_OFFSETS: &str = "TURBO_KSERVE_SINK_TYPED_OFFSETS";
pub const ENV_SINK_TYPED_VIEW: &str = "TURBO_KSERVE_SINK_TYPED_VIEW";
pub const ENV_SINK_PII_OUTPUT: &str = "TURBO_KSERVE_SINK_PII_OUTPUT";
pub const ENV_SINK_SEARCH_PROJECTION: &str = "TURBO_KSERVE_SINK_SEARCH_PROJECTION";

/// The sink flags as given on the command line, before their variables.
#[derive(Default)]
pub(super) struct SinkFlags {
    endpoint: Option<String>,
    items: Option<String>,
    bytes: Option<String>,
    documents: Option<bool>,
    search_projection: Option<bool>,
    typed_offsets: Option<String>,
    typed_view: Option<String>,
    pii_output: Option<String>,
}

impl SinkFlags {
    /// Takes `flag` if it is a sink flag: `inline` is its `=value`, and
    /// `value` reads the value the flag needs. Answers false for any other
    /// flag.
    pub(super) fn take(
        &mut self,
        flag: &str,
        inline: bool,
        value: impl FnOnce() -> Result<String, String>,
    ) -> Result<bool, String> {
        let slot = match flag {
            "--sink-documents" => {
                if inline || self.documents.replace(true).is_some() {
                    return Err("--sink-documents takes no value and may be given once".into());
                }
                return Ok(true);
            }
            "--sink-search-projection" => {
                if inline || self.search_projection.replace(true).is_some() {
                    return Err("--sink-search-projection takes no value and may be given once".into());
                }
                return Ok(true);
            }
            "--sink-endpoint" => &mut self.endpoint,
            "--sink-queue-items" => &mut self.items,
            "--sink-queue-bytes" => &mut self.bytes,
            "--sink-typed-offsets" => &mut self.typed_offsets,
            "--sink-typed-view" => &mut self.typed_view,
            "--sink-pii-output" => &mut self.pii_output,
            _ => return Ok(false),
        };
        super::once(flag, value()?, slot)?;
        Ok(true)
    }

    /// The sink settings, with `env` filling in what the flags leave out.
    pub(super) fn resolve(
        self,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<(Option<SinkConfig>, Option<SinkTyped>), String> {
        let from_env = |slot: Option<String>, k: &str| slot.or_else(|| env(k).filter(|v| !v.is_empty()));
        let items = from_env(self.items, ENV_SINK_QUEUE_ITEMS);
        let bytes = from_env(self.bytes, ENV_SINK_QUEUE_BYTES);
        let typed_offsets = from_env(self.typed_offsets, ENV_SINK_TYPED_OFFSETS);
        let typed_view = from_env(self.typed_view, ENV_SINK_TYPED_VIEW);
        let pii_output = from_env(self.pii_output, ENV_SINK_PII_OUTPUT);
        let parse_bool = |flag: Option<bool>, var: &str| -> Result<bool, String> {
            match (flag, env(var).filter(|v| !v.is_empty())) {
                (Some(on), _) => Ok(on),
                (None, None) => Ok(false),
                (None, Some(v)) => match v.as_str() {
                    "1" | "true" | "yes" => Ok(true),
                    "0" | "false" | "no" => Ok(false),
                    _ => Err(refused(var, &v, super::refused::SWITCH)),
                },
            }
        };
        let documents = parse_bool(self.documents, ENV_SINK_DOCUMENTS)?;
        let search_projection = parse_bool(self.search_projection, ENV_SINK_SEARCH_PROJECTION)?;
        let Some(endpoint) = from_env(self.endpoint, ENV_SINK_ENDPOINT) else {
            if items.is_some() || bytes.is_some() {
                return Err("--sink-queue-items and --sink-queue-bytes need --sink-endpoint".into());
            }
            if documents || search_projection || typed_offsets.is_some() || typed_view.is_some() || pii_output.is_some()
            {
                return Err(
                    "--sink-documents, --sink-search-projection, --sink-typed-offsets, --sink-typed-view and --sink-pii-output need --sink-endpoint"
                        .into(),
                );
            }
            return Ok((None, None));
        };
        let count = |name: &str, v: Option<String>, default: u64| -> Result<u64, String> {
            match v {
                None => Ok(default),
                Some(v) => match v.parse::<u64>() {
                    Ok(n) if n > 0 => Ok(n),
                    _ => Err(refused(name, &v, "a count over 0")),
                },
            }
        };
        let typed = match (typed_offsets, typed_view) {
            (None, None) => None,
            (None, Some(_)) => return Err("--sink-typed-view needs --sink-typed-offsets".into()),
            (Some(offsets), view) => Some(SinkTyped { view: typed_view_of(view)?, offsets: offset_mode(&offsets)? }),
        };
        let sp = search_projection.then(|| typed.unwrap_or(SinkTyped { view: 0, offsets: OffsetMode::Utf8 }));
        Ok((
            Some(SinkConfig {
                endpoint,
                max_queue_items: count("--sink-queue-items", items, DEFAULT_SINK_QUEUE_ITEMS as u64)? as usize,
                max_queue_bytes: count("--sink-queue-bytes", bytes, DEFAULT_SINK_QUEUE_BYTES)?,
                documents,
                typed,
                pii_output: match pii_output {
                    None => Disclosure::FINDINGS,
                    Some(v) => Disclosure::parse(&v).map_err(|e| format!("--sink-pii-output: {e}"))?,
                },
            }),
            sp,
        ))
    }
}

fn offset_mode(v: &str) -> Result<OffsetMode, String> {
    match v {
        "utf8" => Ok(OffsetMode::Utf8),
        "utf16" => Ok(OffsetMode::Utf16),
        "code-point" => Ok(OffsetMode::CodePoint),
        _ => Err(refused("--sink-typed-offsets", v, "one of utf8, utf16, code-point")),
    }
}

/// A view index a document can have; 0, the source view, when not given.
fn typed_view_of(v: Option<String>) -> Result<u32, String> {
    let Some(v) = v else { return Ok(0) };
    match v.parse::<u16>() {
        Ok(n) => Ok(u32::from(n)),
        Err(_) => Err(refused("--sink-typed-view", &v, "a view index from 0 to 65535")),
    }
}
