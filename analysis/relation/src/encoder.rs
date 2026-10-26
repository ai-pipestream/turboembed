//! The DeBERTa-v3 encoder that the GLiNER2 and GLiREL heads read (#920).
//! It is either the CPU encoder itself (`turbo::deberta::Encoder`) or a
//! core bundle of the DeBERTa-v2 family on the device that
//! `EncoderOptions` names (cpu, cuda, levelzero), run through the backend
//! table as GLiNER's is (#883).
//!
//! The heads read the encoder's output at a few positions only: each
//! word's first subword and the label markers. [`EncoderStates::states`]
//! gives those rows and nothing else. On a device they are one-token
//! ranges of the range write, so only the rows the heads read come back
//! to the host.

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::deberta::{Config, Encoder};
use turbo::manifest::Family;
use turbo::turbo_token_range;
use turbo_analysis_embed::Rows;
pub use turbo_analysis_embed::{Device, EncoderOptions, Precision};

/// An encoder the heads read: its shape, and its output at positions.
pub trait EncoderStates {
    fn config(&self) -> &Config;

    /// The encoder's output over `ids` at each of `at`, in order, into
    /// `out` [at, hidden]. Every position is inside `ids`.
    fn states(&self, ids: &[u32], at: &[usize], out: &mut Vec<f32>) -> Result<(), String>;
}

impl EncoderStates for Encoder {
    fn config(&self) -> &Config {
        Encoder::config(self)
    }

    fn states(&self, ids: &[u32], at: &[usize], out: &mut Vec<f32>) -> Result<(), String> {
        let hidden = self.encode(ids, &vec![true; ids.len()]).map_err(|e| e.message)?;
        let (data, d) = (hidden.data(), hidden.width());
        out.clear();
        out.extend(at.iter().flat_map(|&p| data[p * d..(p + 1) * d].iter().copied()));
        Ok(())
    }
}

/// The CPU encoder, or a core bundle on a device.
pub enum DebertaEncoder {
    Cpu(Box<Encoder>),
    Table(Box<Table>),
}

impl From<Encoder> for DebertaEncoder {
    fn from(e: Encoder) -> Self {
        DebertaEncoder::Cpu(Box::new(e))
    }
}

/// One session on a core bundle, with its row and ranges kept between
/// calls.
pub struct Table {
    run: Mutex<Run>,
    config: Config,
    execution: turbo_analysis::Execution,
}

struct Run {
    encoder: turbo_analysis_embed::Encoder,
    rows: Rows,
    ranges: Vec<turbo_token_range>,
}

impl DebertaEncoder {
    /// The FAMILY_DEBERTA_V2 core bundle at `dir` on the device `o` names.
    /// A run is one row of up to the session's max_seq tokens, and a range
    /// is one token, so up to that many come back. The encoder's shape is
    /// given as the CPU encoder takes it, with the session's longest
    /// sequence.
    pub fn open(dir: &Path, o: &EncoderOptions) -> Result<DebertaEncoder, String> {
        let bundle = Bundle::open(dir).map_err(|e| e.message)?;
        shape(&bundle, None)?;
        let seq = match (o.max_seq, &bundle.manifest.embed) {
            (0, Some(e)) => e.max_seq,
            (s, _) => s,
        };
        let mut eo = o.clone();
        eo.max_batch = 1;
        eo.max_ranges = seq.max(1);
        let encoder = turbo_analysis_embed::Encoder::open(dir, &eo).map_err(|e| e.message)?;
        let info = encoder.info().clone();
        let config = shape(&bundle, Some(info.max_seq))?;
        if info.dim as usize != config.hidden {
            return Err(format!("DeBERTa encoder bundle: vectors of {}, hidden {}", info.dim, config.hidden));
        }
        let run = Run { encoder, rows: Rows::new(info.max_seq, 0), ranges: Vec::new() };
        let execution = info.execution;
        Ok(DebertaEncoder::Table(Box::new(Table { run: Mutex::new(run), config, execution })))
    }

    /// The CPU encoder itself on the same bundle's safetensors weights, its
    /// longest sequence the bundle's max_seq: the path the device is held
    /// to, from the one bundle.
    pub fn open_cpu(dir: &Path) -> Result<DebertaEncoder, String> {
        let bundle = Bundle::open(dir).map_err(|e| e.message)?;
        let config = shape(&bundle, None)?;
        let a = bundle.manifest.artifacts.iter().find(|a| a.format == turbo::manifest::Format::Safetensors);
        let file = a.and_then(|a| a.files.first()).ok_or("DeBERTa encoder bundle: no safetensors artifact")?;
        let bytes = bundle.read_verified(file).map_err(|e| e.message)?;
        Encoder::from_safetensors(config, &bytes).map(DebertaEncoder::from).map_err(|e| e.message)
    }

    /// Where the encoder runs: None for the CPU encoder itself, else the
    /// backend, device and precision of its session.
    pub fn execution(&self) -> Option<&turbo_analysis::Execution> {
        match self {
            DebertaEncoder::Cpu(_) => None,
            DebertaEncoder::Table(t) => Some(&t.execution),
        }
    }
}

/// A FAMILY_DEBERTA_V2 bundle's shape as the CPU encoder takes it, its
/// longest sequence `max_seq` (else the bundle's) within its positions.
fn shape(bundle: &Bundle, max_seq: Option<u32>) -> Result<Config, String> {
    let a = bundle.manifest.architecture.as_ref().ok_or("DeBERTa encoder bundle: no architecture")?;
    let Some(r) = a.relative.as_ref().filter(|_| a.family == Family::DebertaV2) else {
        return Err("DeBERTa encoder bundle: the family is not FAMILY_DEBERTA_V2".into());
    };
    let seq = max_seq.or(bundle.manifest.embed.as_ref().map(|e| e.max_seq)).unwrap_or(a.max_positions);
    Ok(Config {
        hidden: a.hidden as usize,
        intermediate: a.intermediate as usize,
        heads: a.heads as usize,
        layers: a.layers as usize,
        vocabulary: a.vocab_size as usize,
        max_sequence: seq.min(a.max_positions) as usize,
        position_buckets: r.position_buckets as usize,
        max_relative_positions: r.max_relative_positions as usize,
        layer_norm_eps: a.layer_norm_eps,
    })
}

impl EncoderStates for DebertaEncoder {
    fn config(&self) -> &Config {
        match self {
            DebertaEncoder::Cpu(e) => e.config(),
            DebertaEncoder::Table(t) => &t.config,
        }
    }

    fn states(&self, ids: &[u32], at: &[usize], out: &mut Vec<f32>) -> Result<(), String> {
        let t = match self {
            DebertaEncoder::Cpu(e) => return EncoderStates::states(&**e, ids, at, out),
            DebertaEncoder::Table(t) => t,
        };
        if at.is_empty() {
            out.clear();
            return Ok(());
        }
        let mut guard = t.run.lock().unwrap_or_else(PoisonError::into_inner);
        let run = &mut *guard;
        let ids: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        run.rows.clear();
        let row = run.rows.push(&[&ids]);
        run.ranges.clear();
        run.ranges.extend(at.iter().map(|&p| turbo_token_range { row, start: p as u32, end: p as u32 + 1 }));
        run.encoder.run_ranges(&run.rows, &run.ranges, out).map_err(|e| e.message)?;
        if out.len() != at.len() * t.config.hidden {
            return Err("DeBERTa encoder bundle: the run returned the wrong number of states".into());
        }
        Ok(())
    }
}
