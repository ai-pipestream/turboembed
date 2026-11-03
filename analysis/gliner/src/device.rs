//! GLiNER's encoder through the backend table (#883): a core bundle of the
//! DeBERTa-v2 family, on the device `EncoderOptions` names, cpu, cuda or
//! levelzero. Only the rows the heads read come back: a one-token range at
//! each entity marker and each word's first subword, through the range
//! write, as SaT reads its tokens. A range of one token is that token's
//! state, unnormalized.

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::manifest::Family;
use turbo::turbo_token_range;
use turbo_analysis_embed::{Encoder, EncoderOptions, Rows};

use crate::{
    PackedInput,
    schema::{MAX_LABELS, MAX_TOKENS},
};

/// One session on the bundle, its row and ranges kept between calls.
pub(crate) struct Table {
    run: Mutex<Run>,
    width: usize,
    /// Backend, device and precision the session runs at.
    pub(crate) execution: turbo_analysis::Execution,
}

struct Run {
    encoder: Encoder,
    rows: Rows,
    ranges: Vec<turbo_token_range>,
}

impl Table {
    /// The bundle at `dir` on the device `o` names, one row a run of up to
    /// the bundle's max_seq tokens, and the encoder's shape as the CPU
    /// encoder takes it, its longest sequence the session's.
    pub(crate) fn open(dir: &Path, o: &EncoderOptions) -> Result<(Table, turbo::deberta::Config), String> {
        let bundle = Bundle::open(dir).map_err(|e| e.message)?;
        let a = bundle.manifest.architecture.as_ref().ok_or("gliner encoder bundle: no architecture")?;
        let relative = a.relative.as_ref().filter(|_| a.family == Family::DebertaV2);
        let Some(r) = relative else {
            return Err("gliner encoder bundle: the family is not FAMILY_DEBERTA_V2".into());
        };
        let mut eo = o.clone();
        eo.max_batch = 1;
        // A range a prompt or word, each a token of the one row: at most
        // the row's tokens.
        let seq = match (o.max_seq, &bundle.manifest.embed) {
            (0, Some(e)) => e.max_seq,
            (s, _) => s,
        };
        eo.max_ranges = ((MAX_LABELS + MAX_TOKENS) as u32).min(seq.max(1));
        let encoder = Encoder::open(dir, &eo).map_err(|e| e.message)?;
        let info = encoder.info().clone();
        let config = turbo::deberta::Config {
            hidden: a.hidden as usize,
            intermediate: a.intermediate as usize,
            heads: a.heads as usize,
            layers: a.layers as usize,
            vocabulary: a.vocab_size as usize,
            max_sequence: info.max_seq.min(a.max_positions) as usize,
            position_buckets: r.position_buckets as usize,
            max_relative_positions: r.max_relative_positions as usize,
            layer_norm_eps: a.layer_norm_eps,
        };
        if info.dim as usize != config.hidden {
            return Err(format!("gliner encoder bundle: vectors of {}, hidden {}", info.dim, config.hidden));
        }
        let rows = Rows::new(info.max_seq, 0);
        let run = Run { encoder, rows, ranges: Vec::new() };
        let execution = info.execution;
        Ok((Table { run: Mutex::new(run), width: config.hidden, execution }, config))
    }

    /// The states of `p`'s prompts then its words into `out`, [prompts +
    /// words, hidden], in that order.
    pub(crate) fn states(&self, p: &PackedInput, out: &mut Vec<f32>) -> Result<(), String> {
        let mut guard = self.run.lock().unwrap_or_else(PoisonError::into_inner);
        let run = &mut *guard;
        let ids: Vec<i32> = p.ids.iter().map(|&i| i as i32).collect();
        run.rows.clear();
        let row = run.rows.push(&[&ids]);
        run.ranges.clear();
        for &at in p.prompts.iter().chain(&p.words) {
            run.ranges.push(turbo_token_range { row, start: at as u32, end: at as u32 + 1 });
        }
        run.encoder.run_ranges(&run.rows, &run.ranges, out).map_err(|e| e.message)?;
        if out.len() != run.ranges.len() * self.width {
            return Err("gliner encoder bundle: the run returned the wrong number of states".into());
        }
        Ok(())
    }
}
