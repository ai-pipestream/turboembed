//! Per-word encoder features: each sequence of words read by the bundle's
//! encoder on whatever device it runs, and each word's final-layer states
//! pooled on the device by the range write (`turbo_embed_write_ranges`):
//! its first piece under WORD_POOLING_FIRST, the mean of its pieces under
//! WORD_POOLING_MEAN. A head's trainer and its annotator both read words
//! through this one path, so the features a head learns from are the
//! features it is run on.
//!
//! Each word is tokenized on its own, without special tokens, so a piece
//! never spans two words. A word the tokenizer gives no piece is read as
//! the unknown piece; a word of more pieces than a row holds keeps the
//! first row's worth. A sequence whose pieces fit one row,
//! `[template start] pieces [template end]`, is read in one; a longer one
//! in windows of whole words, each starting about half a row after the
//! last, and each word's features come from the window where it has the
//! most pieces of context on its nearer side (the earlier on a tie).

use std::path::Path;

use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation, WordPooling};
use turbo::status::{Error, INVALID_ARGUMENT, Result};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::turbo_token_range;
use turbo_analysis::Execution;
use turbo_analysis_embed::{Encoder, EncoderOptions, Rows};

/// The encoder, its tokenizer, and how rows are laid out.
pub struct WordEncoder {
    encoder: Encoder,
    tokenizer: Tokenizer,
    cls: Vec<i32>,
    sep: Vec<i32>,
    /// Pieces per row, beside the special tokens.
    width: usize,
    max_batch: usize,
    max_ranges: usize,
    hidden: usize,
}

/// One window: its sequence, its words `first..end`, and each word's
/// first piece within the window.
struct Window {
    seq: usize,
    first: usize,
    offsets: Vec<usize>,
}

const BARE: Encode =
    Encode { add_special_tokens: false, truncation: Truncation::None, max_tokens: u32::MAX, prompt: PromptRole::None };

impl WordEncoder {
    /// The encoder of the bundle at `dir`. A 0 in `o.max_batch` or
    /// `o.max_seq` is the bundle's; ranges are sized so a full batch of
    /// rows can be read one word a token.
    pub fn open(dir: &Path, o: &EncoderOptions) -> Result<WordEncoder> {
        let bundle = Bundle::open(dir)?;
        let tokenizer = Tokenizer::load(&bundle)?;
        let embed = bundle.manifest.embed();
        let mut eo = o.clone();
        eo.max_batch = if eo.max_batch == 0 { embed.max_batch } else { eo.max_batch };
        eo.max_seq = if eo.max_seq == 0 { embed.max_seq } else { eo.max_seq };
        eo.max_ranges = eo.max_batch.saturating_mul(eo.max_seq);
        let encoder = Encoder::open(dir, &eo)?;
        let (cls, sep) = tokenizer.template_ends();
        let overhead = cls.len() + sep.len();
        let info = encoder.info().clone();
        if info.max_seq as usize <= overhead + 1 {
            return Err(Error::new(
                INVALID_ARGUMENT,
                format!("max_seq {} leaves no room beside {overhead} special tokens", info.max_seq),
            ));
        }
        Ok(WordEncoder {
            width: info.max_seq as usize - overhead,
            max_batch: info.max_batch as usize,
            max_ranges: info.max_ranges as usize,
            hidden: info.dim as usize,
            encoder,
            tokenizer,
            cls,
            sep,
        })
    }

    /// Values per word.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Pieces a row holds beside its special tokens.
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn execution(&self) -> &Execution {
        &self.encoder.info().execution
    }

    /// Each word's pieces, read as the module comment says.
    pub fn pieces(&self, word: &str) -> Result<Vec<i32>> {
        let (mut ids, mut spans) = (Vec::new(), Vec::new());
        self.tokenizer.encode_spans(word, BARE, &mut ids, &mut spans)?;
        if ids.is_empty() {
            ids.push(self.tokenizer.unk_id);
        }
        ids.truncate(self.width);
        Ok(ids)
    }

    /// Each sequence's word features [words, hidden], `sequences[s]` its
    /// words' text, pooled by `pooling`.
    pub fn features(&mut self, sequences: &[Vec<&str>], pooling: WordPooling) -> Result<Vec<Vec<f32>>> {
        let mut out: Vec<Vec<f32>> = sequences.iter().map(|s| vec![0.0; s.len() * self.hidden]).collect();
        let mut rows = Rows::new((self.width + self.cls.len() + self.sep.len()) as u32, self.tokenizer.fill_id());
        let (mut ranges, mut owners, mut states) = (Vec::new(), Vec::<(usize, usize)>::new(), Vec::new());
        let mut row_ids = Vec::new();
        for (s, words) in sequences.iter().enumerate() {
            let pieces: Vec<Vec<i32>> = words.iter().map(|w| self.pieces(w)).collect::<Result<_>>()?;
            for (window, keep) in self.windows(s, &pieces) {
                let n: usize = keep.iter().filter(|&&k| k).count();
                if rows.len() as usize == self.max_batch || ranges.len() + n > self.max_ranges {
                    self.flush(&mut rows, &mut ranges, &mut owners, &mut states, &mut out)?;
                }
                row_ids.clear();
                for p in &pieces[window.first..window.first + window.offsets.len()] {
                    row_ids.extend_from_slice(p);
                }
                let row = rows.push(&[&self.cls, &row_ids, &self.sep]);
                for (k, &at) in window.offsets.iter().enumerate() {
                    if !keep[k] {
                        continue;
                    }
                    let word = window.first + k;
                    let start = (self.cls.len() + at) as u32;
                    let len = match pooling {
                        WordPooling::First => 1,
                        WordPooling::Mean => pieces[word].len() as u32,
                    };
                    ranges.push(turbo_token_range { row, start, end: start + len });
                    owners.push((window.seq, word));
                }
            }
        }
        self.flush(&mut rows, &mut ranges, &mut owners, &mut states, &mut out)?;
        Ok(out)
    }

    /// The windows of sequence `seq`, each with the words whose features
    /// it gives.
    fn windows(&self, seq: usize, pieces: &[Vec<i32>]) -> Vec<(Window, Vec<bool>)> {
        let lens: Vec<usize> = pieces.iter().map(Vec::len).collect();
        let (spans, owner) = plan(&lens, self.width);
        spans
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| {
                let offsets = lens[a..b].iter().scan(0, |at, &p| Some(std::mem::replace(at, *at + p))).collect();
                let keep = (a..b).map(|w| owner[w] == i).collect();
                (Window { seq, first: a, offsets }, keep)
            })
            .collect()
    }

    fn flush(
        &mut self,
        rows: &mut Rows,
        ranges: &mut Vec<turbo_token_range>,
        owners: &mut Vec<(usize, usize)>,
        states: &mut Vec<f32>,
        out: &mut [Vec<f32>],
    ) -> Result<()> {
        if !ranges.is_empty() {
            self.encoder.run_ranges(rows, ranges, states)?;
            let h = self.hidden;
            for (r, &(s, w)) in owners.iter().enumerate() {
                out[s][w * h..][..h].copy_from_slice(&states[r * h..][..h]);
            }
        }
        rows.clear();
        ranges.clear();
        owners.clear();
        Ok(())
    }
}

/// The windows of a sequence whose words have `lens` pieces each (every
/// one at least 1 and at most `width`), as words `a..b`, and the window
/// each word's features come from. See the module comment.
pub fn plan(lens: &[usize], width: usize) -> (Vec<(usize, usize)>, Vec<usize>) {
    let n = lens.len();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut a = 0;
    while a < n {
        let (mut b, mut used) = (a, 0);
        while b < n && used + lens[b] <= width {
            used += lens[b];
            b += 1;
        }
        spans.push((a, b));
        if b == n {
            break;
        }
        // The next window starts at the first word of the second half.
        let (mut next, mut seen) = (a, 0);
        while next < b && seen < used / 2 {
            seen += lens[next];
            next += 1;
        }
        a = next.max(a + 1);
    }
    // A word's context in a window: the pieces on its nearer side, a
    // sequence's own ends counting as all the context there is.
    let context = |&(a, b): &(usize, usize), w: usize| -> usize {
        let before = if a == 0 { usize::MAX } else { lens[a..w].iter().sum() };
        let after = if b == n { usize::MAX } else { lens[w + 1..b].iter().sum() };
        before.min(after)
    };
    let owner = (0..n)
        .map(|w| {
            let mut best: Option<(usize, usize)> = None;
            for (i, span) in spans.iter().enumerate() {
                if span.0 <= w && w < span.1 {
                    let c = context(span, w);
                    if best.is_none_or(|(_, bc)| c > bc) {
                        best = Some((i, c));
                    }
                }
            }
            best.map_or(0, |(i, _)| i)
        })
        .collect();
    (spans, owner)
}
