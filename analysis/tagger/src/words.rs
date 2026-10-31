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
//!
//! A document's `Subwords` stand in for that tokenizing when they give the
//! same pieces ([`WordEncoder::reads`]): made by this bundle's model, per
//! word token, by a WordPiece tokenizer, each piece inside its token, and
//! every token starting and ending where a BERT pre-tokenizer word does
//! under this tokenizer's own normalizer settings
//! (`Tokenizer::wordpiece_words`). WordPiece reads each such word on its
//! own, so its pieces are the token's own; a token that starts inside a
//! word gets continuation pieces (`##ffa` for `ffa` in `unaffable`) that
//! the token alone would not. Other subwords of this model are passed
//! over and the words tokenized.

use std::path::Path;

use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation, WordPooling};
use turbo::status::{Error, INVALID_ARGUMENT, Result};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::turbo_token_range;
use turbo_analysis::{Execution, Ix, Layer as _, View};
use turbo_analysis_embed::{Encoder, EncoderOptions, Rows};
use turbo_analysis_layers::{Subwords, Tokens};

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
    /// The bundle's manifest digest, which names the model of its subwords.
    manifest: String,
    /// Words this encoder has tokenized itself.
    tokenized: u64,
    /// Rows this encoder has read.
    rows: u64,
    /// Layers each range is read at: the last `taps` (1, the final one,
    /// but within `layer_features`).
    taps: usize,
}

/// One window: its sequence, its words `first..end`, and each word's
/// first piece within the window.
struct Window {
    seq: usize,
    first: usize,
    offsets: Vec<usize>,
}

/// Where a range's states go: pooling, sequence, word. A range of one
/// piece of `pieces` (`piece` counted from 0) feeds every pooling, which
/// is pooled from the pieces here.
struct Owner {
    pooling: Option<usize>,
    seq: usize,
    word: usize,
    piece: usize,
    pieces: usize,
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
        if info.max_seq as usize <= overhead {
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
            manifest: bundle.manifest_sha256.clone(),
            tokenized: 0,
            rows: 0,
            taps: 1,
        })
    }

    /// Words tokenized by this encoder so far, not read from subwords.
    pub fn tokenized(&self) -> u64 {
        self.tokenized
    }

    /// Rows read by the encoder so far: one a window, however many
    /// poolings are taken from it.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Whether `sub` gives each token of `tokens` over `view` the pieces
    /// this encoder would make of it (see the module comment). Subwords of
    /// another model, or in another view, are refused.
    pub fn reads(&self, view: &View, sub: &Subwords, tokens: &Tokens) -> std::result::Result<bool, String> {
        if sub.spans.view() != tokens.spans.view() {
            return Err("the subwords are in another view".into());
        }
        sub.check().map_err(|e| format!("invalid subwords: {e}"))?;
        if sub.model.sha256 != self.manifest {
            return Err("the subwords are from another model".into());
        }
        let Some(col) = &sub.token else { return Ok(false) };
        if self.tokenizer.kind_name() != "wordpiece" {
            return Ok(false);
        }
        let n = tokens.spans.len();
        let inside = sub.spans.iter().zip(col).all(|((_, p), t)| {
            t.get() < n && {
                let w = tokens.spans.get(Ix::new(t.get() as u32));
                w.start <= p.start && p.end <= w.end
            }
        });
        if !inside {
            return Ok(false);
        }
        // Every token on pre-tokenizer word bounds, so WordPiece read it
        // as a word of its own.
        // Under this tokenizer's own normalizer settings.
        let Some(words) = self.tokenizer.wordpiece_words(view.text()) else { return Ok(false) };
        let (mut starts, mut ends): (Vec<usize>, Vec<usize>) = words.into_iter().unzip();
        starts.sort_unstable();
        ends.sort_unstable();
        Ok(tokens.spans.iter().all(|(_, w)| {
            w.start == w.end
                || (starts.binary_search(&(w.start as usize)).is_ok() && ends.binary_search(&(w.end as usize)).is_ok())
        }))
    }

    /// Each sequence's word features, a sequence being token indices of
    /// `tokens` over `view`: pieces from `subwords` when this encoder
    /// [`reads`](Self::reads) them, else from each token's text.
    pub fn token_features(
        &mut self,
        view: &View,
        tokens: &Tokens,
        sequences: &[Vec<usize>],
        subwords: Option<&Subwords>,
        pooling: WordPooling,
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let mut out = self.token_features_pooled(view, tokens, sequences, subwords, &[pooling])?;
        Ok(out.pop().expect("one pooling"))
    }

    /// As [`token_features`](Self::token_features), the features of every
    /// pooling in `poolings` from one read of each window: `out[p]` is
    /// `poolings[p]`'s.
    pub fn token_features_pooled(
        &mut self,
        view: &View,
        tokens: &Tokens,
        sequences: &[Vec<usize>],
        subwords: Option<&Subwords>,
        poolings: &[WordPooling],
    ) -> std::result::Result<Vec<Vec<Vec<f32>>>, String> {
        let sub = match subwords {
            Some(s) if self.reads(view, s, tokens)? => Some(s),
            _ => None,
        };
        let mut pieces = Vec::with_capacity(sequences.len());
        for seq in sequences {
            let mut words = Vec::with_capacity(seq.len());
            for &i in seq {
                let t = Ix::new(i as u32);
                words.push(match sub {
                    Some(s) => {
                        let r = s.of(t);
                        self.bounded(s.ids[r.start.get()..r.end.get()].to_vec())
                    }
                    None => self.pieces(&view.text()[tokens.spans.get(t).range()]).map_err(|e| e.to_string())?,
                });
            }
            pieces.push(words);
        }
        self.features_of(&pieces, poolings).map_err(|e| e.to_string())
    }

    /// A word's pieces as a row holds them: the unknown piece for none,
    /// the first row's worth of many.
    fn bounded(&self, mut ids: Vec<i32>) -> Vec<i32> {
        if ids.is_empty() {
            ids.push(self.tokenizer.unk_id);
        }
        ids.truncate(self.width);
        ids
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
    pub fn pieces(&mut self, word: &str) -> Result<Vec<i32>> {
        let (mut ids, mut spans) = (Vec::new(), Vec::new());
        self.tokenizer.encode_spans(word, BARE, &mut ids, &mut spans)?;
        self.tokenized += 1;
        Ok(self.bounded(ids))
    }

    /// Each sequence's word features [words, hidden], `sequences[s]` its
    /// words' text, pooled by `pooling`.
    pub fn features(&mut self, sequences: &[Vec<&str>], pooling: WordPooling) -> Result<Vec<Vec<f32>>> {
        Ok(self.features_pooled(sequences, &[pooling])?.pop().expect("one pooling"))
    }

    /// As [`features`](Self::features), the features of every pooling in
    /// `poolings` from one read of each window: `out[p]` is
    /// `poolings[p]`'s.
    pub fn features_pooled(&mut self, sequences: &[Vec<&str>], poolings: &[WordPooling]) -> Result<Vec<Vec<Vec<f32>>>> {
        let pieces: Vec<Vec<Vec<i32>>> =
            sequences.iter().map(|words| words.iter().map(|w| self.pieces(w)).collect()).collect::<Result<_>>()?;
        self.features_of(&pieces, poolings)
    }

    /// Each sequence's word features at each of the encoder's last
    /// `layers` layers, from one read of each window: `out[l]` is layer
    /// `model layers - layers + 1 + l` (counted from 1), so the last is
    /// what [`features`](Self::features) gives, and each is what the
    /// encoder cut after that layer would give. The CPU backend reads
    /// layers of BERT, RoBERTa, ModernBERT and nomic encoders; a run holds
    /// a window's words at every layer, so `max_batch` must be at least
    /// `layers`.
    pub fn layer_features(
        &mut self,
        sequences: &[Vec<&str>],
        pooling: WordPooling,
        layers: u32,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        if layers == 0 || layers as usize > self.max_batch {
            return Err(Error::new(
                INVALID_ARGUMENT,
                format!("{layers} layers: 1 to max_batch ({}) a read", self.max_batch),
            ));
        }
        let pieces: Vec<Vec<Vec<i32>>> =
            sequences.iter().map(|words| words.iter().map(|w| self.pieces(w)).collect()).collect::<Result<_>>()?;
        self.taps = layers as usize;
        let out = self.features_of(&pieces, &[pooling]);
        self.taps = 1;
        out
    }

    /// The features of each pooling of `poolings`, from each word's
    /// pieces, every word one to `width` pieces. Each window is one row
    /// whatever the poolings: a pooling is only which of the row's states
    /// a range reads.
    fn features_of(&mut self, sequences: &[Vec<Vec<i32>>], poolings: &[WordPooling]) -> Result<Vec<Vec<Vec<f32>>>> {
        // out[t * poolings + p]: pooling p at tap t.
        let mut out: Vec<Vec<Vec<f32>>> = (0..self.taps * poolings.len())
            .map(|_| sequences.iter().map(|s| vec![0.0; s.len() * self.hidden]).collect())
            .collect();
        let max_ranges = self.max_ranges / self.taps;
        let mut rows = Rows::new((self.width + self.cls.len() + self.sep.len()) as u32, self.tokenizer.fill_id());
        let (mut ranges, mut owners, mut states) = (Vec::new(), Vec::<Owner>::new(), Vec::new());
        let mut row_ids = Vec::new();
        for (s, pieces) in sequences.iter().enumerate() {
            for (window, keep) in self.windows(s, pieces) {
                let kept = keep.iter().filter(|&&k| k).count();
                let kept_pieces: usize =
                    keep.iter().enumerate().filter(|&(_, &k)| k).map(|(k, _)| pieces[window.first + k].len()).sum();
                // A range a word and pooling, unless those outgrow a run
                // (one row a batch): then a range a piece, which every
                // pooling reads, so the row is still read once. A row's
                // pieces never outgrow a run.
                let per_piece = kept * poolings.len() > max_ranges;
                let need = if per_piece { kept_pieces } else { kept * poolings.len() };
                if rows.len() as usize == self.max_batch || ranges.len() + need > max_ranges {
                    self.flush(&mut rows, &mut ranges, &mut owners, &mut states, &mut out, poolings)?;
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
                    let (seq, word) = (window.seq, window.first + k);
                    let (start, n) = ((self.cls.len() + at) as u32, pieces[word].len());
                    if per_piece {
                        for piece in 0..n {
                            let at = start + piece as u32;
                            ranges.push(turbo_token_range { row, start: at, end: at + 1 });
                            owners.push(Owner { pooling: None, seq, word, piece, pieces: n });
                        }
                        continue;
                    }
                    for (p, pooling) in poolings.iter().enumerate() {
                        let len = match pooling {
                            WordPooling::First => 1,
                            WordPooling::Mean => n as u32,
                        };
                        ranges.push(turbo_token_range { row, start, end: start + len });
                        owners.push(Owner { pooling: Some(p), seq, word, piece: 0, pieces: n });
                    }
                }
            }
        }
        self.flush(&mut rows, &mut ranges, &mut owners, &mut states, &mut out, poolings)?;
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
        owners: &mut Vec<Owner>,
        states: &mut Vec<f32>,
        out: &mut [Vec<Vec<f32>>],
        poolings: &[WordPooling],
    ) -> Result<()> {
        if !ranges.is_empty() {
            if self.taps == 1 {
                self.encoder.run_ranges(rows, ranges, states)?;
            } else {
                self.encoder.run_layer_ranges(rows, ranges, self.taps as u32, states)?;
            }
            self.rows += u64::from(rows.len());
            let (h, n, taps) = (self.hidden, ranges.len(), self.taps);
            let per = out.len() / taps;
            // states: [taps, ranges, hidden].
            for t in 0..taps {
                for (r, o) in owners.iter().enumerate() {
                    let state = &states[(t * n + r) * h..][..h];
                    let Some(p) = o.pooling else {
                        // A word's pieces come in order in one run; the
                        // mean is the cpu's: the sum in order, times 1 /
                        // pieces.
                        for (p, pooling) in poolings.iter().enumerate() {
                            let dst = &mut out[t * per + p][o.seq][o.word * h..][..h];
                            match pooling {
                                WordPooling::First if o.piece == 0 => dst.copy_from_slice(state),
                                WordPooling::First => {}
                                WordPooling::Mean => {
                                    if o.piece == 0 {
                                        dst.fill(0.0);
                                    }
                                    dst.iter_mut().zip(state).for_each(|(d, &x)| *d += x);
                                    if o.piece + 1 == o.pieces {
                                        let inv = 1.0 / o.pieces as f32;
                                        dst.iter_mut().for_each(|d| *d *= inv);
                                    }
                                }
                            }
                        }
                        continue;
                    };
                    out[t * per + p][o.seq][o.word * h..][..h].copy_from_slice(state);
                }
            }
        }
        rows.clear();
        ranges.clear();
        owners.clear();
        Ok(())
    }
}

pub use turbo::head::run::plan;
