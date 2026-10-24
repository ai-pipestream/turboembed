//! `NameFinderDL` as an analysis step: a token-classification bundle on
//! turboembed's encoder, read the way OpenNLP reads its ONNX model.
//!
//! Each sentence (the whole text when there is no sentence layer) is cut
//! into chunks of words (decode.rs). Each chunk's words, rejoined with
//! single spaces as `AbstractDL` rejoins them, are tokenized without
//! special tokens and read in one row, `[CLS] pieces [SEP]`, as
//! `NameFinderDL` encodes a chunk. A chunk with more pieces than a row
//! holds, which OpenNLP's model would refuse, is read in consecutive rows;
//! an entity across two rows then ends at the first, the next row starting
//! on an `I-` piece.
//! Every position of a row, its special tokens too, gets its own label
//! scores from the bundle's token classification head, and decode.rs
//! turns them into entities, resolved across the whole document.
//!
//! An entity's token range is the tokens its bytes overlap; one between
//! tokens is the empty range where it would start.

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::Head;
use turbo::manifest::{ClassifierKind, PromptRole, Truncation};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::turbo_token_range;
use turbo_analysis::{Annotator, Error, Execution, IxRange, Layer, ModelRef, Refs, Span, Spans, TagSet, View};
use turbo_analysis_embed::{Encoder, EncoderOptions, Rows};
use turbo_analysis_layers::{Entities, Sentences, Tag, Tokens};

use crate::decode::{self, Candidate, Label};
use crate::place::{Word, place};

/// How a `NameFinderDl` runs its model and cuts its input.
#[derive(Clone, Debug)]
pub struct NameFinderDlOptions {
    pub encoder: EncoderOptions,
    /// Words per chunk: `InferenceOptions.documentSplitSize`.
    pub split: usize,
    /// Words consecutive chunks share: `InferenceOptions.splitOverlapSize`.
    pub overlap: usize,
}

impl Default for NameFinderDlOptions {
    fn default() -> NameFinderDlOptions {
        NameFinderDlOptions { encoder: EncoderOptions::default(), split: 250, overlap: 50 }
    }
}

/// Writes `opennlp:entities` from a token-classification bundle. See the
/// module comment.
pub struct NameFinderDl {
    run: Mutex<Run>,
    tokenizer: Tokenizer,
    head: Head,
    labels: Vec<Label>,
    kinds: Arc<TagSet>,
    model: ModelRef,
    execution: Execution,
    cls: Vec<i32>,
    sep: Vec<i32>,
    /// Pieces per row, beside the special tokens.
    width: usize,
    max_batch: usize,
    max_ranges: usize,
    split: usize,
    overlap: usize,
}

/// Each position of a row: its bytes in the text, none for a special token.
type Positions = Vec<Option<(usize, usize)>>;

/// The rows being built, one range per position, and each row's pieces.
struct Run {
    encoder: Encoder,
    rows: Rows,
    ranges: Vec<turbo_token_range>,
    states: Vec<f32>,
    /// Per row: its document and each position's bytes in the text.
    pieces: Vec<(usize, Positions)>,
}

impl NameFinderDl {
    /// The bundle at `dir`, which must carry a token classification head.
    pub fn open(dir: &Path, o: &NameFinderDlOptions) -> Result<NameFinderDl, CoreError> {
        let invalid = |m: String| CoreError::new(INVALID_ARGUMENT, m);
        decode::chunk_ranges(0, o.split, o.overlap).map_err(invalid)?;
        let bundle = Bundle::open(dir)?;
        let tokenizer = Tokenizer::load(&bundle)?;
        let head = Head::load(&bundle)?;
        if head.kind != ClassifierKind::Token {
            return Err(invalid("the bundle's classifier is not a token classifier".into()));
        }
        let mut eo = o.encoder.clone();
        let embed = bundle.manifest.embed();
        eo.max_batch = if eo.max_batch == 0 { embed.max_batch } else { eo.max_batch };
        eo.max_seq = if eo.max_seq == 0 { embed.max_seq } else { eo.max_seq };
        eo.max_ranges = eo.max_batch.saturating_mul(eo.max_seq);
        let encoder = Encoder::open(dir, &eo)?;
        let info = encoder.info().clone();
        if head.hidden != info.dim as usize {
            return Err(invalid(format!("the head reads width {}, the encoder gives {}", head.hidden, info.dim)));
        }
        let (cls, sep) = tokenizer.template_ends();
        let overhead = cls.len() + sep.len();
        if info.max_seq as usize <= overhead {
            return Err(invalid(format!("max_seq {} leaves no room beside {overhead} special tokens", info.max_seq)));
        }
        let names: Vec<String> = (0..head.num_labels()).map(|i| head.label(i).unwrap_or_default().to_owned()).collect();
        let (labels, types) = decode::labels(&names);
        if types.is_empty() {
            return Err(invalid(format!("the head's labels {names:?} name no B-<type> entity type")));
        }
        let types: Vec<&str> = types.iter().map(String::as_str).collect();
        let m = &bundle.manifest.model;
        Ok(NameFinderDl {
            run: Mutex::new(Run {
                rows: Rows::new(info.max_seq, tokenizer.fill_id()),
                ranges: Vec::new(),
                states: Vec::new(),
                pieces: Vec::new(),
                encoder,
            }),
            kinds: TagSet::new("namefind-dl", &types),
            model: ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: bundle.manifest_sha256.clone() },
            execution: info.execution,
            width: info.max_seq as usize - overhead,
            max_batch: eo.max_batch as usize,
            max_ranges: eo.max_ranges as usize,
            split: o.split,
            overlap: o.overlap,
            tokenizer,
            head,
            labels,
            cls,
            sep,
        })
    }

    /// The entity types, in the order the model's labels first name them.
    pub fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }

    /// Each document's candidates, resolved: one result per input.
    pub fn find(&self, inputs: &[(&str, Vec<(usize, usize)>)]) -> Result<Vec<Vec<Candidate>>, String> {
        let mut found: Vec<Vec<Candidate>> = vec![Vec::new(); inputs.len()];
        let mut run = self.run.lock().unwrap_or_else(PoisonError::into_inner);
        run.rows.clear();
        run.ranges.clear();
        run.pieces.clear();
        let mut ids = Vec::new();
        let mut spans = Vec::new();
        // A chunk's words rejoined with single spaces, as `AbstractDL`
        // rejoins them, and each word's start there and in the text.
        let mut joined = String::new();
        let mut starts: Vec<Word> = Vec::new();
        let bare = Encode {
            add_special_tokens: false,
            truncation: Truncation::None,
            max_tokens: u32::MAX,
            prompt: PromptRole::None,
        };
        for (d, (text, regions)) in inputs.iter().enumerate() {
            for &(lo, hi) in regions {
                let region = text.get(lo..hi).ok_or_else(|| format!("sentence {lo}..{hi} is not in the text"))?;
                let words = decode::words(region);
                for (a, b) in decode::chunk_ranges(words.len(), self.split, self.overlap)? {
                    joined.clear();
                    starts.clear();
                    for &(ws, we) in &words[a..b] {
                        if !joined.is_empty() {
                            joined.push(' ');
                        }
                        starts.push((joined.len(), lo + ws, we - ws));
                        joined.push_str(&region[ws..we]);
                    }
                    ids.clear();
                    spans.clear();
                    self.tokenizer.encode_spans(&joined, bare, &mut ids, &mut spans).map_err(|e| e.to_string())?;
                    // Each end of a piece maps through its own word of
                    // `joined` (place.rs), so one touching or crossing a
                    // joining space still lands on the text's words.
                    let locate = |s: &[u32; 2]| -> Result<Option<(usize, usize)>, String> {
                        let at = place(&starts, *s);
                        match at {
                            Some((a, b)) if !(text.is_char_boundary(a) && text.is_char_boundary(b) && b <= hi) => {
                                Err(format!("piece {}..{} maps to {a}..{b}, not text characters", s[0], s[1]))
                            }
                            _ => Ok(at),
                        }
                    };
                    // A chunk of no pieces is still read, as `[CLS] [SEP]`.
                    let parts: Vec<&[i32]> = if ids.is_empty() { vec![&[]] } else { ids.chunks(self.width).collect() };
                    for (k, part) in parts.into_iter().enumerate() {
                        let first = k * self.width;
                        let rows = self.cls.len() + part.len() + self.sep.len();
                        if run.rows.len() as usize == self.max_batch || run.ranges.len() + rows > self.max_ranges {
                            self.flush(&mut run, &mut found)?;
                        }
                        let row = run.rows.push(&[&self.cls, part, &self.sep]);
                        let mut at: Positions = vec![None; self.cls.len()];
                        for s in &spans[first..first + part.len()] {
                            at.push(locate(s)?);
                        }
                        at.resize(rows, None);
                        for p in 0..rows as u32 {
                            run.ranges.push(turbo_token_range { row, start: p, end: p + 1 });
                        }
                        run.pieces.push((d, at));
                    }
                }
            }
        }
        self.flush(&mut run, &mut found)?;
        Ok(inputs.iter().zip(found).map(|((text, _), c)| decode::resolve(text, c)).collect())
    }

    /// Run the rows written and decode each one's scores.
    fn flush(&self, run: &mut Run, found: &mut [Vec<Candidate>]) -> Result<(), String> {
        if run.rows.is_empty() {
            return Ok(());
        }
        let Run { encoder, rows, ranges, states, pieces } = run;
        encoder.run_ranges(rows, ranges, states).map_err(|e| e.to_string())?;
        let logits = self.head.logits(states).map_err(|e| e.to_string())?;
        let n = self.labels.len();
        let mut at = 0;
        for (d, p) in pieces.iter() {
            found[*d].extend(decode::decode(p, &logits[at * n..(at + p.len()) * n], &self.labels)?);
            at += p.len();
        }
        rows.clear();
        ranges.clear();
        pieces.clear();
        Ok(())
    }

    /// The regions a document is read in, once its inputs are checked:
    /// its sentences, or the whole text.
    fn regions(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
    ) -> Result<Vec<(usize, usize)>, String> {
        if tokens.spans.view() != view.id() || sentences.is_some_and(|s| s.spans.view() != view.id()) {
            return Err("the tokens or sentences are in another view".into());
        }
        tokens.check().map_err(|e| format!("invalid tokens: {e}"))?;
        if let Some(s) = sentences {
            s.check().map_err(|e| format!("invalid sentences: {e}"))?;
        }
        Ok(match sentences {
            Some(s) => (0..s.spans.len())
                .map(|i| s.spans.get(turbo_analysis::Ix::new(i as u32)).range())
                .map(|r| (r.start, r.end))
                .collect(),
            None => vec![(0, view.text().len())],
        })
    }

    /// The entities of a document's resolved candidates.
    fn build(&self, view: &View, tokens: &Tokens, found: Vec<Candidate>) -> Result<Entities, String> {
        let ordered = in_text_order(tokens);
        let mut out = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: self.kinds.clone(),
            score: Vec::new(),
        };
        for c in found {
            out.tokens.push(token_range(tokens, ordered, c.start as u32, c.end as u32)?);
            out.spans.push(Span::new(c.start as u32, c.end as u32));
            out.kind.push(Tag(c.kind as u16));
            out.score.push(c.probability);
        }
        Ok(out)
    }

    fn entities(&self, view: &View, tokens: &Tokens, sentences: Option<&Sentences>) -> Result<Entities, String> {
        let regions = self.regions(view, tokens, sentences)?;
        let found = self.find(&[(view.text(), regions)])?.remove(0);
        self.build(view, tokens, found)
    }

    /// Every document of a batch, its chunks read in shared encoder runs.
    /// A document whose inputs fail their checks fails alone; when the
    /// shared read fails, each document is read alone, so only the one at
    /// fault fails.
    fn entities_batch(&self, inputs: &[Refs<'_, Self>]) -> Vec<Result<Entities, String>> {
        let regions: Vec<_> = inputs.iter().map(|(v, t, s)| self.regions(v, t, *s)).collect();
        let valid: Vec<usize> = (0..inputs.len()).filter(|&i| regions[i].is_ok()).collect();
        let batch: Vec<(&str, Vec<(usize, usize)>)> =
            valid.iter().map(|&i| (inputs[i].0.text(), regions[i].clone().unwrap_or_default())).collect();
        let mut found: Vec<Option<Vec<Candidate>>> = (0..inputs.len()).map(|_| None).collect();
        if let Ok(all) = self.find(&batch) {
            for (&i, c) in valid.iter().zip(all) {
                found[i] = Some(c);
            }
        }
        inputs
            .iter()
            .zip(regions)
            .zip(found)
            .map(|(((view, tokens, _), regions), found)| {
                let found = match found {
                    Some(c) => c,
                    None => self.find(&[(view.text(), regions?)])?.remove(0),
                };
                self.build(view, tokens, found)
            })
            .collect()
    }
}

impl Annotator for NameFinderDl {
    const ID: &'static str = "turbo-namefind-dl";
    type Reads = (View, Tokens, Option<Sentences>);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.entities(view, tokens, sentences)
            .map(|e| (e,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        self.entities_batch(&inputs)
            .into_iter()
            .map(|r| r.map(|e| (e,)).map_err(|detail| Error::Annotator { annotator: Self::ID, detail }))
            .collect()
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        // What changes the entities beside the model: the chunks, and the
        // pieces a row holds, where a long chunk is cut.
        turbo_analysis::Fnv1a::new().u64(self.split as u64).u64(self.overlap as u64).u64(self.width as u64).finish()
    }
}

/// The tokens the bytes `start..end` overlap, by their fragments, as one
/// range; when none does, the empty range where the bytes would sit.
///
/// # Errors
/// When the overlapping tokens are not consecutive in the layer, or none
/// overlaps and the tokens are not in text order, so no range names them.
fn token_range(tokens: &Tokens, ordered: bool, start: u32, end: u32) -> Result<IxRange<Tokens>, String> {
    let spans = &tokens.spans;
    let (starts, ends) = spans.columns();
    if ordered {
        let first = ends.partition_point(|&e| e <= start);
        let last = starts.partition_point(|&s| s < end).max(first);
        return Ok(IxRange::new(first as u32, last as u32));
    }
    let hits: Vec<usize> = (0..spans.len())
        .filter(|&i| spans.fragments(turbo_analysis::Ix::new(i as u32)).any(|f| f.start < end && start < f.end))
        .collect();
    match (hits.first(), hits.last()) {
        (Some(&a), Some(&b)) if b - a + 1 == hits.len() => Ok(IxRange::new(a as u32, b as u32 + 1)),
        (Some(_), _) => Err(format!("the tokens overlapping {start}..{end} are not consecutive in the token layer")),
        _ => Err(format!("no token overlaps {start}..{end}, and the tokens are not in text order")),
    }
}

/// Whether the tokens are single fragments in text order, none inverted
/// and none overlapping the next: checked once per document, as each
/// lookup is then a search.
fn in_text_order(tokens: &Tokens) -> bool {
    let (starts, ends) = tokens.spans.columns();
    !tokens.spans.has_discontinuous()
        && starts.iter().zip(ends).all(|(s, e)| s <= e)
        && starts.windows(2).zip(ends.windows(2)).all(|(s, e)| e[0] <= s[1] && s[0] <= s[1])
}
