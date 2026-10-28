//! `NameFinderDL` as an analysis step: a token-classification bundle on
//! turboembed's encoder, read the way OpenNLP reads its ONNX model.
//!
//! Each sentence (the whole text when there is no sentence layer) is cut
//! into chunks of words (decode.rs). Each chunk's words, rejoined with
//! single spaces as `AbstractDL` rejoins them, are tokenized without
//! special tokens and read in one row, `[CLS] pieces [SEP]`, as
//! `NameFinderDL` encodes a chunk. A chunk with more pieces than a row
//! holds, which OpenNLP's model would refuse, is read in rows overlapping
//! by half a row (`decode::row_ranges`), each piece labelled by the row
//! where it has the most context on both sides, and the chunk's pieces
//! decoded as one run, the special tokens between rows left out, so an
//! entity across a row boundary is whole.
//! Every position of a row gets its own label scores from the bundle's
//! token classification head, and decode.rs turns them into entities,
//! each word labelled by its first piece unless `Decoding::Piece` asks
//! for every piece's own label, resolved across the whole document.
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
use turbo_analysis::{Annotator, Error, Execution, IxRange, Layer, ModelRef, Refs, Span, Spans, TagSet, View, ViewId};
use turbo_analysis_embed::{Encoder, EncoderOptions, Rows};
use turbo_analysis_layers::{Entities, Sentences, Subwords, Tag, Tokens};

mod layers;

use crate::decode::{self, Candidate, Decoding, Label};

/// How a `NameFinderDl` runs its model and cuts its input.
#[derive(Clone, Debug)]
pub struct NameFinderDlOptions {
    pub encoder: EncoderOptions,
    /// Words per chunk: `InferenceOptions.documentSplitSize`.
    pub split: usize,
    /// Words consecutive chunks share: `InferenceOptions.splitOverlapSize`.
    pub overlap: usize,
    /// Word labels by default; `Decoding::Piece` as `NameFinderDL` decodes.
    pub decoding: Decoding,
}

impl Default for NameFinderDlOptions {
    fn default() -> NameFinderDlOptions {
        NameFinderDlOptions { encoder: EncoderOptions::default(), split: 250, overlap: 50, decoding: Decoding::Word }
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
    decoding: Decoding,
}

/// Each position of a row: its bytes in the text, none for a special token.
type Positions = Vec<Option<(usize, usize)>>;

/// One row's positions: its document, the positions whose labels it gives
/// (`keep`), whether it ends its chunk, and each position's bytes in the
/// text and word in the chunk, none for a special token.
struct Row {
    doc: usize,
    keep: std::ops::Range<usize>,
    last: bool,
    bytes: Positions,
    words: Vec<Option<usize>>,
}

/// A chunk's positions and scores so far, its rows joined.
#[derive(Default)]
struct Chunk {
    bytes: Positions,
    words: Vec<Option<usize>>,
    scores: Vec<f32>,
}

/// The rows being built, one range per position, and each row's pieces.
struct Run {
    encoder: Encoder,
    rows: Rows,
    ranges: Vec<turbo_token_range>,
    states: Vec<f32>,
    pieces: Vec<Row>,
    /// The chunk whose rows are being read, across flushes.
    chunk: Chunk,
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
                chunk: Chunk::default(),
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
            decoding: o.decoding,
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
        let inputs: Vec<layers::Input<'_>> = inputs
            .iter()
            .map(|(text, regions)| layers::Input {
                view: ViewId::SOURCE,
                text,
                regions: regions.clone(),
                tokens: None,
                subwords: None,
            })
            .collect();
        self.find_layers(&inputs)
    }

    /// Run the rows written and decode each one's scores.
    fn flush(&self, run: &mut Run, found: &mut [Vec<Candidate>]) -> Result<(), String> {
        if run.rows.is_empty() {
            return Ok(());
        }
        let Run { encoder, rows, ranges, states, pieces, chunk } = run;
        encoder.run_ranges(rows, ranges, states).map_err(|e| e.to_string())?;
        let logits = self.head.logits(states).map_err(|e| e.to_string())?;
        let n = self.labels.len();
        let mut at = 0;
        for row in pieces.iter() {
            // Each row gives the labels of the pieces it keeps, the special
            // tokens between two rows left out, so the chunk is one run.
            let (lo, hi) = (row.keep.start, row.keep.end);
            chunk.bytes.extend_from_slice(&row.bytes[lo..hi]);
            chunk.words.extend_from_slice(&row.words[lo..hi]);
            chunk.scores.extend_from_slice(&logits[(at + lo) * n..(at + hi) * n]);
            at += row.bytes.len();
            if row.last {
                let Chunk { bytes, words, scores } = &*chunk;
                found[row.doc].extend(match self.decoding {
                    Decoding::Word => decode::decode_words(bytes, words, scores, &self.labels)?,
                    Decoding::Piece => decode::decode(bytes, scores, &self.labels)?,
                });
                *chunk = Chunk::default();
            }
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
        subwords: Option<&Subwords>,
    ) -> Result<Vec<(usize, usize)>, String> {
        if tokens.spans.view() != view.id()
            || sentences.is_some_and(|s| s.spans.view() != view.id())
            || subwords.is_some_and(|s| s.spans.view() != view.id())
        {
            return Err("the tokens, subwords, or sentences are in another view".into());
        }
        tokens.check().map_err(|e| format!("invalid tokens: {e}"))?;
        if let Some(s) = sentences {
            s.check().map_err(|e| format!("invalid sentences: {e}"))?;
        }
        if let Some(s) = subwords {
            s.check().map_err(|e| format!("invalid subwords: {e}"))?;
            if s.model != self.model {
                return Err("the subwords are from another model".into());
            }
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

    fn entities(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<Entities, String> {
        let regions = self.regions(view, tokens, sentences, subwords)?;
        let input = layers::Input { view: view.id(), text: view.text(), regions, tokens: Some(tokens), subwords };
        let found = self.find_layers(&[input])?.remove(0);
        self.build(view, tokens, found)
    }

    /// Every document of a batch, its chunks read in shared encoder runs.
    /// A document whose inputs fail their checks fails alone; when the
    /// shared read fails, each document is read alone, so only the one at
    /// fault fails.
    fn entities_batch(&self, inputs: &[Refs<'_, Self>]) -> Vec<Result<Entities, String>> {
        let regions: Vec<_> = inputs.iter().map(|(v, t, s, w)| self.regions(v, t, *w, *s)).collect();
        let valid: Vec<usize> = (0..inputs.len()).filter(|&i| regions[i].is_ok()).collect();
        let batch: Vec<layers::Input<'_>> = valid
            .iter()
            .map(|&i| layers::Input {
                view: inputs[i].0.id(),
                text: inputs[i].0.text(),
                regions: regions[i].clone().unwrap_or_default(),
                tokens: Some(inputs[i].1),
                subwords: inputs[i].2,
            })
            .collect();
        let mut found: Vec<Option<Vec<Candidate>>> = (0..inputs.len()).map(|_| None).collect();
        if let Ok(all) = self.find_layers(&batch) {
            for (&i, c) in valid.iter().zip(all) {
                found[i] = Some(c);
            }
        }
        inputs
            .iter()
            .zip(regions)
            .zip(found)
            .map(|(((view, tokens, subwords, _), regions), found)| {
                let found = match found {
                    Some(c) => c,
                    None => {
                        let input = layers::Input {
                            view: view.id(),
                            text: view.text(),
                            regions: regions?,
                            tokens: Some(tokens),
                            subwords: *subwords,
                        };
                        self.find_layers(&[input])?.remove(0)
                    }
                };
                self.build(view, tokens, found)
            })
            .collect()
    }
}

impl Annotator for NameFinderDl {
    const ID: &'static str = "turbo-namefind-dl";
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.entities(view, tokens, sentences, subwords)
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
        // What changes the entities beside the model: the chunks, the
        // pieces a row holds, where a long chunk is cut, and the decoding.
        let decoding = match self.decoding {
            Decoding::Word => 0,
            Decoding::Piece => 1,
        };
        turbo_analysis::Fnv1a::new()
            .u64(self.split as u64)
            .u64(self.overlap as u64)
            .u64(self.width as u64)
            .u64(decoding)
            .finish()
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
    if ordered && start < end {
        return Ok(spans.overlapping_range(turbo_analysis::Span::new(start, end)));
    }
    if ordered {
        let (starts, ends) = spans.columns();
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
