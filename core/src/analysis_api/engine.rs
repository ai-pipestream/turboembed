//! In-process analysis engine backing `turbo_analyzer`: the rule sentence
//! splitter and rule tokenizer, the PII detector, and the heads of the
//! bundle the descriptor names (heads/), run a sentence at a time. A
//! bundle with a sentence boundary head finds the sentences in place of
//! the rule splitter, over the rule tokenizer's tokens.

use turbo_analysis::{
    Codecs, Document, DocumentGraph, Key, Layer, OffsetTable, Provenance, Span, Spans, Tag, ViewId, export, import,
};
use turbo_analysis_layers::export::register_codecs;
use turbo_analysis_layers::{Entities, Lemmas, MorphFeatures, Pii, PiiCandidates, PosTags, Sentences, Tokens};
use turbo_pii::{Pack, PiiDetector};
use turbo_sentdetect::RuleSplitter;
use turbo_tokenize::{TokenizerOptions, simple_spans};

use super::graph::{Origin, extract_graph};
use super::heads::{BundleHeads, Want, unsupported};
use super::pii::{extract_pii_candidates_layer, extract_pii_layer};
use super::{
    AnalysisResultInner, PiiRows, SpanDraft, SpanPoolBuilder, TURBO_ANALYSIS_LAYER_ENTITIES,
    TURBO_ANALYSIS_LAYER_LEMMAS, TURBO_ANALYSIS_LAYER_MORPHOLOGY, TURBO_ANALYSIS_LAYER_POS, coordinate_range,
};
use crate::status::{BUNDLE_INVALID, CAPACITY, Error, INVALID_ARGUMENT, Result};

mod direct;
mod layers;

/// Configuration decoded from `turbo_analyzer_desc` using only the fields
/// proven present by `struct_size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AnalyzerConfig<'a> {
    pub(crate) language: &'a str,
    pub(crate) bundle_path: &'a str,
    /// What PII rows carry: their value (`TURBO_ANALYZER_PII_VALUES`,
    /// #1904) and their numeric kind (`TURBO_ANALYZER_PII_KIND_IDS`, #1940).
    pub(crate) pii: PiiRows,
}

impl Default for AnalyzerConfig<'_> {
    fn default() -> Self {
        Self { language: "en", bundle_path: "", pii: PiiRows::default() }
    }
}

pub struct AnalyzerEngine {
    rule_splitter: RuleSplitter,
    heads: Option<BundleHeads>,
    pii_detector: PiiDetector,
    codecs: Codecs,
    pii: PiiRows,
}

fn index_u32(index: usize, what: &str) -> Result<u32> {
    u32::try_from(index).map_err(|_| Error::new(CAPACITY, format!("{what} exceeds u32")))
}

const SOURCE_VIEW: u32 = ViewId::SOURCE.0 as u32;

/// `turbo_analyzer_desc.bundle_path` as `turbo_error.field` numbers it.
const BUNDLE_PATH_FIELD: u32 = 7;

/// Each token's text, checked to lie in `text` on character boundaries:
/// a span outside its text is TURBO_E_CAPACITY, as for any row (turbo.h,
/// turbo_span).
fn forms<'t>(text: &'t str, tokens: &[Span]) -> Result<Vec<&'t str>> {
    tokens
        .iter()
        .map(|s| {
            let (start, end) = (s.start as usize, s.end as usize);
            if start > end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                return Err(Error::new(CAPACITY, "tokenizer returned a span outside the text"));
            }
            Ok(&text[start..end])
        })
        .collect()
}

/// The token index runs the heads read, in order: the tokens wholly
/// inside each sentence, and between them any run of tokens inside no
/// sentence, so every token is in exactly one run.
fn sequences(tokens: &[Span], sentences: &[Span]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut last: Option<Option<usize>> = None;
    let mut s = 0;
    for (i, t) in tokens.iter().enumerate() {
        while s < sentences.len() && sentences[s].end <= t.start {
            s += 1;
        }
        let key = (s < sentences.len() && sentences[s].start <= t.start && t.end <= sentences[s].end).then_some(s);
        if last != Some(key) {
            out.push(Vec::new());
            last = Some(key);
        }
        out.last_mut().expect("pushed").push(i);
    }
    out
}

impl AnalyzerEngine {
    /// The engine `cfg` configures.
    pub(crate) fn from_desc(cfg: AnalyzerConfig<'_>) -> Result<Self> {
        let heads = match cfg.bundle_path {
            "" => None,
            path => Some(BundleHeads::open(path).map_err(|e| Error {
                field: BUNDLE_PATH_FIELD,
                message: format!("turbo_analyzer_desc.bundle_path {path:?}: {}", e.message),
                ..e
            })?),
        };
        let mut codecs = Codecs::new();
        register_codecs(&mut codecs);
        Ok(Self {
            rule_splitter: RuleSplitter::new(cfg.language),
            heads,
            pii_detector: PiiDetector::pack(Pack::AllStructured),
            codecs,
            pii: cfg.pii,
        })
    }

    /// Whether the bundle finds the sentences with a head.
    fn head_sentences(&self) -> bool {
        self.heads.as_ref().is_some_and(BundleHeads::has_sentences)
    }

    /// The sentences of `text`: the bundle's sentence boundary head's
    /// over the rule tokens when it has one, the rule splitter's
    /// otherwise.
    fn sentence_spans(&self, text: &str) -> Result<Vec<Span>> {
        if self.head_sentences() {
            return self.sentences_over(text, &self.token_spans(text));
        }
        Ok(self.rule_sentences(text))
    }

    /// As [`sentence_spans`](Self::sentence_spans), over `text`'s rule
    /// tokens `tokens` already made. A head's sentence runs from its
    /// first token's start to its last token's end.
    fn sentences_over(&self, text: &str, tokens: &[Span]) -> Result<Vec<Span>> {
        let Some(heads) = self.heads.as_ref().filter(|h| h.has_sentences()) else {
            return Ok(self.rule_sentences(text));
        };
        let ranges = heads.sentences(&forms(text, tokens)?)?.unwrap_or_default();
        Ok(ranges.iter().map(|r| Span::new(tokens[r.start].start, tokens[r.end - 1].end)).collect())
    }

    fn rule_sentences(&self, text: &str) -> Vec<Span> {
        let mut spans = Vec::new();
        self.rule_splitter.split(text, &mut spans);
        spans
    }

    fn token_spans(&self, text: &str) -> Vec<Span> {
        let mut spans = Vec::new();
        simple_spans(text, TokenizerOptions::default(), &mut spans);
        spans
    }

    /// The bundle's heads, or the refusal of `task` without a bundle.
    fn heads(&self, task: &str, layer: &str) -> Result<&BundleHeads> {
        self.heads.as_ref().ok_or_else(|| unsupported(task, layer, None))
    }

    /// A result of no rows: what every task writes for no input, bundle
    /// or not.
    fn empty() -> AnalysisResultInner {
        SpanPoolBuilder::default().finish(Vec::new())
    }

    pub fn pos_tag(&self, tokens: &[&str]) -> Result<AnalysisResultInner> {
        if tokens.is_empty() {
            return Ok(Self::empty());
        }
        let heads = self.heads("POS tagging", "pos")?;
        let tagset = heads.pos_tagset()?;
        let written = heads.run(&[tokens.to_vec()], &Want { pos: true, ..Want::default() })?;
        let decoded = written.pos.expect("asked for").remove(0);
        let mut builder = SpanPoolBuilder::with_capacity(tokens.len(), tokens.len() * 6);
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_POS, PosTags::ID, "", SOURCE_VIEW)?;
        for (idx, (&tag, &prob)) in decoded.tags.iter().zip(&decoded.prob).enumerate() {
            let item = index_u32(idx, "POS token count")?;
            let label = builder.string_ref(tagset.label(Tag(tag as u16)))?;
            let draft = SpanDraft {
                item_index: item,
                tokens: Some((item, item + 1)),
                label: Some(label),
                prob: Some(prob),
                ..SpanDraft::default()
            };
            builder.push(layer, draft)?;
        }
        Ok(builder.finish(Vec::new()))
    }

    pub fn morphology(&self, tokens: &[&str]) -> Result<AnalysisResultInner> {
        if tokens.is_empty() {
            return Ok(Self::empty());
        }
        let heads = self.heads("morphology", "morphology")?;
        let tagset = heads.morph_tagset()?;
        let written = heads.run(&[tokens.to_vec()], &Want { morphology: true, ..Want::default() })?;
        let decoded = written.morphology.expect("asked for").remove(0);
        let mut builder = SpanPoolBuilder::with_capacity(tokens.len(), tokens.len() * 8);
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_MORPHOLOGY, MorphFeatures::<Tokens>::ID, "", SOURCE_VIEW)?;
        for (idx, &tag) in decoded.tags.iter().enumerate() {
            let item = index_u32(idx, "morphology token count")?;
            let label = builder.string_ref(tagset.label(Tag(tag as u16)))?;
            let draft = SpanDraft {
                item_index: item,
                tokens: Some((item, item + 1)),
                label: Some(label),
                ..SpanDraft::default()
            };
            builder.push(layer, draft)?;
        }
        Ok(builder.finish(Vec::new()))
    }

    pub fn lemmatize(&self, tokens: &[&str], tags: &[&str]) -> Result<AnalysisResultInner> {
        if tokens.len() != tags.len() {
            return Err(Error::new(INVALID_ARGUMENT, "tokens and tags must have the same count"));
        }
        if tokens.is_empty() {
            return Ok(Self::empty());
        }
        let heads = self.heads("lemmatization", "lemmas")?;
        // A lemma head that reads tags reads the caller's, which must be
        // its tagger's labels; one that reads none lemmatizes the tokens
        // alone.
        let given: Option<Vec<Vec<u32>>> = match heads.lemma_tag_labels()? {
            None => None,
            Some(labels) => Some(vec![
                tags.iter()
                    .enumerate()
                    .map(|(i, tag)| {
                        labels.iter().position(|l| l == tag).map(|p| p as u32).ok_or_else(|| {
                            Error::new(INVALID_ARGUMENT, format!("tags[{i}] {tag:?} is not a label of the tag head"))
                        })
                    })
                    .collect::<Result<Vec<u32>>>()?,
            ]),
        };
        let want = Want { lemmas: true, tags: given.as_deref(), ..Want::default() };
        let lemmas = heads.run(&[tokens.to_vec()], &want)?.lemmas.expect("asked for").remove(0);
        let mut builder = SpanPoolBuilder::with_capacity(tokens.len(), tokens.len() * 8);
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_LEMMAS, Lemmas::ID, "", SOURCE_VIEW)?;
        for (idx, lemma) in lemmas.iter().enumerate() {
            let item = index_u32(idx, "lemma token count")?;
            let label = builder.string_ref(lemma)?;
            let draft = SpanDraft {
                item_index: item,
                tokens: Some((item, item + 1)),
                label: Some(label),
                ..SpanDraft::default()
            };
            builder.push(layer, draft)?;
        }
        Ok(builder.finish(Vec::new()))
    }

    pub fn find_tokens(&self, tokens: &[&str]) -> Result<AnalysisResultInner> {
        if tokens.is_empty() {
            return Ok(Self::empty());
        }
        let heads = self.heads("named entity recognition", "entities")?;
        let kinds = heads.entity_kinds()?;
        let found = heads.run(&[tokens.to_vec()], &Want { entities: true, ..Want::default() })?;
        let found = found.entities.expect("asked for").remove(0);
        let mut builder = SpanPoolBuilder::with_capacity(found.len(), found.len() * 12);
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_ENTITIES, Entities::ID, "", SOURCE_VIEW)?;
        for (idx, e) in found.iter().enumerate() {
            let label = builder.string_ref(kinds.label(Tag(e.kind as u16)))?;
            let draft = SpanDraft {
                item_index: index_u32(idx, "entity count")?,
                tokens: Some((e.start, e.end)),
                label: Some(label),
                prob: Some(e.prob),
                ..SpanDraft::default()
            };
            builder.push(layer, draft)?;
        }
        Ok(builder.finish(Vec::new()))
    }

    pub fn find_text(&self, text: &str) -> Result<AnalysisResultInner> {
        if text.is_empty() {
            return Ok(Self::empty());
        }
        let heads = self.heads("named entity recognition", "entities")?;
        let kinds = heads.entity_kinds()?;
        let offsets = OffsetTable::build(text);
        let tok_spans = self.token_spans(text);
        let tok_strs = forms(text, &tok_spans)?;
        let seqs = sequences(&tok_spans, &self.sentences_over(text, &tok_spans)?);
        let words: Vec<Vec<&str>> = seqs.iter().map(|s| s.iter().map(|&i| tok_strs[i]).collect()).collect();
        let found = heads.run(&words, &Want { entities: true, ..Want::default() })?.entities.expect("asked for");
        let mut builder = SpanPoolBuilder::default();
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_ENTITIES, Entities::ID, "", SOURCE_VIEW)?;
        let mut idx = 0;
        for (seq, found) in seqs.iter().zip(found) {
            for e in found {
                let (ts, te) = (seq[e.start as usize], seq[e.end as usize - 1] + 1);
                let coords = coordinate_range(text, &offsets, (tok_spans[ts].start, tok_spans[te - 1].end))?;
                let label = builder.string_ref(kinds.label(Tag(e.kind as u16)))?;
                let draft = SpanDraft {
                    item_index: index_u32(idx, "entity count")?,
                    view: Some(coords),
                    source: Some(coords),
                    tokens: Some((index_u32(ts, "token index")?, index_u32(te, "token index")?)),
                    label: Some(label),
                    prob: Some(e.prob),
                    ..SpanDraft::default()
                };
                builder.push(layer, draft)?;
                idx += 1;
            }
        }
        Ok(builder.finish(Vec::new()))
    }

    pub fn detect_pii(&self, text: &str) -> Result<AnalysisResultInner> {
        let offsets = OffsetTable::build(text);
        let coords = |span: Span| -> Result<_> {
            let r = coordinate_range(text, &offsets, (span.start, span.end))?;
            Ok((r, r))
        };
        let invalid = |msg: String| Error::new(BUNDLE_INVALID, msg);
        if self.pii.detail {
            let (pii, candidates) = self.pii_detector.layer_with_candidates(text, ViewId::SOURCE);
            let mut builder = SpanPoolBuilder::with_capacity(pii.spans.len(), pii.spans.len().saturating_mul(16));
            extract_pii_layer(&mut builder, &pii, "", SOURCE_VIEW, self.pii, invalid, coords)?;
            extract_pii_candidates_layer(&mut builder, &candidates, ("", ""), SOURCE_VIEW, invalid, coords)?;
            Ok(builder.finish(Vec::new()))
        } else {
            let pii = self.pii_detector.layer(text, ViewId::SOURCE);
            let mut builder = SpanPoolBuilder::with_capacity(pii.spans.len(), pii.spans.len().saturating_mul(16));
            extract_pii_layer(&mut builder, &pii, "", SOURCE_VIEW, self.pii, invalid, coords)?;
            Ok(builder.finish(Vec::new()))
        }
    }

    pub fn analyze_graph(&self, text: &str) -> Result<AnalysisResultInner> {
        let added = |e: turbo_analysis::Error| Error::new(BUNDLE_INVALID, format!("add layer: {e}"));
        let mut doc = Document::new(text);
        let tok_vec = self.token_spans(text);
        forms(text, &tok_vec)?;
        let sent_vec = self.sentences_over(text, &tok_vec)?;
        let mut sent_spans = Spans::new(ViewId::SOURCE);
        for span in &sent_vec {
            sent_spans.push(*span);
        }
        let by = match &self.heads {
            Some(heads) if heads.has_sentences() => Self::head_provenance(heads, "turbo-tagger-sentences"),
            _ => Provenance::by("turbo:sentdetect"),
        };
        doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sent_spans }, by).map_err(added)?;
        doc.add(&Key::<Tokens>::DEFAULT, layers::tokens_of(&tok_vec), Provenance::by("turbo:tokenize"))
            .map_err(added)?;
        self.add_head_layers(&mut doc, text, &tok_vec, &sent_vec)?;
        if self.pii.detail {
            let (pii, candidates) = self.pii_detector.layer_with_candidates(text, ViewId::SOURCE);
            doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("turbo:pii")).map_err(added)?;
            let mut cand_prov = Provenance::by("turbo:pii");
            cand_prov.on.push(Key::<Pii>::DEFAULT.id());
            doc.add(&Key::<PiiCandidates>::DEFAULT, candidates, cand_prov).map_err(added)?;
        } else {
            let pii = self.pii_detector.layer(text, ViewId::SOURCE);
            doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("turbo:pii")).map_err(added)?;
        }

        let graph = export(&doc, &self.codecs).map_err(|e| Error::new(BUNDLE_INVALID, format!("export graph: {e}")))?;
        let wire = graph.to_bytes().map_err(|e| Error::new(BUNDLE_INVALID, format!("encode graph: {e}")))?;
        extract_graph(&doc, wire, Origin::Analyzed, self.pii)
    }

    pub fn inspect_graph(&self, graph_bytes: &[u8]) -> Result<AnalysisResultInner> {
        let graph = DocumentGraph::from_bytes(graph_bytes)
            .map_err(|e| Error::new(INVALID_ARGUMENT, format!("invalid DocumentGraph wire bytes: {e}")))?;
        let doc = import(&graph, &self.codecs)
            .map_err(|e| Error::new(INVALID_ARGUMENT, format!("DocumentGraph import failed: {e}")))?;
        extract_graph(&doc, graph_bytes.to_vec(), Origin::Caller, self.pii)
    }
}

#[cfg(test)]
mod tests;
