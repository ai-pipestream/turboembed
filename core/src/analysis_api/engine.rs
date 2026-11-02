//! In-process analysis engine backing `turbo_analyzer`: the rule sentence
//! splitter and rule tokenizer, the PII detector, and the heads of the
//! bundle the descriptor names (heads/), run a sentence at a time.

use std::sync::Arc;

use turbo_analysis::{
    Codecs, Document, DocumentGraph, Execution, IxRange, Key, Layer, ModelRef, OffsetTable, Provenance, Span, Spans,
    Tag, TagSet, ViewId, export, import,
};
use turbo_analysis_layers::export::register_codecs;
use turbo_analysis_layers::{
    Dependencies, Entities, Lemmas, MorphFeatures, Pii, PiiConfidence, PosTags, Sentences, Strs, Tokens,
};
use turbo_pii::{Pack, PiiDetector};
use turbo_sentdetect::RuleSplitter;
use turbo_tokenize::{TokenizerOptions, simple_spans};

use super::graph::extract_graph;
use super::heads::{BundleHeads, Want, unsupported};
use super::{
    AnalysisResultInner, SpanDraft, SpanPoolBuilder, TURBO_ANALYSIS_LAYER_ENTITIES, TURBO_ANALYSIS_LAYER_LEMMAS,
    TURBO_ANALYSIS_LAYER_MORPHOLOGY, TURBO_ANALYSIS_LAYER_PII, TURBO_ANALYSIS_LAYER_POS, TURBO_PII_CONFIDENCE_CHECKSUM,
    TURBO_PII_CONFIDENCE_FORMAT, TURBO_PII_CONFIDENCE_LABELLED, coordinate_range, turbo_analyzer_desc,
};
use crate::ffi::text;
use crate::status::{BUNDLE_INVALID, CAPACITY, Error, INVALID_ARGUMENT, Result};

mod direct;

pub struct AnalyzerEngine {
    rule_splitter: RuleSplitter,
    heads: Option<BundleHeads>,
    pii_detector: PiiDetector,
    codecs: Codecs,
}

fn index_u32(index: usize, what: &str) -> Result<u32> {
    u32::try_from(index).map_err(|_| Error::new(CAPACITY, format!("{what} exceeds u32")))
}

const SOURCE_VIEW: u32 = ViewId::SOURCE.0 as u32;

/// The retired model path fields, each with its field number in the
/// header's numbering.
/// `turbo_analyzer_desc.bundle_path` as `turbo_error.field` numbers it.
const BUNDLE_PATH_FIELD: u32 = 7;

const RETIRED: [(&str, u32); 5] = [
    ("sent_model_path", 2),
    ("tok_model_path", 3),
    ("pos_model_path", 4),
    ("lemma_model_path", 5),
    ("ner_model_path", 6),
];

/// Each token's text, checked to lie in `text` on character boundaries.
fn forms<'t>(text: &'t str, tokens: &[Span]) -> Result<Vec<&'t str>> {
    tokens
        .iter()
        .map(|s| {
            let (start, end) = (s.start as usize, s.end as usize);
            if start > end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                return Err(Error::new(BUNDLE_INVALID, "tokenizer returned a span outside the text"));
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
    /// The engine `desc` configures: `with_bundle` says whether the
    /// descriptor is long enough to carry `bundle_path`.
    pub(crate) unsafe fn from_desc(desc: Option<(&turbo_analyzer_desc, bool)>) -> Result<Self> {
        let (lang, bundle) = match desc {
            Some((d, with_bundle)) => unsafe {
                let lang = text(d.language, "language")?;
                let paths =
                    [d.sent_model_path, d.tok_model_path, d.pos_model_path, d.lemma_model_path, d.ner_model_path];
                for (value, (name, field)) in paths.iter().zip(RETIRED) {
                    if value.len != 0 {
                        return Err(Error::field(
                            INVALID_ARGUMENT,
                            field,
                            format!("turbo_analyzer_desc.{name} is retired: learned components load from bundle_path"),
                        ));
                    }
                }
                let bundle = if with_bundle { text(d.bundle_path, "bundle_path")? } else { "" };
                (if lang.is_empty() { "en" } else { lang }, bundle)
            },
            None => ("en", ""),
        };
        let heads = match bundle {
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
            rule_splitter: RuleSplitter::new(lang),
            heads,
            pii_detector: PiiDetector::pack(Pack::AllStructured),
            codecs,
        })
    }

    fn sentence_spans(&self, text: &str) -> Vec<Span> {
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
        let seqs = sequences(&tok_spans, &self.sentence_spans(text));
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
        let pii = self.pii_detector.layer(text, ViewId::SOURCE);
        let mut builder = SpanPoolBuilder::with_capacity(pii.spans.len(), pii.spans.len().saturating_mul(16));
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_PII, Pii::ID, "", SOURCE_VIEW)?;
        for (idx, mention) in pii.mentions().enumerate() {
            let coords = coordinate_range(text, &offsets, (mention.span.start, mention.span.end))?;
            let label = builder.string_ref(mention.kind.name())?;
            let value = builder.string_ref(mention.normalized)?;
            let classification = match mention.confidence {
                PiiConfidence::Format => TURBO_PII_CONFIDENCE_FORMAT,
                PiiConfidence::Checksum => TURBO_PII_CONFIDENCE_CHECKSUM,
                PiiConfidence::LabelledChecksum => TURBO_PII_CONFIDENCE_LABELLED,
            };
            builder.push(
                layer,
                SpanDraft {
                    item_index: index_u32(idx, "PII mention count")?,
                    classification,
                    view: Some(coords),
                    source: Some(coords),
                    label: Some(label),
                    value: Some(value),
                    ..SpanDraft::default()
                },
            )?;
        }
        Ok(builder.finish(Vec::new()))
    }

    /// The provenance of a layer the bundle's heads wrote.
    fn head_provenance(heads: &BundleHeads, annotator: &'static str) -> Provenance {
        let mut p = Provenance::by(annotator);
        p.model = Some(ModelRef {
            id: heads.model_id.clone(),
            revision: heads.model_revision.clone(),
            sha256: heads.manifest_sha256.clone(),
        });
        p.execution = Some(Execution {
            backend: heads.backend.clone(),
            device: heads.device.clone(),
            precision: "f32 (model)".into(),
        });
        p
    }

    /// Every layer the bundle's heads write, added to `doc` over its
    /// tokens, read a sentence at a time.
    fn add_head_layers(&self, doc: &mut Document, text: &str, tok_spans: &[Span], sent_spans: &[Span]) -> Result<()> {
        let Some(heads) = &self.heads else { return Ok(()) };
        let added = |e: turbo_analysis::Error| Error::new(BUNDLE_INVALID, format!("add layer: {e}"));
        let tok_strs = forms(text, tok_spans)?;
        let seqs = sequences(tok_spans, sent_spans);
        let words: Vec<Vec<&str>> = seqs.iter().map(|s| s.iter().map(|&i| tok_strs[i]).collect()).collect();
        let written = heads.run(&words, &heads.has())?;
        let n = tok_spans.len();
        if let Some(decoded) = written.pos {
            let (mut tags, mut prob) = (vec![Tag(0); n], vec![0.0; n]);
            for (s, d) in seqs.iter().zip(&decoded) {
                for (k, &i) in s.iter().enumerate() {
                    (tags[i], prob[i]) = (Tag(d.tags[k] as u16), d.prob[k]);
                }
            }
            let tagset: Arc<TagSet> = heads.pos_tagset()?.clone();
            let layer = PosTags { tagset, tags, prob: Some(prob) };
            doc.add(&Key::<PosTags>::DEFAULT, layer, Self::head_provenance(heads, "turbo-tagger-pos"))
                .map_err(added)?;
        }
        if let Some(decoded) = written.morphology {
            let tagset: Arc<TagSet> = heads.morph_tagset()?.clone();
            let mut of: Vec<&str> = vec![""; n];
            for (s, d) in seqs.iter().zip(&decoded) {
                for (k, &i) in s.iter().enumerate() {
                    of[i] = tagset.label(Tag(d.tags[k] as u16));
                }
            }
            let mut feats = Strs::default();
            for f in of {
                feats.push(f);
            }
            doc.add(
                &Key::<MorphFeatures<Tokens>>::DEFAULT,
                MorphFeatures::new(feats),
                Self::head_provenance(heads, "turbo-tagger-morphology"),
            )
            .map_err(added)?;
        }
        if let Some(lemmas) = written.lemmas {
            let mut of: Vec<&str> = vec![""; n];
            for (s, l) in seqs.iter().zip(&lemmas) {
                for (&i, lemma) in s.iter().zip(l) {
                    of[i] = lemma;
                }
            }
            let mut lemma = Strs::default();
            for l in of {
                lemma.push(l);
            }
            doc.add(&Key::<Lemmas>::DEFAULT, Lemmas { lemma }, Self::head_provenance(heads, "turbo-tagger-lemmas"))
                .map_err(added)?;
        }
        if let Some(parses) = written.dependencies {
            let (mut head, mut relation) = (vec![None; n], vec![Tag(0); n]);
            for (s, p) in seqs.iter().zip(&parses) {
                for (k, &i) in s.iter().enumerate() {
                    let h = p.heads[k] as usize;
                    head[i] = (h > 0).then(|| turbo_analysis::Ix::new(s[h - 1] as u32));
                    relation[i] = Tag(p.relations[k] as u16);
                }
            }
            let layer = Dependencies::<Tokens> { head, relation, relations: heads.relations()?.clone() };
            doc.add(
                &Key::<Dependencies<Tokens>>::DEFAULT,
                layer,
                Self::head_provenance(heads, "turbo-tagger-dependencies"),
            )
            .map_err(added)?;
        }
        if let Some(found) = written.entities {
            let mut layer = Entities {
                spans: Spans::new(ViewId::SOURCE),
                tokens: Vec::new(),
                kind: Vec::new(),
                kinds: heads.entity_kinds()?.clone(),
                score: Vec::new(),
            };
            for (s, found) in seqs.iter().zip(&found) {
                for e in found {
                    let (a, b) = (s[e.start as usize], s[e.end as usize - 1]);
                    layer.spans.push(Span::new(tok_spans[a].start, tok_spans[b].end));
                    layer.tokens.push(IxRange::new(a as u32, b as u32 + 1));
                    layer.kind.push(Tag(e.kind as u16));
                    layer.score.push(e.prob);
                }
            }
            doc.add(&Key::<Entities>::DEFAULT, layer, Self::head_provenance(heads, "turbo-tagger-entities"))
                .map_err(added)?;
        }
        Ok(())
    }

    pub fn analyze_graph(&self, text: &str) -> Result<AnalysisResultInner> {
        let added = |e: turbo_analysis::Error| Error::new(BUNDLE_INVALID, format!("add layer: {e}"));
        let mut doc = Document::new(text);
        let sent_vec = self.sentence_spans(text);
        let mut sent_spans = Spans::new(ViewId::SOURCE);
        for span in &sent_vec {
            sent_spans.push(*span);
        }
        doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("turbo:sentdetect"))
            .map_err(added)?;
        let tok_vec = self.token_spans(text);
        forms(text, &tok_vec)?;
        let mut tok_spans = Spans::new(ViewId::SOURCE);
        for span in &tok_vec {
            tok_spans.push(*span);
        }
        doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("turbo:tokenize"))
            .map_err(added)?;
        self.add_head_layers(&mut doc, text, &tok_vec, &sent_vec)?;
        let pii = self.pii_detector.layer(text, ViewId::SOURCE);
        doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("turbo:pii")).map_err(added)?;

        let graph = export(&doc, &self.codecs).map_err(|e| Error::new(BUNDLE_INVALID, format!("export graph: {e}")))?;
        let wire = graph.to_bytes().map_err(|e| Error::new(BUNDLE_INVALID, format!("encode graph: {e}")))?;
        extract_graph(&doc, wire)
    }

    pub fn inspect_graph(&self, graph_bytes: &[u8]) -> Result<AnalysisResultInner> {
        let graph = DocumentGraph::from_bytes(graph_bytes)
            .map_err(|e| Error::new(INVALID_ARGUMENT, format!("invalid DocumentGraph wire bytes: {e}")))?;
        let doc = import(&graph, &self.codecs)
            .map_err(|e| Error::new(INVALID_ARGUMENT, format!("DocumentGraph import failed: {e}")))?;
        extract_graph(&doc, graph_bytes.to_vec())
    }
}
