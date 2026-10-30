//! In-process analysis engine backing `turbo_analyzer`.

use std::path::Path;

use turbo_analysis::{
    Codecs, Document, DocumentGraph, IxRange, Key, Layer, OffsetTable, Provenance, Span, Spans, Tag, TagSet, ViewId,
    export, import,
};
use turbo_analysis_layers::export::register_codecs;
use turbo_analysis_layers::{Confidence, Entities, Lemmas, Pii, PiiConfidence, PosTags, Sentences, Strs, Tokens};
use turbo_lemmatizer::{DictionaryLemmatizer, LemmatizerModel, MaxentLemmatizer};
use turbo_pii::{Pack, PiiDetector};
use turbo_sentdetect::{RuleSplitter, SentenceDetector, SentenceModel};
use turbo_sequence::TokenTag;
use turbo_tokenize::me::{TokenizerMe, TokenizerModel};
use turbo_tokenize::{TokenizerOptions, simple_spans};

use super::graph::extract_graph;
use super::models::{LemmaOutcome, NerEngine, PosEngine, PosOutcome, read_model_file};
use super::{
    AnalysisResultInner, SpanDraft, SpanPoolBuilder, TURBO_ANALYSIS_LAYER_ENTITIES, TURBO_ANALYSIS_LAYER_LEMMAS,
    TURBO_ANALYSIS_LAYER_PII, TURBO_ANALYSIS_LAYER_POS, TURBO_PII_CONFIDENCE_CHECKSUM, TURBO_PII_CONFIDENCE_FORMAT,
    TURBO_PII_CONFIDENCE_LABELLED, coordinate_range, turbo_analyzer_desc,
};
use crate::ffi::text;
use crate::status::{BUNDLE_INVALID, BUNDLE_NOT_FOUND, CAPACITY, Error, INVALID_ARGUMENT, Result, UNSUPPORTED_TASK};

mod direct;

pub struct AnalyzerEngine {
    sentence_model: Option<SentenceDetector>,
    rule_splitter: RuleSplitter,
    tokenizer_model: Option<TokenizerMe>,
    pos_engine: Option<PosEngine>,
    dict_lemmatizer: Option<DictionaryLemmatizer>,
    maxent_lemmatizer: Option<MaxentLemmatizer>,
    ner_engine: Option<NerEngine>,
    pii_detector: PiiDetector,
    codecs: Codecs,
}

fn unsupported(task: &str, field: &str) -> Error {
    Error::new(UNSUPPORTED_TASK, format!("{task} requires {field}; no fallback task is substituted"))
}

fn index_u32(index: usize, what: &str) -> Result<u32> {
    u32::try_from(index).map_err(|_| Error::new(CAPACITY, format!("{what} exceeds u32")))
}

const SOURCE_VIEW: u32 = ViewId::SOURCE.0 as u32;

impl AnalyzerEngine {
    pub(crate) unsafe fn from_desc(desc: Option<&turbo_analyzer_desc>) -> Result<Self> {
        let (lang, sent_path, tok_path, pos_path, lemma_path, ner_path) = match desc {
            Some(d) => unsafe {
                let lang = text(d.language, "language")?;
                let sent = text(d.sent_model_path, "sent_model_path")?;
                let tok = text(d.tok_model_path, "tok_model_path")?;
                let pos = text(d.pos_model_path, "pos_model_path")?;
                let lemma = text(d.lemma_model_path, "lemma_model_path")?;
                let ner = text(d.ner_model_path, "ner_model_path")?;
                (if lang.is_empty() { "en" } else { lang }, sent, tok, pos, lemma, ner)
            },
            None => ("en", "", "", "", "", ""),
        };

        let sentence_model = if sent_path.is_empty() {
            None
        } else {
            let file = read_model_file(sent_path, "sent_model_path")?;
            let model =
                SentenceModel::open(file).map_err(|e| Error::new(BUNDLE_INVALID, format!("sent_model_path: {e}")))?;
            Some(SentenceDetector::new(model))
        };
        let rule_splitter = RuleSplitter::new(lang);

        let tokenizer_model = if tok_path.is_empty() {
            None
        } else {
            let file = read_model_file(tok_path, "tok_model_path")?;
            let model =
                TokenizerModel::open(file).map_err(|e| Error::new(BUNDLE_INVALID, format!("tok_model_path: {e}")))?;
            Some(TokenizerMe::new(model, TokenizerOptions::default()))
        };

        let pos_engine = if pos_path.is_empty() {
            None
        } else {
            Some(PosEngine::open(read_model_file(pos_path, "pos_model_path")?)?)
        };

        let (dict_lemmatizer, maxent_lemmatizer) = if lemma_path.is_empty() {
            (None, None)
        } else if lemma_path.ends_with(".txt") || lemma_path.ends_with(".tsv") || lemma_path.ends_with(".dict") {
            let p = Path::new(lemma_path);
            if !p.exists() {
                return Err(Error::new(BUNDLE_NOT_FOUND, format!("lemma_model_path: file not found at {lemma_path}")));
            }
            let dictionary = DictionaryLemmatizer::read(p, "dictionary")
                .map_err(|e| Error::new(BUNDLE_INVALID, format!("lemma_model_path: {e}")))?;
            (Some(dictionary), None)
        } else {
            let file = read_model_file(lemma_path, "lemma_model_path")?;
            let model = LemmatizerModel::open(file)
                .map_err(|e| Error::new(BUNDLE_INVALID, format!("lemma_model_path: {e}")))?;
            (None, Some(MaxentLemmatizer::new(model)))
        };

        let ner_engine = if ner_path.is_empty() {
            None
        } else {
            Some(NerEngine::open(read_model_file(ner_path, "ner_model_path")?)?)
        };

        let pii_detector = PiiDetector::pack(Pack::AllStructured);
        let mut codecs = Codecs::new();
        register_codecs(&mut codecs);

        Ok(Self {
            sentence_model,
            rule_splitter,
            tokenizer_model,
            pos_engine,
            dict_lemmatizer,
            maxent_lemmatizer,
            ner_engine,
            pii_detector,
            codecs,
        })
    }

    pub(crate) fn sentence_spans(&self, text: &str) -> Vec<(Span, Option<f64>)> {
        let mut out = Vec::new();
        if let Some(me) = &self.sentence_model {
            me.detect(text, &mut out);
            out.into_iter().map(|(span, prob)| (span, Some(prob))).collect()
        } else {
            let mut spans = Vec::new();
            self.rule_splitter.split(text, &mut spans);
            spans.into_iter().map(|span| (span, None)).collect()
        }
    }

    pub(crate) fn token_spans(&self, text: &str) -> Vec<(Span, Option<f64>)> {
        let mut out = Vec::new();
        if let Some(me) = &self.tokenizer_model {
            me.tokenize(text, &mut out);
            out.into_iter().map(|(span, prob)| (span, Some(prob))).collect()
        } else {
            let mut spans = Vec::new();
            simple_spans(text, TokenizerOptions::default(), &mut spans);
            spans.into_iter().map(|span| (span, None)).collect()
        }
    }

    fn pos_outcomes(&self, tokens: &[&str]) -> Result<Vec<PosOutcome>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let Some(pos) = &self.pos_engine else {
            return Err(unsupported("POS tagging", "pos_model_path"));
        };
        pos.tag_tokens(tokens)
    }

    pub fn pos_tag(&self, tokens: &[&str]) -> Result<AnalysisResultInner> {
        let tagged = self.pos_outcomes(tokens)?;
        let mut builder = SpanPoolBuilder::with_capacity(tagged.len(), tagged.len() * 6);
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_POS, PosTags::ID, "", SOURCE_VIEW)?;
        for (idx, outcome) in tagged.into_iter().enumerate() {
            let item = index_u32(idx, "POS token count")?;
            let label = match outcome.tag.as_deref() {
                Some(tag) => Some(builder.string_ref(tag)?),
                None => None,
            };
            builder.push(
                layer,
                SpanDraft {
                    item_index: item,
                    tokens: Some((item, item + 1)),
                    label,
                    prob: outcome.prob,
                    ..SpanDraft::default()
                },
            )?;
        }
        Ok(builder.finish(Vec::new()))
    }

    fn lemma_outcomes(&self, tokens: &[&str], tags: &[&str]) -> Result<Vec<LemmaOutcome>> {
        if tokens.len() != tags.len() {
            return Err(Error::new(INVALID_ARGUMENT, "tokens and tags must have the same count"));
        }
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(dict) = &self.dict_lemmatizer {
            return Ok(tokens
                .iter()
                .zip(tags.iter())
                .map(|(&word, &tag)| LemmaOutcome {
                    lemma: dict.lemmas(word, tag).and_then(|lemmas| lemmas.first().cloned()),
                    prob: None,
                })
                .collect());
        }
        let Some(lemmatizer) = &self.maxent_lemmatizer else {
            return Err(unsupported("lemmatization", "lemma_model_path"));
        };
        let pairs: Vec<TokenTag<'_>> =
            tokens.iter().zip(tags.iter()).map(|(&token, &tag)| TokenTag { token, tag }).collect();
        let mut model_lemmas = Strs::default();
        let mut model_probs = Vec::with_capacity(tokens.len());
        let has_model = lemmatizer
            .lemmatize(&pairs, &mut model_lemmas, &mut model_probs)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("lemmatize: {e}")))?;
        if !has_model {
            return Ok(tokens.iter().map(|_| LemmaOutcome::unknown()).collect());
        }
        if model_lemmas.len() != tokens.len() || model_probs.len() != tokens.len() {
            return Err(Error::new(BUNDLE_INVALID, "lemmatizer returned an outcome count mismatch"));
        }
        Ok(model_lemmas
            .iter()
            .zip(model_probs)
            .map(|(lemma, prob)| LemmaOutcome { lemma: Some(lemma.to_owned()), prob: Some(prob) })
            .collect())
    }

    pub fn lemmatize(&self, tokens: &[&str], tags: &[&str]) -> Result<AnalysisResultInner> {
        let outcomes = self.lemma_outcomes(tokens, tags)?;
        let mut builder = SpanPoolBuilder::with_capacity(outcomes.len(), outcomes.len() * 8);
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_LEMMAS, Lemmas::ID, "", SOURCE_VIEW)?;
        for (idx, outcome) in outcomes.into_iter().enumerate() {
            let item = index_u32(idx, "lemma token count")?;
            let label = match outcome.lemma.as_deref() {
                Some(lemma) => Some(builder.string_ref(lemma)?),
                None => None,
            };
            builder.push(
                layer,
                SpanDraft {
                    item_index: item,
                    tokens: Some((item, item + 1)),
                    label,
                    prob: outcome.prob,
                    ..SpanDraft::default()
                },
            )?;
        }
        Ok(builder.finish(Vec::new()))
    }

    fn model_entities(&self, tokens: &[&str]) -> Result<Vec<(u32, u32, String, f64)>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let Some(ner) = &self.ner_engine else {
            return Err(unsupported("named entity recognition", "ner_model_path"));
        };
        ner.find(tokens)
    }

    pub fn find_tokens(&self, tokens: &[&str]) -> Result<AnalysisResultInner> {
        let entities = self.model_entities(tokens)?;
        let mut builder = SpanPoolBuilder::with_capacity(entities.len(), entities.len() * 12);
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_ENTITIES, Entities::ID, "", SOURCE_VIEW)?;
        for (idx, (start, end, kind, prob)) in entities.into_iter().enumerate() {
            let item = index_u32(idx, "entity count")?;
            let label = builder.string_ref(&kind)?;
            builder.push(
                layer,
                SpanDraft {
                    item_index: item,
                    tokens: Some((start, end)),
                    label: Some(label),
                    prob: Some(prob),
                    ..SpanDraft::default()
                },
            )?;
        }
        Ok(builder.finish(Vec::new()))
    }

    pub fn find_text(&self, text: &str) -> Result<AnalysisResultInner> {
        let offsets = OffsetTable::build(text);
        let tok_spans = self.token_spans(text);
        let tok_strs: Vec<&str> = tok_spans
            .iter()
            .map(|&(s, _)| {
                let (start, end) = (s.start as usize, s.end as usize);
                if start > end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                    return Err(Error::new(BUNDLE_INVALID, "tokenizer returned a span outside the text"));
                }
                Ok(&text[start..end])
            })
            .collect::<Result<Vec<_>>>()?;
        let entities = self.model_entities(&tok_strs)?;
        let mut builder = SpanPoolBuilder::with_capacity(entities.len(), entities.len() * 12);
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_ENTITIES, Entities::ID, "", SOURCE_VIEW)?;
        for (idx, (ts, te, kind, prob)) in entities.into_iter().enumerate() {
            if ts >= te || te as usize > tok_spans.len() {
                return Err(Error::new(BUNDLE_INVALID, "entity token range is outside the token sequence"));
            }
            let first = tok_spans[ts as usize].0;
            let last = tok_spans[te as usize - 1].0;
            let coords = coordinate_range(text, &offsets, (first.start, last.end))?;
            let label = builder.string_ref(&kind)?;
            builder.push(
                layer,
                SpanDraft {
                    item_index: index_u32(idx, "entity count")?,
                    view: Some(coords),
                    source: Some(coords),
                    tokens: Some((ts, te)),
                    label: Some(label),
                    prob: Some(prob),
                    ..SpanDraft::default()
                },
            )?;
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

    pub fn analyze_graph(&self, text: &str) -> Result<AnalysisResultInner> {
        let mut doc = Document::new(text);
        let mut sent_spans = Spans::new(ViewId::SOURCE);
        let mut sent_probs = Vec::new();
        let mut sent_has_probs = true;
        for (span, prob) in self.sentence_spans(text) {
            sent_spans.push(span);
            match prob {
                Some(prob) => sent_probs.push(prob),
                None => sent_has_probs = false,
            }
        }
        doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("turbo:sentdetect"))
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add sentences: {e}")))?;
        if sent_has_probs {
            doc.add(
                &Key::<Confidence<Sentences>>::DEFAULT,
                Confidence::<Sentences>::new(sent_probs),
                Provenance::by("turbo:sentdetect"),
            )
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add sentence confidence: {e}")))?;
        }

        let tok_vec = self.token_spans(text);
        let mut tok_spans = Spans::new(ViewId::SOURCE);
        let mut tok_strs = Vec::with_capacity(tok_vec.len());
        for (span, _) in &tok_vec {
            let (start, end) = (span.start as usize, span.end as usize);
            if start > end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                return Err(Error::new(BUNDLE_INVALID, "tokenizer returned a span outside the text"));
            }
            tok_spans.push(*span);
            tok_strs.push(&text[start..end]);
        }
        doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("turbo:tokenize"))
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add tokens: {e}")))?;
        if tok_vec.iter().all(|(_, prob)| prob.is_some()) {
            let token_probs = tok_vec.iter().map(|(_, prob)| prob.expect("checked present")).collect();
            doc.add(
                &Key::<Confidence<Tokens>>::DEFAULT,
                Confidence::<Tokens>::new(token_probs),
                Provenance::by("turbo:tokenize"),
            )
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add token confidence: {e}")))?;
        }

        if let Some(pos_engine) = &self.pos_engine {
            let tagged = self.pos_outcomes(&tok_strs)?;
            if tagged.iter().all(|outcome| outcome.tag.is_some()) {
                let tag_labels: Vec<String> =
                    tagged.iter().map(|outcome| outcome.tag.clone().expect("checked known")).collect();
                let tagset = pos_engine.tagset();
                let tags: Vec<Tag> = tag_labels
                    .iter()
                    .map(|tag| {
                        tagset.tag(tag).ok_or_else(|| {
                            Error::new(BUNDLE_INVALID, format!("POS outcome {tag} is outside its tag set"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let probs: Vec<f64> = tagged.iter().map(|outcome| outcome.prob.unwrap_or(f64::NAN)).collect();
                let prob = tagged.iter().all(|outcome| outcome.prob.is_some()).then_some(probs);
                doc.add(&Key::<PosTags>::DEFAULT, PosTags { tagset, tags, prob }, Provenance::by("turbo:postag"))
                    .map_err(|e| Error::new(BUNDLE_INVALID, format!("add pos: {e}")))?;

                if self.dict_lemmatizer.is_some() || self.maxent_lemmatizer.is_some() {
                    let tag_refs: Vec<&str> = tag_labels.iter().map(String::as_str).collect();
                    let outcomes = self.lemma_outcomes(&tok_strs, &tag_refs)?;
                    let mut lemma_strs = Strs::default();
                    for outcome in &outcomes {
                        lemma_strs.push(outcome.lemma.as_deref().unwrap_or("_"));
                    }
                    doc.add(&Key::<Lemmas>::DEFAULT, Lemmas { lemma: lemma_strs }, Provenance::by("turbo:lemmatizer"))
                        .map_err(|e| Error::new(BUNDLE_INVALID, format!("add lemmas: {e}")))?;
                    if outcomes.iter().all(|outcome| outcome.prob.is_some()) {
                        let lemma_probs =
                            outcomes.iter().map(|outcome| outcome.prob.expect("checked present")).collect();
                        doc.add(
                            &Key::<Confidence<Lemmas>>::DEFAULT,
                            Confidence::<Lemmas>::new(lemma_probs),
                            Provenance::by("turbo:lemmatizer"),
                        )
                        .map_err(|e| Error::new(BUNDLE_INVALID, format!("add lemma confidence: {e}")))?;
                    }
                }
            }
        }

        if self.ner_engine.is_some() {
            let ent_res = self.model_entities(&tok_strs)?;
            let mut ent_kinds: Vec<&str> = ent_res.iter().map(|(_, _, kind, _)| kind.as_str()).collect();
            ent_kinds.sort_unstable();
            ent_kinds.dedup();
            let kinds = TagSet::new("entities", &ent_kinds);
            let mut ent_spans = Spans::new(ViewId::SOURCE);
            let mut ent_tokens = Vec::new();
            let mut ent_kind = Vec::new();
            let mut ent_score = Vec::new();
            for (ts, te, kind, prob) in ent_res {
                if ts >= te || te as usize > tok_vec.len() {
                    return Err(Error::new(BUNDLE_INVALID, "entity token range is outside the token sequence"));
                }
                ent_spans.push(Span::new(tok_vec[ts as usize].0.start, tok_vec[te as usize - 1].0.end));
                ent_tokens.push(IxRange::new(ts, te));
                ent_kind.push(kinds.tag(&kind).unwrap_or(Tag(0)));
                ent_score.push(prob);
            }
            doc.add(
                &Key::<Entities>::DEFAULT,
                Entities { spans: ent_spans, tokens: ent_tokens, kind: ent_kind, kinds, score: ent_score },
                Provenance::by("turbo:namefind"),
            )
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add entities: {e}")))?;
        }

        let pii = self.pii_detector.layer(text, ViewId::SOURCE);
        doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("turbo:pii"))
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add pii: {e}")))?;
        self.attach_default_chunks(&mut doc)?;

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
