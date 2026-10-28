//! In-process analysis engine backing `turbo_analyzer`.

use std::path::Path;

use turbo_analysis::{
    Codecs, Document, DocumentGraph, Ix, IxRange, Key, OffsetTable, Provenance, Span, Spans, Tag, TagSet, ViewId,
    export, import,
};
use turbo_analysis_layers::export::register_codecs;
use turbo_analysis_layers::{Entities, Lemmas, Pii, PosTags, Sentences, Strs, Tokens};
use turbo_lemmatizer::{DictionaryLemmatizer, LemmatizerModel, MaxentLemmatizer, NOT_FOUND};
use turbo_maxent::{Dictionary, ModelFile};
use turbo_namefind::{BuiltIn, DictionaryNameFinder, RegexNameFinder, TokenNameFinder};
use turbo_pii::{Pack, PiiDetector};
use turbo_sentdetect::{RuleSplitter, SentenceDetector, SentenceModel, WhitespaceMode};
use turbo_sequence::TokenTag;
use turbo_tokenize::me::{TokenizerMe, TokenizerModel};
use turbo_tokenize::{TokenizerOptions, simple_spans};

use super::models::{
    DEFAULT_LEMMA_DICT, DEFAULT_LEMMA_MODEL, DEFAULT_NER_MODEL, DEFAULT_PEOPLE_XML, DEFAULT_PLACES_XML,
    DEFAULT_POS_MODEL, NerEngine, PosEngine, read_model_file,
};
use super::{AnalysisResultInner, SpanPoolBuilder, turbo_analyzer_desc};
use crate::ffi::text;
use crate::status::{BUNDLE_INVALID, BUNDLE_NOT_FOUND, Error, INVALID_ARGUMENT, Result};

pub struct AnalyzerEngine {
    sentence_model: Option<SentenceDetector>,
    rule_splitter: RuleSplitter,
    tokenizer_model: Option<TokenizerMe>,
    pos_engine: PosEngine,
    dict_lemmatizer: Option<DictionaryLemmatizer>,
    maxent_lemmatizer: MaxentLemmatizer,
    ner_engine: NerEngine,
    person_dict: DictionaryNameFinder,
    place_dict: DictionaryNameFinder,
    regex_finder: RegexNameFinder,
    pii_detector: PiiDetector,
    codecs: Codecs,
}

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
            Some(SentenceDetector::new(model, WhitespaceMode::Unicode))
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

        let pos_file = if pos_path.is_empty() {
            ModelFile::open(DEFAULT_POS_MODEL.to_vec())
                .map_err(|e| Error::new(BUNDLE_INVALID, format!("default POS: {e}")))?
        } else {
            read_model_file(pos_path, "pos_model_path")?
        };
        let pos_engine = PosEngine::open(pos_file)?;

        let default_lemma_file = ModelFile::open(DEFAULT_LEMMA_MODEL.to_vec())
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("default lemmatizer: {e}")))?;
        let default_lemma_model = LemmatizerModel::open(default_lemma_file)
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("default lemmatizer model: {e}")))?;
        let (dict_lemmatizer, maxent_lemmatizer) = if lemma_path.is_empty() {
            let dict = DictionaryLemmatizer::parse(DEFAULT_LEMMA_DICT, lang)
                .map_err(|e| Error::new(BUNDLE_INVALID, format!("default lemma dictionary: {e}")))?;
            (Some(dict), MaxentLemmatizer::new(default_lemma_model))
        } else if lemma_path.ends_with(".txt") || lemma_path.ends_with(".tsv") || lemma_path.ends_with(".dict") {
            let p = Path::new(lemma_path);
            if !p.exists() {
                return Err(Error::new(BUNDLE_NOT_FOUND, format!("lemma_model_path: file not found at {lemma_path}")));
            }
            let dict = DictionaryLemmatizer::read(p, lang)
                .map_err(|e| Error::new(BUNDLE_INVALID, format!("lemma_model_path: {e}")))?;
            (Some(dict), MaxentLemmatizer::new(default_lemma_model))
        } else {
            let file = read_model_file(lemma_path, "lemma_model_path")?;
            let model = LemmatizerModel::open(file)
                .map_err(|e| Error::new(BUNDLE_INVALID, format!("lemma_model_path: {e}")))?;
            (None, MaxentLemmatizer::new(model))
        };

        let ner_file = if ner_path.is_empty() {
            ModelFile::open(DEFAULT_NER_MODEL.to_vec())
                .map_err(|e| Error::new(BUNDLE_INVALID, format!("default NER: {e}")))?
        } else {
            read_model_file(ner_path, "ner_model_path")?
        };
        let ner_engine = NerEngine::open(ner_file)?;

        let person_dict = DictionaryNameFinder::new(
            Dictionary::read(DEFAULT_PEOPLE_XML)
                .map_err(|e| Error::new(BUNDLE_INVALID, format!("people dictionary: {e}")))?,
            "person",
        );
        let place_dict = DictionaryNameFinder::new(
            Dictionary::read(DEFAULT_PLACES_XML)
                .map_err(|e| Error::new(BUNDLE_INVALID, format!("places dictionary: {e}")))?,
            "location",
        );
        let regex_finder = RegexNameFinder::defaults(&BuiltIn::ALL);
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
            person_dict,
            place_dict,
            regex_finder,
            pii_detector,
            codecs,
        })
    }

    fn sentence_spans(&self, text: &str) -> Vec<(Span, f64)> {
        let mut out = Vec::new();
        if let Some(me) = &self.sentence_model {
            me.detect(text, &mut out);
        } else {
            let mut spans = Vec::new();
            self.rule_splitter.split(text, &mut spans);
            out.extend(spans.into_iter().map(|s| (s, 1.0)));
        }
        out
    }

    fn token_spans(&self, text: &str) -> Vec<(Span, f64)> {
        let mut out = Vec::new();
        if let Some(me) = &self.tokenizer_model {
            me.tokenize(text, &mut out);
        } else {
            let mut spans = Vec::new();
            simple_spans(text, TokenizerOptions::default(), &mut spans);
            out.extend(spans.into_iter().map(|s| (s, 1.0)));
        }
        out
    }

    pub fn detect_sentences(&self, text: &str) -> Result<AnalysisResultInner> {
        let offsets = OffsetTable::build(text);
        let sents = self.sentence_spans(text);
        let mut b = SpanPoolBuilder::with_capacity(sents.len(), text.len());
        for (idx, (span, prob)) in sents.into_iter().enumerate() {
            let covered = &text[span.start as usize..span.end as usize];
            b.push_text_span(text, &offsets, (span.start, span.end), (idx as u32, idx as u32 + 1), covered, prob);
        }
        Ok(b.finish(Vec::new()))
    }

    pub fn tokenize(&self, text: &str) -> Result<AnalysisResultInner> {
        let offsets = OffsetTable::build(text);
        let toks = self.token_spans(text);
        let mut b = SpanPoolBuilder::with_capacity(toks.len(), text.len());
        for (idx, (span, prob)) in toks.into_iter().enumerate() {
            let covered = &text[span.start as usize..span.end as usize];
            b.push_text_span(text, &offsets, (span.start, span.end), (idx as u32, idx as u32 + 1), covered, prob);
        }
        Ok(b.finish(Vec::new()))
    }

    pub fn pos_tag(&self, tokens: &[&str]) -> Result<AnalysisResultInner> {
        let tagged = self.pos_engine.tag_tokens(tokens)?;
        let mut b = SpanPoolBuilder::with_capacity(tagged.len(), tagged.len() * 6);
        for (idx, (tag, prob)) in tagged.into_iter().enumerate() {
            b.push_token_span(idx as u32, idx as u32 + 1, &tag, prob);
        }
        Ok(b.finish(Vec::new()))
    }

    pub fn lemmatize(&self, tokens: &[&str], tags: &[&str]) -> Result<AnalysisResultInner> {
        if tokens.len() != tags.len() {
            return Err(Error::new(INVALID_ARGUMENT, "tokens and tags must have the same count"));
        }
        let pairs: Vec<TokenTag<'_>> =
            tokens.iter().zip(tags.iter()).map(|(&token, &tag)| TokenTag { token, tag }).collect();
        let mut model_lemmas = Strs::default();
        let mut model_probs = Vec::with_capacity(tokens.len());
        let has_model = if tokens.is_empty() {
            false
        } else {
            self.maxent_lemmatizer
                .lemmatize(&pairs, &mut model_lemmas, &mut model_probs)
                .map_err(|e| Error::new(BUNDLE_INVALID, format!("lemmatize: {e}")))?
        };
        let mut b = SpanPoolBuilder::with_capacity(tokens.len(), tokens.len() * 8);
        for (idx, (&word, &tag)) in tokens.iter().zip(tags.iter()).enumerate() {
            if let Some(dict) = &self.dict_lemmatizer {
                let d_lemma = dict.lemma(word, tag);
                if d_lemma != NOT_FOUND {
                    b.push_token_span(idx as u32, idx as u32 + 1, d_lemma, 1.0);
                    continue;
                }
            }
            if has_model && idx < model_lemmas.len() {
                b.push_token_span(idx as u32, idx as u32 + 1, model_lemmas.get(idx), model_probs[idx]);
            } else {
                b.push_token_span(idx as u32, idx as u32 + 1, &word.to_ascii_lowercase(), 1.0);
            }
        }
        Ok(b.finish(Vec::new()))
    }

    fn collect_token_entities(&self, tokens: &[&str]) -> Result<Vec<(u32, u32, String, f64)>> {
        let mut entities = Vec::new();
        let mut names = Vec::new();
        for finder in [&self.person_dict as &dyn TokenNameFinder, &self.place_dict, &self.regex_finder] {
            names.clear();
            finder.find(tokens, &mut names).map_err(|e| Error::new(BUNDLE_INVALID, format!("namefind: {e}")))?;
            for n in &names {
                let kind = finder.kinds().label(n.kind).to_owned();
                if !entities.iter().any(|&(s, e, _, _)| s < n.end && n.start < e) {
                    entities.push((n.start, n.end, kind, 1.0));
                }
            }
        }
        for (start, end, kind, prob) in self.ner_engine.find(tokens)? {
            if !entities.iter().any(|&(s, e, _, _)| s < end && start < e) {
                entities.push((start, end, kind, prob));
            }
        }
        entities.sort_by_key(|&(s, e, _, _)| (s, e));
        Ok(entities)
    }

    pub fn find_tokens(&self, tokens: &[&str]) -> Result<AnalysisResultInner> {
        let entities = self.collect_token_entities(tokens)?;
        let mut b = SpanPoolBuilder::with_capacity(entities.len(), entities.len() * 12);
        for (start, end, kind, prob) in entities {
            b.push_token_span(start, end, &kind, prob);
        }
        Ok(b.finish(Vec::new()))
    }

    pub fn find_text(&self, text: &str) -> Result<AnalysisResultInner> {
        let offsets = OffsetTable::build(text);
        let tok_spans = self.token_spans(text);
        let tok_strs: Vec<&str> = tok_spans.iter().map(|&(s, _)| &text[s.start as usize..s.end as usize]).collect();
        let mut found = Vec::new();
        for (ts, te, kind, prob) in self.collect_token_entities(&tok_strs)? {
            if (ts as usize) < tok_spans.len() && (te as usize) <= tok_spans.len() && ts < te {
                let bs = tok_spans[ts as usize].0.start;
                let be = tok_spans[te as usize - 1].0.end;
                found.push((bs, be, ts, te, kind, prob));
            }
        }
        let pii = self.pii_detector.layer(text, ViewId::SOURCE);
        for m in pii.mentions() {
            let bs = m.span.start;
            let be = m.span.end;
            if !found.iter().any(|&(s, e, _, _, _, _)| s < be && bs < e) {
                let ts = tok_spans.partition_point(|&(s, _)| s.end <= bs) as u32;
                let te = tok_spans.partition_point(|&(s, _)| s.start < be).max(ts as usize + 1) as u32;
                found.push((bs, be, ts, te, m.kind.name().to_owned(), 1.0));
            }
        }
        found.sort_by_key(|&(bs, be, _, _, _, _)| (bs, be));
        let mut b = SpanPoolBuilder::with_capacity(found.len(), found.len() * 12);
        for (bs, be, ts, te, kind, prob) in found {
            b.push_text_span(text, &offsets, (bs, be), (ts, te), &kind, prob);
        }
        Ok(b.finish(Vec::new()))
    }

    pub fn analyze_graph(&self, text: &str) -> Result<AnalysisResultInner> {
        let mut doc = Document::new(text);
        let mut sent_spans = Spans::new(ViewId::SOURCE);
        for (s, _) in self.sentence_spans(text) {
            sent_spans.push(s);
        }
        doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("turbo:sentdetect"))
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add sentences: {e}")))?;

        let tok_vec = self.token_spans(text);
        let mut tok_spans = Spans::new(ViewId::SOURCE);
        let mut tok_strs = Vec::with_capacity(tok_vec.len());
        for (s, _) in &tok_vec {
            tok_spans.push(*s);
            tok_strs.push(&text[s.start as usize..s.end as usize]);
        }
        doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("turbo:tokenize"))
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add tokens: {e}")))?;

        let tagged = self.pos_engine.tag_tokens(&tok_strs)?;
        let tag_labels: Vec<&str> = tagged.iter().map(|(t, _)| t.as_str()).collect();
        let mut unique_tags: Vec<&str> = tag_labels.clone();
        unique_tags.sort_unstable();
        unique_tags.dedup();
        let tagset = TagSet::new("ud", &unique_tags);
        let tags: Vec<Tag> = tag_labels.iter().map(|&t| tagset.tag(t).unwrap_or(Tag(0))).collect();
        let probs: Vec<f64> = tagged.iter().map(|&(_, p)| p).collect();
        doc.add(&Key::<PosTags>::DEFAULT, PosTags { tagset, tags, prob: Some(probs) }, Provenance::by("turbo:postag"))
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add pos: {e}")))?;

        let lemma_res = self.lemmatize(&tok_strs, &tag_labels)?;
        let mut lemma_strs = Strs::default();
        for sp in &lemma_res.spans {
            let s = &lemma_res.pool[sp.label_offset as usize..(sp.label_offset + sp.label_len) as usize];
            lemma_strs.push(std::str::from_utf8(s).unwrap_or(""));
        }
        doc.add(&Key::<Lemmas>::DEFAULT, Lemmas { lemma: lemma_strs }, Provenance::by("turbo:lemmatizer"))
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add lemmas: {e}")))?;

        let ent_res = self.collect_token_entities(&tok_strs)?;
        let mut ent_kinds: Vec<&str> = ent_res.iter().map(|(_, _, k, _)| k.as_str()).collect();
        ent_kinds.sort_unstable();
        ent_kinds.dedup();
        let kinds = TagSet::new("entities", &ent_kinds);
        let mut ent_spans = Spans::new(ViewId::SOURCE);
        let mut ent_tokens = Vec::new();
        let mut ent_kind = Vec::new();
        let mut ent_score = Vec::new();
        for (ts, te, k, p) in ent_res {
            if (ts as usize) < tok_vec.len() && (te as usize) <= tok_vec.len() && ts < te {
                ent_spans.push(Span::new(tok_vec[ts as usize].0.start, tok_vec[te as usize - 1].0.end));
                ent_tokens.push(IxRange::new(ts, te));
                ent_kind.push(kinds.tag(&k).unwrap_or(Tag(0)));
                ent_score.push(p);
            }
        }
        doc.add(
            &Key::<Entities>::DEFAULT,
            Entities { spans: ent_spans, tokens: ent_tokens, kind: ent_kind, kinds, score: ent_score },
            Provenance::by("turbo:namefind"),
        )
        .map_err(|e| Error::new(BUNDLE_INVALID, format!("add entities: {e}")))?;

        let pii = self.pii_detector.layer(text, ViewId::SOURCE);
        doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("turbo:pii"))
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add pii: {e}")))?;

        let graph = export(&doc, &self.codecs).map_err(|e| Error::new(BUNDLE_INVALID, format!("export graph: {e}")))?;
        let wire = graph.to_bytes().map_err(|e| Error::new(BUNDLE_INVALID, format!("encode graph: {e}")))?;
        self.inspect_graph(text, &wire).map(|mut r| {
            r.graph = wire;
            r
        })
    }

    pub fn inspect_graph(&self, text: &str, graph_bytes: &[u8]) -> Result<AnalysisResultInner> {
        let graph = DocumentGraph::from_bytes(graph_bytes)
            .map_err(|e| Error::new(INVALID_ARGUMENT, format!("invalid DocumentGraph wire bytes: {e}")))?;
        let doc = import(&graph, &self.codecs)
            .map_err(|e| Error::new(INVALID_ARGUMENT, format!("DocumentGraph import failed: {e}")))?;
        let offsets = OffsetTable::build(text);
        let mut b = SpanPoolBuilder::default();
        if let Some(sents) = doc.get::<Sentences>() {
            for idx in 0..sents.spans.len() {
                let s = sents.spans.get(Ix::new(idx as u32));
                b.push_text_span(text, &offsets, (s.start, s.end), (idx as u32, idx as u32 + 1), "sentence", 1.0);
            }
        }
        if let Some(toks) = doc.get::<Tokens>() {
            let pos = doc.get::<PosTags>();
            for idx in 0..toks.spans.len() {
                let s = toks.spans.get(Ix::new(idx as u32));
                let (label, prob) = match pos {
                    Some(p) if idx < p.tags.len() => {
                        (p.tagset.label(p.tags[idx]), p.prob.as_ref().map_or(1.0, |pr| pr[idx]))
                    }
                    _ => ("token", 1.0),
                };
                b.push_text_span(text, &offsets, (s.start, s.end), (idx as u32, idx as u32 + 1), label, prob);
            }
        }
        if let Some(ents) = doc.get::<Entities>() {
            for idx in 0..ents.spans.len() {
                let s = ents.spans.get(Ix::new(idx as u32));
                let tr = ents.tokens[idx];
                let label = ents.kinds.label(ents.kind[idx]);
                b.push_text_span(
                    text,
                    &offsets,
                    (s.start, s.end),
                    (tr.start.get() as u32, tr.end.get() as u32),
                    label,
                    ents.score[idx],
                );
            }
        }
        if let Some(pii) = doc.get::<Pii>() {
            for m in pii.mentions() {
                b.push_text_span(text, &offsets, (m.span.start, m.span.end), (0, 0), m.kind.name(), 1.0);
            }
        }
        Ok(b.finish(graph_bytes.to_vec()))
    }
}
