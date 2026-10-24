//! Built-in deterministic DAG step implementations for `AnalyzeDocumentProgressive`.

use std::time::Duration;

use turbo_analysis::{Annotator, Error, Ix, IxRange, Span, Spans, TagSet, View};
use turbo_analysis_layers::{
    Boundaries, Chunks, Cut, Language, Lemmas, PosTags, Sentences, Stems, Strs, Terms, Tokens, Words,
};

pub(super) struct LanguageStep {
    pub(super) sleep_ms: u64,
}
impl Annotator for LanguageStep {
    const ID: &'static str = "turbo:language";
    type Reads = (View,);
    type Writes = (Language,);

    fn annotate(&self, (view,): (&View,)) -> Result<(Language,), Error> {
        if self.sleep_ms > 0 {
            std::thread::sleep(Duration::from_millis(self.sleep_ms));
        }
        let mut code = Strs::default();
        let text = view.text();
        let ascii_alpha = text.bytes().filter(|b| b.is_ascii_alphabetic()).count();
        let lang = if ascii_alpha > 0 || text.is_empty() { "eng" } else { "und" };
        code.push(lang);
        Ok((Language { code, score: vec![0.99] },))
    }
}

pub(super) struct SentencesStep;
impl Annotator for SentencesStep {
    const ID: &'static str = "turbo:sentences";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn annotate(&self, (view,): (&View,)) -> Result<(Sentences,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut start = 0usize;
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if matches!(bytes[i], b'.' | b'!' | b'?' | b'\n') {
                let end = i + 1;
                while start < end && bytes[start].is_ascii_whitespace() {
                    start += 1;
                }
                if start < end {
                    spans.push(Span::new(start as u32, end as u32));
                }
                start = end;
            }
            i += 1;
        }
        while start < bytes.len() && bytes[start].is_ascii_whitespace() {
            start += 1;
        }
        if start < bytes.len() {
            spans.push(Span::new(start as u32, bytes.len() as u32));
        }
        Ok((Sentences { spans },))
    }
}

pub(super) struct TokensStep;
impl Annotator for TokensStep {
    const ID: &'static str = "turbo:tokens";
    type Reads = (View, Sentences);
    type Writes = (Tokens,);

    fn annotate(&self, (view, sents): (&View, &Sentences)) -> Result<(Tokens,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut sentence = Vec::new();
        for (s_ix, s_span) in sents.spans.iter() {
            let slice = &text[s_span.range()];
            let mut cur_start: Option<usize> = None;
            for (rel, ch) in slice.char_indices() {
                if ch.is_whitespace() {
                    if let Some(st) = cur_start.take() {
                        spans.push(Span::new(s_span.start + st as u32, s_span.start + rel as u32));
                        sentence.push(s_ix);
                    }
                } else if ch.is_ascii_punctuation() && ch != '\'' && ch != '-' {
                    if let Some(st) = cur_start.take() {
                        spans.push(Span::new(s_span.start + st as u32, s_span.start + rel as u32));
                        sentence.push(s_ix);
                    }
                    let next = rel + ch.len_utf8();
                    spans.push(Span::new(s_span.start + rel as u32, s_span.start + next as u32));
                    sentence.push(s_ix);
                } else if cur_start.is_none() {
                    cur_start = Some(rel);
                }
            }
            if let Some(st) = cur_start.take() {
                spans.push(Span::new(s_span.start + st as u32, s_span.end));
                sentence.push(s_ix);
            }
        }
        Ok((Tokens { spans, sentence: Some(sentence) },))
    }
}

pub(super) struct WordsStep;
impl Annotator for WordsStep {
    const ID: &'static str = "turbo:words";
    type Reads = (Tokens,);
    type Writes = (Words,);

    fn annotate(&self, (tokens,): (&Tokens,)) -> Result<(Words,), Error> {
        Ok((Words::from_tokens(tokens),))
    }
}

pub(super) struct TermsStep;
impl Annotator for TermsStep {
    const ID: &'static str = "turbo:terms";
    type Reads = (View, Tokens);
    type Writes = (Terms,);

    fn annotate(&self, (view, tokens): (&View, &Tokens)) -> Result<(Terms,), Error> {
        let text = view.text();
        let mut surface = Strs::default();
        let mut folded = Strs::default();
        let mut stem = Strs::default();
        for (_, span) in tokens.spans.iter() {
            let s = &text[span.range()];
            let low = s.to_lowercase();
            let st = low
                .strip_suffix("ing")
                .or_else(|| low.strip_suffix("ed"))
                .or_else(|| low.strip_suffix('s'))
                .unwrap_or(&low);
            let st = if st.is_empty() { low.as_str() } else { st };
            surface.push(s);
            folded.push(&low);
            stem.push(st);
        }
        Ok((Terms { surface, folded, stem: Some(stem), lemma: None },))
    }
}

pub(super) struct StemsStep;
impl Annotator for StemsStep {
    const ID: &'static str = "turbo:stems";
    type Reads = (Tokens, Terms);
    type Writes = (Stems,);

    fn annotate(&self, (_tokens, terms): (&Tokens, &Terms)) -> Result<(Stems,), Error> {
        let mut stem = Strs::default();
        for slot in terms.iter() {
            stem.push(slot.stem.unwrap_or(slot.folded));
        }
        Ok((Stems { stem },))
    }
}

pub(super) struct PosStep {
    pub(super) force_fail: bool,
}
impl Annotator for PosStep {
    const ID: &'static str = "turbo:pos";
    type Reads = (View, Tokens);
    type Writes = (PosTags,);

    fn annotate(&self, (view, tokens): (&View, &Tokens)) -> Result<(PosTags,), Error> {
        let text = view.text();
        if self.force_fail || text.contains("[FAIL_POS]") {
            return Err(Error::Annotator { annotator: Self::ID, detail: "POS tagger branch failure".into() });
        }
        let tagset = TagSet::new("upos", &["NOUN", "VERB", "PUNCT", "PROPN"]);
        let noun = tagset.tag("NOUN").unwrap();
        let punct = tagset.tag("PUNCT").unwrap();
        let propn = tagset.tag("PROPN").unwrap();
        let mut tags = Vec::with_capacity(tokens.spans.len());
        let mut probs = Vec::with_capacity(tokens.spans.len());
        for (_, span) in tokens.spans.iter() {
            let s = &text[span.range()];
            let tag = if s.chars().all(|c| c.is_ascii_punctuation()) {
                punct
            } else if s.chars().next().is_some_and(char::is_uppercase) {
                propn
            } else {
                noun
            };
            tags.push(tag);
            probs.push(0.95);
        }
        Ok((PosTags { tagset, tags, prob: Some(probs) },))
    }
}

pub(super) struct LemmasStep;
impl Annotator for LemmasStep {
    const ID: &'static str = "turbo:lemmas";
    type Reads = (View, Tokens, PosTags);
    type Writes = (Lemmas,);

    fn annotate(&self, (view, tokens, _pos): (&View, &Tokens, &PosTags)) -> Result<(Lemmas,), Error> {
        let text = view.text();
        let mut lemma = Strs::default();
        for (_, span) in tokens.spans.iter() {
            lemma.push(&text[span.range()].to_lowercase());
        }
        Ok((Lemmas { lemma },))
    }
}

pub(super) struct BoundariesStep;
impl Annotator for BoundariesStep {
    const ID: &'static str = "turbo:boundaries";
    type Reads = (Sentences,);
    type Writes = (Boundaries,);

    fn annotate(&self, (sents,): (&Sentences,)) -> Result<(Boundaries,), Error> {
        let n = sents.spans.len();
        Ok((Boundaries { cost: vec![Boundaries::UNIT; n], reason: vec![Cut::Sentence; n] },))
    }
}

pub(super) struct ChunksStep;
impl Annotator for ChunksStep {
    const ID: &'static str = "turbo:chunks";
    type Reads = (Sentences, Boundaries);
    type Writes = (Chunks,);

    fn annotate(&self, (sents, _bounds): (&Sentences, &Boundaries)) -> Result<(Chunks,), Error> {
        let mut spans = Spans::new(sents.spans.view());
        let n = sents.spans.len();
        if n > 0 {
            let first = sents.spans.get(Ix::new(0));
            let last = sents.spans.get(Ix::new(n as u32 - 1));
            spans.push(Span::new(first.start, last.end));
            Ok((Chunks {
                spans,
                level: vec![0],
                parent: vec![None],
                sentences: vec![IxRange::new(0, n as u32)],
                cut: vec![Cut::DocumentStart],
                cost: vec![0],
                model_tokens: vec![n as u32 * 8],
                over_budget: vec![false],
                prompt_overhead: 0,
            },))
        } else {
            Ok((Chunks {
                spans,
                level: Vec::new(),
                parent: Vec::new(),
                sentences: Vec::new(),
                cut: Vec::new(),
                cost: Vec::new(),
                model_tokens: Vec::new(),
                over_budget: Vec::new(),
                prompt_overhead: 0,
            },))
        }
    }
}
