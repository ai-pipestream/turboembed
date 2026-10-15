//! Documents for the tests: words joined by single spaces, as spaCy's
//! `Doc(vocab, words=...)` builds them, with the token layers a test sets.

#![allow(dead_code)]

use turbo_analysis::{IxRange, Span, Spans, ViewId};
use turbo_analysis_layers::{Entities, PosTags, Strs, Tag, TagSet};
use turbo_matcher::{Match, Matcher, PhraseMatcher, Sequence};
use turbo_stopword_layers::{StopwordLanguage, Stopwords};

pub struct Doc {
    pub text: String,
    pub start: Vec<u32>,
    pub end: Vec<u32>,
    pub pos: Option<PosTags>,
    pub entities: Option<Entities>,
    pub stop: Option<Stopwords>,
    pub columns: Vec<(&'static str, Strs)>,
}

pub fn doc(words: &[&str]) -> Doc {
    let mut text = String::new();
    let (mut start, mut end) = (Vec::new(), Vec::new());
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            text.push(' ');
        }
        start.push(text.len() as u32);
        text.push_str(w);
        end.push(text.len() as u32);
    }
    Doc { text, start, end, pos: None, entities: None, stop: None, columns: Vec::new() }
}

/// Splits on single spaces, as the tests' `text.split()`.
pub fn split(text: &str) -> Doc {
    doc(&text.split(' ').collect::<Vec<_>>())
}

impl Doc {
    pub fn tags(mut self, tags: &[&str]) -> Doc {
        let mut labels: Vec<&str> = tags.to_vec();
        labels.sort_unstable();
        labels.dedup();
        let set = TagSet::new("tags", &labels);
        let tags = tags.iter().map(|t| set.tag(t).unwrap()).collect();
        self.pos = Some(PosTags { tagset: set, tags, prob: None });
        self
    }

    /// Entities as `(start, end, kind)` token ranges.
    pub fn ents(mut self, ents: &[(u32, u32, &str)]) -> Doc {
        let mut kinds: Vec<&str> = ents.iter().map(|e| e.2).collect();
        kinds.sort_unstable();
        kinds.dedup();
        let set = TagSet::new("kinds", &kinds);
        let mut spans = Spans::new(ViewId::SOURCE);
        let mut tokens = Vec::new();
        let mut kind = Vec::new();
        for &(s, e, k) in ents {
            spans.push(Span::new(self.start[s as usize], self.end[e as usize - 1]));
            tokens.push(IxRange::new(s, e));
            kind.push(set.tag(k).unwrap());
        }
        let score = vec![1.0; kind.len()];
        self.entities = Some(Entities { spans, tokens, kind, kinds: set, score });
        self
    }

    pub fn stop(mut self, stop: &[bool]) -> Doc {
        self.stop = Some(Stopwords { language: StopwordLanguage::English, stop: stop.to_vec() });
        self
    }

    pub fn column(mut self, name: &'static str, values: &[&str]) -> Doc {
        let mut s = Strs::default();
        for v in values {
            s.push(v);
        }
        self.columns.push((name, s));
        self
    }

    pub fn seq(&self) -> Sequence<'_> {
        let mut seq = Sequence::new(&self.text, &self.start, &self.end).unwrap();
        if let Some(p) = &self.pos {
            seq = seq.with_pos(p).unwrap();
        }
        if let Some(e) = &self.entities {
            seq = seq.with_entities(e).unwrap();
        }
        if let Some(s) = &self.stop {
            seq = seq.with_stopwords(s).unwrap();
        }
        for (name, values) in &self.columns {
            seq = seq.with_column(name, values).unwrap();
        }
        seq
    }

    /// The text of tokens `[start, end)`, joined by spaces.
    pub fn words(&self, start: u32, end: u32) -> &str {
        &self.text[self.start[start as usize] as usize..self.end[end as usize - 1] as usize]
    }
}

pub fn find(m: &Matcher, d: &Doc) -> Vec<Match> {
    m.find(&d.seq()).unwrap()
}

/// The matches as `(label, start, end)`.
pub fn found(m: &Matcher, d: &Doc) -> Vec<(String, u32, u32)> {
    find(m, d).iter().map(|x| (m.labels().label(x.label).to_string(), x.start, x.end)).collect()
}

pub fn spans(m: &Matcher, d: &Doc) -> Vec<(u32, u32)> {
    find(m, d).iter().map(|x| (x.start, x.end)).collect()
}

pub fn phrase_found(m: &PhraseMatcher, d: &Doc) -> Vec<(String, u32, u32)> {
    m.find(&d.seq()).unwrap().iter().map(|x| (m.labels().label(x.label).to_string(), x.start, x.end)).collect()
}

pub fn tag(i: u16) -> Tag {
    Tag(i)
}
