//! Documents for the tests, built from words as spaCy's `Doc(vocab,
//! words=...)` builds them: the text is the words joined by one space, so
//! every match's bytes can be checked against the text.

#![allow(dead_code)]

use std::sync::Arc;

use turbo_analysis::{Ix, IxRange, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Entities, PosTags, Strs, Tokens};
use turbo_matcher::{Input, Match, Matcher, PhraseMatch, PhraseMatcher};

pub struct Doc {
    pub text: String,
    pub tokens: Tokens,
    pub pos: Option<PosTags>,
    pub entities: Option<Entities>,
    pub stop: Option<Vec<bool>>,
    pub norms: Option<Strs>,
    pub lemmas: Option<Strs>,
}

impl Doc {
    pub fn words(words: &[&str]) -> Doc {
        Doc::words_then(words, "")
    }

    /// The words, then `tail` after the last one.
    pub fn words_then(words: &[&str], tail: &str) -> Doc {
        let mut text = String::new();
        let mut spans = Spans::new(ViewId::SOURCE);
        for (i, w) in words.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            let start = text.len() as u32;
            text.push_str(w);
            spans.push(Span::new(start, text.len() as u32));
        }
        text.push_str(tail);
        Doc {
            text,
            tokens: Tokens { spans, sentence: None },
            pos: None,
            entities: None,
            stop: None,
            norms: None,
            lemmas: None,
        }
    }

    /// The words of `s` split at spaces.
    pub fn split(s: &str) -> Doc {
        Doc::words(&s.split(' ').collect::<Vec<_>>())
    }

    pub fn with_tags(mut self, tags: &[&str]) -> Doc {
        let mut labels: Vec<&str> = tags.to_vec();
        labels.sort_unstable();
        labels.dedup();
        let tagset = TagSet::new("tags", &labels);
        let tags = tags.iter().map(|t| tagset.tag(t).unwrap()).collect();
        self.pos = Some(PosTags { tagset, tags, prob: None });
        self
    }

    /// Entities as `(first token, past the last, kind)`.
    pub fn with_ents(mut self, ents: &[(u32, u32, &str)]) -> Doc {
        let mut kinds: Vec<&str> = ents.iter().map(|e| e.2).collect();
        kinds.sort_unstable();
        kinds.dedup();
        let kinds: Arc<TagSet> = TagSet::new("kinds", &kinds);
        let mut spans = Spans::new(ViewId::SOURCE);
        let mut tokens = Vec::new();
        let mut kind = Vec::new();
        for &(s, e, k) in ents {
            let r = IxRange::new(s, e);
            spans.push(Span::new(self.tokens.spans.get(Ix::new(s)).start, self.tokens.spans.get(Ix::new(e - 1)).end));
            tokens.push(r);
            kind.push(kinds.tag(k).unwrap());
        }
        let score = vec![1.0; tokens.len()];
        self.entities = Some(Entities { spans, tokens, kind, kinds, score });
        self
    }

    pub fn with_stop(mut self, stop: &[bool]) -> Doc {
        self.stop = Some(stop.to_vec());
        self
    }

    pub fn with_norms(mut self, norms: &[&str]) -> Doc {
        let mut s = Strs::default();
        norms.iter().for_each(|n| s.push(n));
        self.norms = Some(s);
        self
    }

    pub fn with_lemmas(mut self, lemmas: &[&str]) -> Doc {
        let mut s = Strs::default();
        lemmas.iter().for_each(|n| s.push(n));
        self.lemmas = Some(s);
        self
    }

    pub fn input(&self) -> Input<'_> {
        let mut input = Input::new(&self.text, &self.tokens).unwrap();
        if let Some(p) = &self.pos {
            input = input.with_pos(p).unwrap();
        }
        if let Some(e) = &self.entities {
            input = input.with_entities(e).unwrap();
        }
        if let Some(s) = &self.stop {
            input = input.with_stopwords(s).unwrap();
        }
        if let Some(n) = &self.norms {
            input = input.with_norms(n).unwrap();
        }
        if let Some(l) = &self.lemmas {
            input = input.with_lemmas(l).unwrap();
        }
        input
    }

    /// The text a token range covers, read from the source bytes, checked
    /// to be the range's words joined by a space.
    pub fn text_of(&self, r: IxRange<Tokens>) -> &str {
        let span = self.input().span(r);
        let text = &self.text[span.range()];
        let words: Vec<&str> = r.iter().map(|i| self.input().token(i.get())).collect();
        assert_eq!(text, words.join(" "), "a match's bytes are its tokens'");
        text
    }

    /// Every match as `(label, start, end)`.
    pub fn run(&self, m: &Matcher) -> Vec<(String, usize, usize)> {
        let found: Vec<Match> = m.find(&self.input()).unwrap();
        found
            .iter()
            .map(|x| {
                self.text_of(x.tokens);
                (m.label_name(x.label).to_string(), x.tokens.start.get(), x.tokens.end.get())
            })
            .collect()
    }

    /// Every match's text.
    pub fn texts(&self, m: &Matcher) -> Vec<String> {
        m.find(&self.input()).unwrap().iter().map(|x| self.text_of(x.tokens).to_string()).collect()
    }

    /// Every phrase match as `(label, start, end)`.
    pub fn run_phrases(&self, m: &PhraseMatcher) -> Vec<(String, usize, usize)> {
        let found: Vec<PhraseMatch> = m.find(&self.input()).unwrap();
        found
            .iter()
            .map(|x| {
                self.text_of(x.tokens);
                (m.label_name(x.label).to_string(), x.tokens.start.get(), x.tokens.end.get())
            })
            .collect()
    }
}

/// A tag, for building layers by hand.
pub fn tag(t: u16) -> Tag {
    Tag(t)
}
