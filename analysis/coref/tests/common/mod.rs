//! A test fixture: whitespace-tokenized sentences of `form/TAG` tokens,
//! with entities, chunks, parse phrases and speakers marked by text.

#![allow(dead_code)]

use std::sync::Arc;

use turbo_analysis::{Document, Groups, Ix, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Entities, PhraseChunks, Phrases, PosTags, Sentences, Tokens};
use turbo_coref::{CorefAnnotator, CorefInput, Mentions, Speakers};

#[derive(Default, Clone)]
pub struct Fixture {
    pub text: String,
    pub sentences: Vec<Span>,
    pub tokens: Vec<Span>,
    pub tags: Vec<String>,
    pub entities: Vec<(Span, String)>,
    pub chunks: Vec<(Span, String)>,
    pub phrases: Vec<(Span, String, Span)>,
    pub speakers: Vec<(Span, String)>,
    pub with_chunks: bool,
    pub with_phrases: bool,
    pub with_speakers: bool,
}

pub struct Built {
    pub text: String,
    pub sentences: Sentences,
    pub tokens: Tokens,
    pub tags: PosTags,
    pub entities: Entities,
    pub chunks: Option<PhraseChunks>,
    pub phrases: Option<Phrases>,
    pub speakers: Option<Speakers>,
}

fn tagset(name: &str, labels: &[String]) -> (Arc<TagSet>, Vec<Tag>) {
    let mut unique: Vec<&str> = Vec::new();
    for l in labels {
        if !unique.contains(&l.as_str()) {
            unique.push(l);
        }
    }
    let set = TagSet::new(name, &unique);
    let tags = labels.iter().map(|l| set.tag(l).unwrap()).collect();
    (set, tags)
}

fn spans<L>(items: impl IntoIterator<Item = Span>) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for span in items {
        s.push(span);
    }
    s
}

impl Fixture {
    pub fn new() -> Fixture {
        Fixture::default()
    }

    /// Appends one sentence of `form/TAG` tokens.
    pub fn sentence(mut self, tagged: &[&str]) -> Fixture {
        if !self.text.is_empty() {
            self.text.push(' ');
        }
        let start = self.text.len() as u32;
        for (i, t) in tagged.iter().enumerate() {
            if i > 0 {
                self.text.push(' ');
            }
            let slash = t.rfind('/').unwrap();
            let form = &t[..slash];
            let s = self.text.len() as u32;
            self.text.push_str(form);
            self.tokens.push(Span::new(s, self.text.len() as u32));
            self.tags.push(t[slash + 1..].to_owned());
        }
        self.sentences.push(Span::new(start, self.text.len() as u32));
        self
    }

    fn find(&self, phrase: &str, after: Option<Span>) -> Span {
        let from = after.map_or(0, |s| s.end as usize);
        let start = self.text[from..].find(phrase).unwrap_or_else(|| panic!("{phrase} is not in {}", self.text)) + from;
        Span::new(start as u32, (start + phrase.len()) as u32)
    }

    /// Marks the first occurrence of a phrase after the last entity.
    pub fn entity(mut self, phrase: &str, ty: &str) -> Fixture {
        let span = self.find(phrase, self.entities.last().map(|e| e.0));
        self.entities.push((span, ty.to_owned()));
        self
    }

    /// Marks the first occurrence of a phrase after the last chunk.
    pub fn chunk(mut self, phrase: &str, ty: &str) -> Fixture {
        let span = self.find(phrase, self.chunks.last().map(|e| e.0));
        self.chunks.push((span, ty.to_owned()));
        self.with_chunks = true;
        self
    }

    /// Adds a parse phrase over the text found from byte `start`, headed
    /// by the first occurrence of `head` inside it.
    pub fn phrase(mut self, label: &str, start: usize, text: &str, head: &str) -> Fixture {
        let from = self.text[start..].find(text).unwrap() + start;
        let head_start = self.text[from..].find(head).unwrap() + from;
        let span = Span::new(from as u32, (from + text.len()) as u32);
        let head = Span::new(head_start as u32, (head_start + head.len()) as u32);
        self.phrases.push((span, label.to_owned(), head));
        self.with_phrases = true;
        self
    }

    pub fn speaker(mut self, sentence: usize, label: &str) -> Fixture {
        self.speakers.push((self.sentences[sentence], label.to_owned()));
        self.with_speakers = true;
        self
    }

    fn token_range(&self, span: Span) -> IxRange<Tokens> {
        let first = self.tokens.iter().position(|t| t.end > span.start).unwrap_or(self.tokens.len());
        let end = self.tokens.iter().position(|t| t.start >= span.end).unwrap_or(self.tokens.len());
        IxRange::new(first as u32, end.max(first) as u32)
    }

    pub fn build(&self) -> Built {
        let sentence_of = |t: &Span| Ix::new(self.sentences.iter().position(|s| s.contains(*t)).unwrap_or(0) as u32);
        let (penn, tags) = tagset("penn", &self.tags);
        let (kinds, kind) = tagset_of("entity", &self.entities);
        let entities = Entities {
            spans: spans(self.entities.iter().map(|e| e.0)),
            tokens: self.entities.iter().map(|e| self.token_range(e.0)).collect(),
            kind,
            kinds,
            score: vec![1.0; self.entities.len()],
        };
        let chunks = self.with_chunks.then(|| {
            let (kinds, kind) = tagset_of("chunk", &self.chunks);
            PhraseChunks {
                spans: spans(self.chunks.iter().map(|c| c.0)),
                tokens: self.chunks.iter().map(|c| self.token_range(c.0)).collect(),
                kind,
                kinds,
            }
        });
        let phrases = self.with_phrases.then(|| {
            let labels: Vec<String> = self.phrases.iter().map(|p| p.1.clone()).collect();
            let (kinds, kind) = tagset("phrase", &labels);
            Phrases {
                spans: spans(self.phrases.iter().map(|p| p.0)),
                tokens: self.phrases.iter().map(|p| self.token_range(p.0)).collect(),
                kind,
                kinds,
                head: self.phrases.iter().map(|p| Ix::new(self.token_range(p.2).start.get() as u32)).collect(),
                parent: vec![None; self.phrases.len()],
            }
        });
        let speakers = self.with_speakers.then(|| {
            let (speakers, speaker) = tagset_of("speaker", &self.speakers);
            Speakers { spans: spans(self.speakers.iter().map(|s| s.0)), speaker, speakers }
        });
        Built {
            text: self.text.clone(),
            sentences: Sentences { spans: spans(self.sentences.iter().copied()) },
            tokens: Tokens {
                spans: spans(self.tokens.iter().copied()),
                sentence: Some(self.tokens.iter().map(sentence_of).collect()),
            },
            tags: PosTags { tagset: penn, tags, prob: None },
            entities,
            chunks,
            phrases,
            speakers,
        }
    }
}

fn tagset_of(name: &str, items: &[(Span, String)]) -> (Arc<TagSet>, Vec<Tag>) {
    let labels: Vec<String> = items.iter().map(|i| i.1.clone()).collect();
    tagset(name, &labels)
}

impl Built {
    pub fn input(&self) -> CorefInput<'_> {
        CorefInput {
            text: &self.text,
            view: ViewId::SOURCE,
            sentences: &self.sentences,
            tokens: &self.tokens,
            tags: &self.tags,
            entities: &self.entities,
            chunks: self.chunks.as_ref(),
            phrases: self.phrases.as_ref(),
            speakers: self.speakers.as_ref(),
        }
    }

    /// The layers as a document, each given by a `test` step.
    pub fn document(self) -> Document {
        let mut doc = Document::new(self.text.as_str());
        let by = Provenance::by("test");
        doc.add(&Key::DEFAULT, self.sentences, by.clone()).unwrap();
        doc.add(&Key::DEFAULT, self.tokens, by.clone()).unwrap();
        doc.add(&Key::DEFAULT, self.tags, by.clone()).unwrap();
        doc.add(&Key::DEFAULT, self.entities, by.clone()).unwrap();
        if let Some(c) = self.chunks {
            doc.add(&Key::DEFAULT, c, by.clone()).unwrap();
        }
        if let Some(p) = self.phrases {
            doc.add(&Key::DEFAULT, p, by.clone()).unwrap();
        }
        if let Some(s) = self.speakers {
            doc.add(&Key::DEFAULT, s, by).unwrap();
        }
        doc
    }
}

/// The surface forms of every mention, in layer order.
pub fn forms(text: &str, mentions: &Mentions) -> Vec<String> {
    mentions.spans.iter().map(|(_, s)| text[s.range()].to_owned()).collect()
}

/// Every chain of two or more mentions as its surface forms.
pub fn chain_forms(text: &str, mentions: &Mentions, chains: &Groups<Mentions>) -> Vec<Vec<String>> {
    chains
        .iter()
        .filter(|(_, m)| m.len() > 1)
        .map(|(_, m)| m.iter().map(|&i| text[mentions.spans.get(i).range()].to_owned()).collect())
        .collect()
}

/// Resolves with the default rule-based annotator.
pub fn chains(fixture: &Fixture) -> Vec<Vec<String>> {
    chains_with(fixture, &CorefAnnotator::default())
}

pub fn chains_with(fixture: &Fixture, annotator: &CorefAnnotator) -> Vec<Vec<String>> {
    let built = fixture.build();
    let (mentions, groups, _) = annotator.resolve(&built.input()).unwrap();
    chain_forms(&built.text, &mentions, &groups)
}

pub fn v(chains: &[&[&str]]) -> Vec<Vec<String>> {
    chains.iter().map(|c| c.iter().map(|s| s.to_string()).collect()).collect()
}

/// The chain index of every mention, chains in first-mention order.
pub fn chain_ids(fixture: &Fixture, annotator: &CorefAnnotator) -> Vec<usize> {
    let built = fixture.build();
    let (mentions, groups, _) = annotator.resolve(&built.input()).unwrap();
    (0..mentions.spans.len())
        .map(|m| groups.iter().position(|(_, g)| g.contains(&Ix::new(m as u32))).unwrap())
        .collect()
}

/// The refusal message for a fixture, which must be refused.
pub fn refusal(built: &Built) -> String {
    CorefAnnotator::default().resolve(&built.input()).expect_err("the input must be refused")
}
