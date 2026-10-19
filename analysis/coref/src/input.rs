//! The layers coreference reads, validated and flattened into the arrays
//! detection and the sieves index (`CorefAnnotator.resolver`).

use std::collections::HashMap;

use turbo_analysis::{Document, Ix, Span, View, ViewId};
use turbo_analysis_layers::{Entities, PhraseChunks, Phrases, PosTags, Sentences, Tokens};

use crate::layer::Speakers;

/// The layers one coreference run reads, all in one text view.
#[derive(Copy, Clone)]
pub struct CorefInput<'a> {
    /// The text of the view every layer's spans are in.
    pub text: &'a str,
    pub view: ViewId,
    pub sentences: &'a Sentences,
    pub tokens: &'a Tokens,
    pub tags: &'a PosTags,
    pub entities: &'a Entities,
    pub chunks: Option<&'a PhraseChunks>,
    pub phrases: Option<&'a Phrases>,
    pub speakers: Option<&'a Speakers>,
}

impl std::fmt::Debug for CorefInput<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CorefInput")
            .field("view", &self.view)
            .field("sentences", &self.sentences.spans.len())
            .field("tokens", &self.tokens.spans.len())
            .field("entities", &self.entities.spans.len())
            .field("chunks", &self.chunks.map(|c| c.spans.len()))
            .field("phrases", &self.phrases.map(|p| p.spans.len()))
            .field("speakers", &self.speakers.map(|s| s.spans.len()))
            .finish()
    }
}

impl<'a> CorefInput<'a> {
    /// The default instances of the layers in `doc`, or a message naming
    /// the first required layer it lacks.
    pub fn from_document(doc: &'a Document) -> Result<CorefInput<'a>, String> {
        fn need<L: turbo_analysis::Layer>(doc: &Document) -> Result<&L, String> {
            doc.get::<L>().ok_or_else(|| format!("document lacks the required layer {}", L::ID))
        }
        let view: &View = need(doc)?;
        Ok(CorefInput {
            text: view.text(),
            view: view.id(),
            sentences: need(doc)?,
            tokens: need(doc)?,
            tags: need(doc)?,
            entities: need(doc)?,
            chunks: doc.get(),
            phrases: doc.get(),
            speakers: doc.get(),
        })
    }
}

/// An entity with its lowercased type.
#[derive(Clone, Debug)]
pub(crate) struct EntityIn {
    pub span: Span,
    pub fragments: Option<Vec<Span>>,
    pub ty: String,
}

/// A chunk or speaker turn with its label.
#[derive(Clone, Debug)]
pub(crate) struct Labeled {
    pub span: Span,
    pub label: String,
}

/// A parse phrase with its label and head token span.
#[derive(Clone, Debug)]
pub(crate) struct PhraseIn {
    pub span: Span,
    pub label: String,
    pub head: Span,
}

/// The validated, flattened input.
#[derive(Debug)]
pub(crate) struct Prepared<'a> {
    pub tokens: Vec<Span>,
    /// The token texts as written.
    pub forms: Vec<&'a str>,
    /// The token texts lowercased as `String.toLowerCase(Locale.ROOT)`.
    pub lower: Vec<String>,
    pub tags: Vec<&'a str>,
    pub sentence_of_token: Vec<usize>,
    pub entities: Vec<EntityIn>,
    pub chunks: Option<Vec<Labeled>>,
    pub phrases: Option<Vec<PhraseIn>>,
    pub speakers: Option<Vec<Labeled>>,
}

pub(crate) fn lowercase(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    turbo_unicode::java::to_lowercase(text, &mut out);
    out
}

fn blank(text: &str) -> bool {
    text.chars().all(turbo_unicode::java::is_whitespace)
}

fn same_view(name: &str, view: ViewId, expected: ViewId) -> Result<(), String> {
    if view == expected {
        Ok(())
    } else {
        Err(format!("the {name} are in view {}, not in view {}", view.0, expected.0))
    }
}

impl<'a> CorefInput<'a> {
    /// Validates the layers as OpenNLP's `CorefAnnotator` does and
    /// flattens them; `None` for a document without tokens, whose chains
    /// are empty.
    pub(crate) fn prepare(&self) -> Result<Option<Prepared<'a>>, String> {
        let view = self.view;
        same_view("sentences", self.sentences.spans.view(), view)?;
        same_view("tokens", self.tokens.spans.view(), view)?;
        same_view("entities", self.entities.spans.view(), view)?;
        if let Some(chunks) = self.chunks {
            same_view("chunks", chunks.spans.view(), view)?;
        }
        if let Some(phrases) = self.phrases {
            same_view("phrases", phrases.spans.view(), view)?;
        }
        if let Some(speakers) = self.speakers {
            same_view("speakers", speakers.spans.view(), view)?;
        }
        let text = self.text;
        let sentences: Vec<Span> = self.sentences.spans.iter().map(|(_, s)| s).collect();
        let tokens: Vec<Span> = self.tokens.spans.iter().map(|(_, s)| s).collect();
        if self.tags.tags.len() != tokens.len() {
            return Err("document needs aligned opennlp:tokens and opennlp:pos layers".into());
        }
        for s in 1..sentences.len() {
            if sentences[s - 1].end > sentences[s].start {
                return Err(format!("sentence spans overlap or are out of order at index {s}"));
            }
        }
        let mut entities = Vec::with_capacity(self.entities.spans.len());
        for (i, span) in self.entities.spans.iter() {
            let e = i.get();
            if span.is_empty() || blank(&text[span.range()]) {
                return Err(format!("entity at index {e} must cover non-blank text"));
            }
            let label = self.entities.kinds.label(self.entities.kind[e]);
            if blank(label) {
                return Err(format!("entity at index {e} must have a non-blank type"));
            }
            if !sentences.iter().any(|s| s.contains(span)) {
                return Err(format!("entity at index {e} falls outside the sentence layer"));
            }
            let fragments = self.entities.spans.is_discontinuous(i).then(|| self.entities.spans.fragments(i).collect());
            entities.push(EntityIn { span, fragments, ty: lowercase(label) });
        }
        let chunks = self.chunks.map(|c| {
            c.spans
                .iter()
                .map(|(i, span)| Labeled { span, label: c.kinds.label(c.kind[i.get()]).to_owned() })
                .collect::<Vec<_>>()
        });
        let phrases = self.phrases.map(|p| {
            p.spans
                .iter()
                .map(|(i, span)| PhraseIn {
                    span,
                    label: p.kinds.label(p.kind[i.get()]).to_owned(),
                    head: self.tokens.spans.get(p.head[i.get()]),
                })
                .collect::<Vec<_>>()
        });
        let speakers = self.speakers.map(|s| {
            s.spans
                .iter()
                .map(|(i, span)| Labeled { span, label: s.speakers.label(s.speaker[i.get()]).to_owned() })
                .collect::<Vec<_>>()
        });
        let mut starts = HashMap::with_capacity(tokens.len());
        let mut ends = HashMap::with_capacity(tokens.len());
        let mut sentence_of_token = Vec::with_capacity(tokens.len());
        if !tokens.is_empty() && sentences.is_empty() {
            return Err("document needs a non-empty opennlp:sentences layer".into());
        }
        let mut sentence = 0;
        for (t, &token) in tokens.iter().enumerate() {
            if t > 0 && tokens[t - 1].end > token.start {
                return Err(format!("token spans overlap or are out of order at index {t}"));
            }
            while sentence < sentences.len() - 1 && token.start >= sentences[sentence].end {
                sentence += 1;
            }
            if !sentences[sentence].contains(token) {
                return Err(format!("token at index {t} falls outside the sentence layer"));
            }
            sentence_of_token.push(sentence);
            starts.insert(token.start, t);
            ends.insert(token.end, t);
        }
        let aligned = |layer: &str, index: usize, span: Span| -> Result<(), String> {
            match (starts.get(&span.start), ends.get(&span.end)) {
                (Some(&first), Some(&last)) if first <= last => {
                    if sentence_of_token[first] != sentence_of_token[last] {
                        return Err(format!("{layer} at index {index} crosses a sentence boundary"));
                    }
                    Ok(())
                }
                _ => Err(format!("{layer} at index {index} must align with token boundaries")),
            }
        };
        if let Some(chunks) = &chunks {
            for (c, chunk) in chunks.iter().enumerate() {
                if blank(&chunk.label) {
                    return Err(format!("chunk at index {c} must have a non-blank type"));
                }
                aligned("chunk", c, chunk.span)?;
                if c > 0 && chunks[c - 1].span.end > chunk.span.start {
                    return Err(format!("chunk spans overlap or are out of order at index {c}"));
                }
            }
        }
        if let Some(phrases) = &phrases {
            for (p, phrase) in phrases.iter().enumerate() {
                aligned("phrase", p, phrase.span)?;
                let one = starts.get(&phrase.head.start).is_some_and(|s| ends.get(&phrase.head.end) == Some(s));
                if !one || !phrase.span.contains(phrase.head) {
                    return Err(format!("phrase at index {p} must name one of its tokens as its head"));
                }
            }
        }
        if let Some(speakers) = &speakers {
            for (s, speaker) in speakers.iter().enumerate() {
                if speaker.span.is_empty() || blank(&speaker.label) {
                    return Err(format!("speaker at index {s} must cover text and have a non-blank label"));
                }
                if s > 0 && speakers[s - 1].span.end > speaker.span.start {
                    return Err(format!("speaker spans overlap or are out of order at index {s}"));
                }
            }
        }
        if tokens.is_empty() {
            return Ok(None);
        }
        let forms: Vec<&'a str> = tokens.iter().map(|t| &text[t.range()]).collect();
        let lower = forms.iter().map(|f| lowercase(f)).collect();
        let tags = self.tags.tags.iter().map(|&t| self.tags.tagset.label(t)).collect();
        Ok(Some(Prepared { tokens, forms, lower, tags, sentence_of_token, entities, chunks, phrases, speakers }))
    }
}

/// The index `i` as a layer index.
pub(crate) fn ix<L>(i: usize) -> Ix<L> {
    Ix::new(i as u32)
}
