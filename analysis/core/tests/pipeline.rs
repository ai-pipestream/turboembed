//! A whole pipeline of toy annotators: the point is the API's mechanics
//! (types, offsets, ordering, batching, failures), not the linguistics.

#[path = "pipeline/aligned.rs"]
mod aligned;
#[path = "pipeline/assembly.rs"]
mod assembly;
#[path = "pipeline/layers.rs"]
mod layers;
#[path = "pipeline/run.rs"]
mod run;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo_analysis::*;
use turbo_analysis_layers::{Confidence, Language, Lemmas, PosTags, Sentences, Strs, Tag, TagSet, Tokens, Vectors};

/// Folds runs of whitespace to one space and expands the "fi" ligature,
/// writing the view "folded" with its alignment to the source.
struct Fold;

impl Annotator for Fold {
    const ID: &'static str = "test:fold";
    type Reads = (View,);
    type Writes = (View,);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View,), Error> {
        let text = src.text();
        let mut b = ViewBuilder::new(text);
        let mut chars = text.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            if c == '\u{FB01}' {
                b.keep(i);
                b.replace(i + c.len_utf8(), "fi");
            } else if c.is_whitespace() {
                let mut end = i + c.len_utf8();
                while let Some(&(j, d)) = chars.peek() {
                    if !d.is_whitespace() {
                        break;
                    }
                    end = j + d.len_utf8();
                    chars.next();
                }
                if &text[i..end] != " " {
                    b.keep(i);
                    b.replace(end, " ");
                }
            }
        }
        let (out, alignment) = b.finish();
        Ok((View::derived(src, out, alignment),))
    }
}

/// Splits after ". ".
struct Sentencer;

impl Annotator for Sentencer {
    const ID: &'static str = "test:sentences";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut start = 0;
        for (i, _) in text.match_indices(". ") {
            spans.push(Span::new(start as u32, i as u32 + 1));
            start = i + 2;
        }
        if start < text.len() {
            spans.push(Span::new(start as u32, text.len() as u32));
        }
        Ok((Sentences { spans },))
    }
}

/// Whitespace tokens, per sentence.
struct Tokenizer;

impl Annotator for Tokenizer {
    const ID: &'static str = "test:tokens";
    type Reads = (View, Sentences);
    type Writes = (Tokens,);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        if view.text().contains("FAIL") {
            return Err(Error::Annotator { annotator: Self::ID, detail: "asked to fail".into() });
        }
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut sentence = Vec::new();
        for (s, span) in sentences.spans.iter() {
            let mut at = span.start as usize;
            for word in text[span.range()].split(' ') {
                if !word.is_empty() {
                    spans.push(Span::new(at as u32, (at + word.len()) as u32));
                    sentence.push(s);
                }
                at += word.len() + 1;
            }
        }
        Ok((Tokens { spans, sentence: Some(sentence) },))
    }
}

/// Tags words ending in "s" as VBZ, the rest X.
struct Tagger {
    tagset: Arc<TagSet>,
}

impl Annotator for Tagger {
    const ID: &'static str = "test:tagger";
    type Reads = (View, Tokens);
    type Writes = (PosTags,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(PosTags,), Error> {
        let (vbz, x) = (self.tagset.tag("VBZ").unwrap(), self.tagset.tag("X").unwrap());
        let text = view.text();
        let tags: Vec<Tag> =
            tokens.spans.iter().map(|(_, s)| if text[s.range()].ends_with('s') { vbz } else { x }).collect();
        Ok((PosTags { tagset: self.tagset.clone(), tags, prob: None },))
    }
}

/// An embedder that takes the whole batch's sentences in one call, as a
/// device run would, and counts its calls.
struct Embedder {
    calls: Arc<AtomicUsize>,
}

impl Annotator for Embedder {
    const ID: &'static str = "test:embedder";
    type Reads = (View, Sentences);
    type Writes = (Vectors<Sentences>,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<(Vectors<Sentences>,), Error> {
        self.annotate_batch(vec![input]).pop().unwrap()
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<(Vectors<Sentences>,), Error>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        // Pack every sentence of every document into one run, then split.
        let packed: Vec<&str> =
            inputs.iter().flat_map(|(v, s)| s.spans.iter().map(|(_, sp)| &v.text()[sp.range()])).collect();
        let vectors: Vec<[f32; 2]> = packed.iter().map(|t| [t.len() as f32, t.split(' ').count() as f32]).collect();
        let mut at = 0;
        inputs
            .iter()
            .map(|(_, s)| {
                let n = s.spans.len();
                let data = vectors[at..at + n].iter().flatten().copied().collect();
                at += n;
                Ok((Vectors::new(2, data, Vec::new(), toy_model()),))
            })
            .collect()
    }

    fn model(&self) -> Option<ModelRef> {
        Some(toy_model())
    }

    fn execution(&self) -> Option<Execution> {
        Some(Execution { backend: "cpu".into(), device: "test".into(), precision: "f32".into() })
    }
}

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

/// A language layer naming English alone.
fn english(score: f64) -> Language {
    let mut code = Strs::default();
    code.push("en");
    Language { code, score: vec![score] }
}

/// Declines a document it finds no language evidence in, rather than
/// guessing a language.
struct Detect;

impl Annotator for Detect {
    const ID: &'static str = "test:language";
    type Reads = (View,);
    type Writes = (Language,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Language,), Error> {
        if !view.text().chars().any(char::is_alphabetic) {
            return Err(Error::Skipped("no language evidence".into()));
        }
        Ok((english(0.9),))
    }
}

/// Strips a plural "s" from English words. It needs the language, so a
/// document whose language was not found gets no lemmas.
struct Stemmer;

impl Annotator for Stemmer {
    const ID: &'static str = "test:stemmer";
    type Reads = (View, Language, Tokens);
    type Writes = (Lemmas,);

    fn annotate(&self, (view, language, tokens): Refs<'_, Self>) -> Result<(Lemmas,), Error> {
        let text = view.text();
        let mut lemma = Strs::default();
        for (_, s) in tokens.spans.iter() {
            let word = &text[s.range()];
            lemma.push(if language.best().is_some_and(|(code, _)| code == "en") {
                word.strip_suffix('s').unwrap_or(word)
            } else {
                word
            });
        }
        Ok((Lemmas { lemma },))
    }
}

/// Whitespace tokens: per sentence when the document has sentences, over
/// the whole text when it does not.
struct LooseTokenizer;

impl Annotator for LooseTokenizer {
    const ID: &'static str = "test:loose-tokens";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens,);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        let text = view.text();
        let ranges: Vec<(Option<Ix<Sentences>>, Span)> = match sentences {
            Some(s) => s.spans.iter().map(|(i, span)| (Some(i), span)).collect(),
            None => vec![(None, Span::new(0, text.len() as u32))],
        };
        let mut spans = Spans::new(view.id());
        let mut sentence = Vec::new();
        for (s, range) in ranges {
            let mut at = range.start as usize;
            for word in text[range.range()].split(' ') {
                if !word.is_empty() {
                    spans.push(Span::new(at as u32, (at + word.len()) as u32));
                    sentence.extend(s);
                }
                at += word.len() + 1;
            }
        }
        Ok((Tokens { spans, sentence: sentences.map(|_| sentence) },))
    }
}

/// Lowercases each word, and strips the "s" of one tagged VBZ: it reads
/// the tags and the tokens they are for.
struct Lemmatizer;

impl Annotator for Lemmatizer {
    const ID: &'static str = "test:lemmatizer";
    type Reads = (View, Tokens, PosTags);
    type Writes = (Lemmas,);

    fn annotate(&self, (view, tokens, tags): Refs<'_, Self>) -> Result<(Lemmas,), Error> {
        let mut lemma = Strs::default();
        for ((_, s), &t) in tokens.spans.iter().zip(&tags.tags) {
            let word = view.text()[s.range()].to_lowercase();
            lemma.push(if tags.tagset.label(t) == "VBZ" { word.strip_suffix('s').unwrap_or(&word) } else { &word });
        }
        Ok((Lemmas { lemma },))
    }
}

/// Panics on a text it cannot handle. Run over a batch, the panic takes
/// every document of the batch down with it.
struct Fragile;

impl Annotator for Fragile {
    const ID: &'static str = "test:fragile";
    type Reads = (View,);
    type Writes = (Language,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Language,), Error> {
        assert!(!view.text().contains("PANIC"), "cannot handle this one");
        Ok((english(1.0),))
    }
}

const FOLDED: Key<View> = Key::named("folded");

fn pipeline(calls: Arc<AtomicUsize>) -> Analyzer {
    // Added out of order on purpose: the analyzer orders by data.
    Analyzer::builder()
        .step(Bound::new(Tagger { tagset: TagSet::new("toy", &["X", "VBZ"]) }).read(0, "folded"))
        .step(Bound::new(Embedder { calls }).read(0, "folded"))
        .step(Bound::new(Tokenizer).read(0, "folded"))
        .step(Bound::new(Sentencer).read(0, "folded"))
        .step(Bound::new(Fold).write(0, "folded"))
        .build()
        .unwrap()
}
