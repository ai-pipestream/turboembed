//! The tokenizers as pipeline steps: optional sentences, the sentence
//! column, provenance, and the word types aligned to the tokens.

use turbo_analysis::annotate::Step;
use turbo_analysis::{Analyzer, Annotator, Bound, Document, Error, Ix, IxRange, Key, Layer, Refs, Span, Spans, View};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_tokenize::{
    SimpleTokenizer, TokenizerOptions, UntypedWordTokenizer, WhitespaceTokenizer, WordTokenizer, WordType, WordTypes,
};

/// A toy sentence splitter: a sentence ends after ". ".
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

const TEXT: &str = "Hi there. \u{65e5}\u{672c} 42. Bye \u{1f600}";

fn words(doc: &Document) -> Vec<&str> {
    let tokens = doc.get::<Tokens>().unwrap();
    tokens.spans.iter().map(|(_, s)| &doc.source()[s.range()]).collect()
}

#[test]
fn word_tokens_per_sentence() {
    let analyzer = Analyzer::builder().add(UntypedWordTokenizer::default()).add(Sentencer).build().unwrap();
    assert_eq!(analyzer.plan(), [vec!["test:sentences"], vec!["turbo-tokenize:uax29-untyped"]]);
    let mut docs = vec![Document::new(TEXT)];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let doc = &docs[0];
    assert_eq!(words(doc), ["Hi", "there", "\u{65e5}", "\u{672c}", "42", "Bye", "\u{1f600}"]);
    let tokens = doc.get::<Tokens>().unwrap();
    assert_eq!(tokens.of(Ix::new(0)), IxRange::new(0, 2));
    assert_eq!(tokens.of(Ix::new(1)), IxRange::new(2, 5));
    assert_eq!(tokens.of(Ix::new(2)), IxRange::new(5, 7));
    // The optional input was there, so provenance lists it.
    let reads = &doc.provenance(Key::<Tokens>::DEFAULT.id()).unwrap().reads;
    assert_eq!(reads, &[Key::<View>::DEFAULT.id(), Key::<Sentences>::DEFAULT.id()]);
}

#[test]
fn whole_view_without_sentences() {
    let analyzer = Analyzer::builder().add(SimpleTokenizer::default()).build().unwrap();
    let mut docs = vec![Document::new("Mr. Smith's car."), Document::new("")];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    assert_eq!(words(&docs[0]), ["Mr", ".", "Smith", "'", "s", "car", "."]);
    assert!(docs[0].get::<Tokens>().unwrap().sentence.is_none());
    assert!(docs[1].get::<Tokens>().unwrap().is_empty());
    let p = docs[0].provenance(Key::<Tokens>::DEFAULT.id()).unwrap();
    assert_eq!(p.reads, [Key::<View>::DEFAULT.id()]);
    assert_eq!(p.annotator, "turbo-tokenize:simple");
    assert_eq!(p.config, SimpleTokenizer::default().config_hash());
}

#[test]
fn whitespace_tokens_keep_newlines() {
    let options = TokenizerOptions { keep_newlines: true, ..TokenizerOptions::default() };
    let analyzer = Analyzer::builder().add(WhitespaceTokenizer::new(options)).add(Sentencer).build().unwrap();
    let mut docs = vec![Document::new("a b.\nc. d")];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    assert_eq!(words(&docs[0]), ["a", "b.", "\n", "c.", "d"]);
    let tokens = docs[0].get::<Tokens>().unwrap();
    assert_eq!(tokens.of(Ix::new(0)), IxRange::new(0, 4));
    assert_eq!(tokens.of(Ix::new(1)), IxRange::new(4, 5));
    let mut plain = vec![Document::new("a\nb")];
    Analyzer::builder().add(WhitespaceTokenizer::new(options)).build().unwrap().run(&mut plain);
    assert_eq!(words(&plain[0]), ["a", "\n", "b"]);
}

/// The analyzer refuses, when it builds a pipeline, a step whose aligned
/// output hangs on a layer it does not read, and it does not yet count the
/// step's own outputs as such a layer. `WordTokenizer` writes the tokens
/// and the word types aligned to them, so it is refused today. This test
/// pins that; when the analyzer accepts an anchor among a step's own
/// outputs, it fails, and `typed_word_tokens` should run through the
/// analyzer instead.
#[test]
fn analyzer_refuses_types_aligned_to_own_tokens() {
    let err = Analyzer::builder().add(WordTokenizer::new()).build().err().unwrap();
    assert_eq!(
        err,
        Error::Pipeline(
            "turbo-tokenize:uax29 writes turbo:word-types, aligned to opennlp:tokens, without reading opennlp:tokens"
                .into()
        )
    );
}

/// `WordTokenizer` driven through the same `Step` interface the analyzer
/// uses, after a pipeline wrote the sentences: the document checks that
/// the word types are as long as the tokens when it takes them.
#[test]
fn typed_word_tokens() {
    let analyzer = Analyzer::builder().add(Sentencer).build().unwrap();
    let mut docs = vec![Document::new(TEXT), Document::new("...")];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let step = Bound::new(WordTokenizer::new());
    let out = step.compute(&docs);
    let mut status = vec![Ok(()), Ok(())];
    step.store(out, &mut docs, &mut status);
    assert!(status.iter().all(Result::is_ok), "{status:?}");

    let doc = &docs[0];
    let types = doc.get::<WordTypes>().unwrap();
    assert_eq!(types.len(), doc.get::<Tokens>().unwrap().len());
    use WordType::*;
    assert_eq!(types.kind, [Alphanumeric, Alphanumeric, Ideographic, Ideographic, Numeric, Alphanumeric, Emoji]);
    assert_eq!(doc.made_on(Key::<WordTypes>::DEFAULT.id()), [Key::<Tokens>::DEFAULT.id()]);
    assert_eq!(doc.get::<Tokens>().unwrap().of(Ix::new(1)), IxRange::new(2, 5));
    // Punctuation alone is no word: an empty, complete pair of layers.
    assert!(docs[1].get::<Tokens>().unwrap().is_empty() && docs[1].get::<WordTypes>().unwrap().is_empty());
}
