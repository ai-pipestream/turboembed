//! TokenizerMe in a pipeline: after a toy sentence annotator, writing the
//! tokens with their sentence column and the confidence per token, with
//! the model recorded in the provenance. The token values are what
//! TokenizerME returned on the JVM for each sentence.

mod me_common;

use me_common::model;
use turbo_analysis::{Analyzer, Annotator, Document, Error, Key, Refs, Span, Spans, View};
use turbo_analysis_layers::{Confidence, Sentences, Tokens};
use turbo_tokenize::TokenizerOptions;
use turbo_tokenize::me::TokenizerMe;

/// Sentences end after ". "; a text of only whitespace has none.
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
        if !text[start..].trim().is_empty() {
            spans.push(Span::new(start as u32, text.len() as u32));
        }
        Ok((Sentences { spans },))
    }
}

fn english() -> TokenizerMe {
    TokenizerMe::new(model("opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin"), TokenizerOptions::default())
}

/// (token text, sentence, probability) of a document's layers.
fn layers(doc: &Document) -> Vec<(&str, Option<usize>, f32)> {
    let tokens = doc.get::<Tokens>().unwrap();
    let confidence = doc.get::<Confidence<Tokens>>().unwrap();
    assert_eq!(confidence.value.len(), tokens.spans.len());
    tokens
        .spans
        .iter()
        .enumerate()
        .map(|(i, (_, s))| {
            let sentence = tokens.sentence.as_ref().map(|c| c[i].get());
            (&doc.source()[s.range()], sentence, confidence.value[i])
        })
        .collect()
}

#[test]
fn tokens_with_sentences_and_confidence() {
    let analyzer = Analyzer::builder().add(Sentencer).boxed(english().into_step()).build().unwrap();
    let mut docs = vec![
        Document::new("The cat sat. It's 5 o'clock, isn't it?"),
        Document::new("  \n\t \u{3000} "),
        Document::new("Prices rose 3.5% in Q1. Dr. Who left."),
    ];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let p = |x: f64| x as f32;
    assert_eq!(
        layers(&docs[0]),
        [
            ("The", Some(0), p(0.9999123302765786)),
            ("cat", Some(0), p(0.9998775446210091)),
            ("sat", Some(0), p(0.9764420535542908)),
            (".", Some(0), 1.0),
            ("It's", Some(1), p(0.998677134411433)),
            ("5", Some(1), 1.0),
            ("o'clock", Some(1), p(0.9895741126332173)),
            (",", Some(1), 1.0),
            ("isn't", Some(1), p(0.9964921186776551)),
            ("it", Some(1), p(0.993784172275783)),
            ("?", Some(1), 1.0),
        ]
    );
    // No sentences, so no tokens: an empty layer, complete.
    assert_eq!(layers(&docs[1]), []);
    assert_eq!(docs[1].get::<Tokens>().unwrap().sentence.as_deref(), Some(&[][..]));
    assert_eq!(
        layers(&docs[2]),
        [
            ("Prices", Some(0), p(0.9999035672167623)),
            ("rose", Some(0), p(0.9998512632579185)),
            ("3.5", Some(0), p(0.8719787625742007)),
            ("%", Some(0), 1.0),
            ("in", Some(0), p(0.9999968838916649)),
            ("Q1", Some(0), p(0.9975105717782285)),
            (".", Some(0), 1.0),
            ("Dr.", Some(1), p(0.939533219482152)),
            ("Who", Some(2), p(0.9999621574025851)),
            ("left", Some(2), p(0.9807677562261841)),
            (".", Some(2), 1.0),
        ]
    );
    let tokens = docs[2].get::<Tokens>().unwrap();
    assert_eq!(tokens.of(turbo_analysis::Ix::new(1)), turbo_analysis::IxRange::new(7, 8));

    // The provenance names the model file, and the sentences read.
    let provenance = docs[0].provenance(Key::<Tokens>::DEFAULT.id()).unwrap();
    assert_eq!(provenance.annotator, "turbo-tokenize:maxent");
    let m = provenance.model.as_ref().unwrap();
    assert_eq!(m.id, "TokenizerME:en");
    assert_eq!(m.revision, "2.5.4");
    assert_eq!(m.sha256, "95139111197ba9f1165fc12be8d4cbac05ea3912f995d167efa1a0e04e3e649a");
    assert_eq!(provenance.reads, [Key::<View>::DEFAULT.id(), Key::<Sentences>::DEFAULT.id()]);
    assert_eq!(provenance.config, TokenizerOptions::default().config_hash());
    let confidence = docs[0].provenance(Key::<Confidence<Tokens>>::DEFAULT.id()).unwrap();
    assert_eq!(confidence.model, provenance.model);
}

#[test]
fn without_sentences_the_view_is_one_unit() {
    let analyzer = Analyzer::builder().boxed(english().into_step()).build().unwrap();
    let mut docs = vec![Document::new("The cat sat. Who left.")];
    analyzer.run(&mut docs)[0].clone().unwrap();
    let got = layers(&docs[0]);
    let words: Vec<&str> = got.iter().map(|t| t.0).collect();
    // The units are the same whitespace units as per sentence.
    assert_eq!(words, ["The", "cat", "sat", ".", "Who", "left", "."]);
    assert!(got.iter().all(|t| t.1.is_none()));
    let provenance = docs[0].provenance(Key::<Tokens>::DEFAULT.id()).unwrap();
    assert_eq!(provenance.reads, [Key::<View>::DEFAULT.id()]);
}

#[test]
fn the_plain_bound_annotator_is_refused_by_the_analyzer() {
    // Why `into_step` exists: the confidence is aligned to the tokens the
    // same step writes, which `Analyzer::build` takes for an unread anchor.
    let e = Analyzer::builder().add(Sentencer).add(english()).build().err().unwrap();
    assert_eq!(
        e.to_string(),
        "pipeline: turbo-tokenize:maxent writes turbo:confidence<opennlp:tokens>, aligned to opennlp:tokens, without reading opennlp:tokens"
    );
}
