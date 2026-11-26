//! Outputs a step aligns to a layer it writes itself.

use super::*;

/// Whitespace tokens with a confidence for each, from one call, as a
/// statistical tokenizer reports them. `short` leaves the last token
/// without a confidence.
struct ScoringTokenizer {
    short: bool,
}

impl Annotator for ScoringTokenizer {
    const ID: &'static str = "test:scoring-tokens";
    type Reads = (View,);
    type Writes = (Tokens, Confidence<Tokens>);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Tokens, Confidence<Tokens>), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut at = 0;
        for word in text.split(' ') {
            if !word.is_empty() {
                spans.push(Span::new(at as u32, (at + word.len()) as u32));
            }
            at += word.len() + 1;
        }
        let n = spans.len() - usize::from(self.short);
        let confidence = (0..n).map(|i| 1.0 / (i + 1) as f64).collect();
        Ok((Tokens { spans, sentence: None }, Confidence::new(confidence)))
    }
}

#[test]
fn a_step_aligns_an_output_to_a_layer_it_writes_itself() {
    let a = Analyzer::builder().add(ScoringTokenizer { short: false }).build().unwrap();
    let mut docs = vec![Document::new("Dogs bark loudly")];
    assert!(a.run(&mut docs)[0].is_ok());
    let doc = &docs[0];
    let (tokens, confidence) = (Key::<Tokens>::DEFAULT.id(), Key::<Confidence<Tokens>>::DEFAULT.id());
    assert_eq!(doc.get::<Confidence<Tokens>>().unwrap().value, [1.0, 0.5, 1.0 / 3.0]);
    // The confidence is made on the tokens the step wrote. Its provenance
    // still lists only what the step read, and names the tokens apart.
    assert_eq!(doc.made_on(confidence), [tokens]);
    let p = doc.provenance(confidence).unwrap();
    assert_eq!((p.reads.as_slice(), p.on.as_slice()), ([Key::<View>::DEFAULT.id()].as_slice(), [tokens].as_slice()));
    assert!(doc.provenance(tokens).unwrap().on.is_empty());

    // A confidence column one short of its tokens is refused by the
    // document; the tokens before it are kept.
    let a = Analyzer::builder().add(ScoringTokenizer { short: true }).build().unwrap();
    let mut docs = vec![Document::new("Dogs bark loudly")];
    let e = a.run(&mut docs).remove(0).unwrap_err();
    assert_eq!(e, Error::Misaligned { layer: confidence, anchor: tokens, len: 2, anchor_len: 3 });
    assert_eq!(docs[0].status(tokens), Some(&Status::Complete));
    assert_eq!(docs[0].status(confidence), Some(&Status::Failed(e)));
}

#[test]
fn a_named_output_aligns_to_the_instance_its_step_wrote() {
    const MINE: Key<Tokens> = Key::named("mine");
    const SCORES: Key<Confidence<Tokens>> = Key::named("mine");
    // The document arrives with default tokens of another length, so a
    // confidence measured against them would be refused as misaligned.
    let mut doc = Document::new("Dogs bark loudly");
    let mut one = Spans::new(ViewId::SOURCE);
    one.push(Span::new(0, 16));
    doc.add(&Key::DEFAULT, Tokens { spans: one, sentence: None }, Provenance::by("corpus")).unwrap();

    let a = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(ScoringTokenizer { short: false }).write(0, "mine").write(1, "mine"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert!(a.run(&mut docs)[0].is_ok());
    assert_eq!(docs[0].made_on(SCORES.id()), [MINE.id()]);
    assert_eq!(docs[0].get_key(&SCORES).unwrap().len(), 3);
}

#[test]
fn an_index_into_a_layer_the_step_wrote_resolves_to_that_instance() {
    /// Sentences and the tokens in them, from one call.
    struct Splitter;
    impl Annotator for Splitter {
        const ID: &'static str = "test:splitter";
        type Reads = (View,);
        type Writes = (Sentences, Tokens);
        fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences, Tokens), Error> {
            let (mut s, mut t) = (Spans::new(view.id()), Spans::new(view.id()));
            s.push(Span::new(0, 4));
            s.push(Span::new(5, 9));
            t.push(Span::new(0, 4));
            t.push(Span::new(5, 9));
            Ok((Sentences { spans: s }, Tokens { spans: t, sentence: Some(vec![Ix::new(0), Ix::new(1)]) }))
        }
    }
    let a = Analyzer::builder().step(Bound::new(Splitter).write(0, "x").write(1, "x")).build().unwrap();
    let mut docs = vec![Document::new("Dogs bark")];
    assert!(a.run(&mut docs)[0].is_ok());
    assert_eq!(docs[0].made_on(Key::<Tokens>::named("x").id()), [Key::<Sentences>::named("x").id()]);
}

#[test]
fn an_aligned_output_needs_its_base_read_or_written_first() {
    /// Writes a confidence column before the tokens it is for.
    struct Backwards;
    impl Annotator for Backwards {
        const ID: &'static str = "test:backwards";
        type Reads = (View,);
        type Writes = (Confidence<Tokens>, Tokens);
        fn annotate(&self, _: Refs<'_, Self>) -> Result<(Confidence<Tokens>, Tokens), Error> {
            unreachable!("refused when the pipeline is built")
        }
    }
    let e = Analyzer::builder().add(Backwards).build().err().unwrap();
    assert_eq!(
        e.to_string(),
        "pipeline: test:backwards writes turbo:confidence<turbo:tokens>, aligned to turbo:tokens, \
         before turbo:tokens: the base must come first in its writes"
    );

    /// Writes a confidence column for tokens it neither reads nor writes.
    struct Unread;
    impl Annotator for Unread {
        const ID: &'static str = "test:unread";
        type Reads = (View,);
        type Writes = (Confidence<Tokens>,);
        fn annotate(&self, _: Refs<'_, Self>) -> Result<(Confidence<Tokens>,), Error> {
            unreachable!("refused when the pipeline is built")
        }
    }
    let e = Analyzer::builder().add(Tokenizer).add(Sentencer).add(Unread).build().err().unwrap();
    assert_eq!(
        e,
        Error::Pipeline(
            "test:unread writes turbo:confidence<turbo:tokens>, aligned to turbo:tokens, \
             without reading turbo:tokens"
                .into()
        )
    );
}
