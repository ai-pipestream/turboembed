//! `EnglishContractions` and `EnglishContractionsWithPos`: the
//! expectations of OpenNLP's EnglishContractionCharSequenceNormalizerTest,
//! EnglishContractionBoundaryTest and EnglishContractionContractTest, the
//! additions, and the part-of-speech resolution of 's and 'd.

mod scanners_support;

use std::borrow::Cow;
use std::sync::Arc;

use scanners_support::run;
use turbo_analysis::{Annotator, Document, Refs, Span, Spans, View};
use turbo_analysis_layers::{PosTags, TagSet, Tokens};
use turbo_normalize::{EnglishContractions, EnglishContractionsWithPos, apply, normalize};

fn expand(input: &str) -> String {
    run(&EnglishContractions, input)
}

fn span(s: usize, e: usize) -> Span {
    Span::new(s as u32, e as u32)
}

#[test]
fn expands_unambiguous_english_contractions() {
    for (input, expected) in [
        ("can't", "can not"),
        ("won't", "will not"),
        ("shan't", "shall not"),
        ("don't", "do not"),
        ("we're", "we are"),
        ("they've", "they have"),
        ("I'll", "I will"),
        ("I'm", "I am"),
        ("let's", "let us"),
        ("CAN'T", "CAN NOT"),
        ("Won't", "Will not"),
        ("can\u{2019}t", "can not"),
        ("can\u{2BC}t", "can not"),
        ("can\u{FF07}t", "can not"),
        ("we can't and they won't", "we can not and they will not"),
    ] {
        assert_eq!(expand(input), expected, "{input:?}");
    }
}

#[test]
fn leaves_ambiguous_and_lexicalized_apostrophes_untouched() {
    for input in ["he's", "she'd", "John's", "ain't", "o'clock"] {
        assert!(matches!(apply(&EnglishContractions, input), Cow::Borrowed(_)), "{input:?}");
    }
}

#[test]
fn alignment_keeps_the_original_contraction() {
    let (out, a) = normalize(&EnglishContractions, "can't");
    assert_eq!(out, "can not");
    assert_eq!(a.span_to_parent(span(0, 7)), span(0, 5));
    assert_eq!(a.span_to_parent(span(0, 3)), span(0, 3));
    assert_eq!(a.span_to_parent(span(4, 7)), span(3, 5));
    let (out, a) = normalize(&EnglishContractions, "we 'can't' leave");
    assert_eq!(out, "we 'can not' leave");
    assert_eq!(a.span_to_parent(span(4, 11)), span(4, 9));
    // A multi-byte apostrophe: the expansion maps back to all of it.
    let (out, a) = normalize(&EnglishContractions, "can\u{2019}t");
    assert_eq!(out, "can not");
    assert_eq!(a.span_to_parent(span(3, 7)), span(3, 7));
}

#[test]
fn does_not_expand_part_of_a_longer_word() {
    for input in [
        "\u{E9}can't",
        "can't\u{E9}",
        "\u{10400}can't",
        "can't\u{10400}",
        "_can't",
        "can't_",
        "1can't",
        "can't1",
        "e\u{301}can't",
        "can't\u{301}",
        "can't\u{200D}go",
        "rock'can't",
        "can't'roll",
        "can't've",
        "I.don't",
        "don't.com",
    ] {
        let (out, a) = normalize(&EnglishContractions, input);
        assert_eq!(out, input);
        assert!(a.edits().is_empty(), "{input:?}");
    }
}

#[test]
fn expands_quoted_and_punctuation_delimited_words() {
    for (input, expected) in [
        ("'can't'", "'can not'"),
        ("\u{2018}can\u{2019}t\u{2019}", "\u{2018}can not\u{2019}"),
        ("'we're ready'", "'we are ready'"),
        ("'I'm'", "'I am'"),
        ("(don't)", "(do not)"),
        ("\u{1F642}can't!", "\u{1F642}can not!"),
        ("x-can't", "x-can not"),
        ("can't:", "can not:"),
        ("can't,", "can not,"),
    ] {
        let (out, a) = normalize(&EnglishContractions, input);
        assert_eq!(out, expected, "{input:?}");
        assert_eq!(a.span_to_parent(span(0, out.len())), span(0, input.len()));
    }
}

#[test]
fn quoted_expansion_offsets() {
    for (contraction, expansion) in
        [("can't", "can not"), ("won't", "will not"), ("shan't", "shall not"), ("we're", "we are")]
    {
        let prefix = "\u{1F642} '";
        let input = format!("{prefix}{contraction}' now");
        let (out, a) = normalize(&EnglishContractions, &input);
        assert_eq!(out, format!("{prefix}{expansion}' now"));
        let p = prefix.len();
        assert_eq!(a.span_to_parent(span(p, p + expansion.len())), span(p, p + contraction.len()));
        assert_eq!(a.span_to_parent(span(p + expansion.len(), out.len())), span(p + contraction.len(), input.len()));
    }
}

#[test]
fn supported_forms_in_both_cases() {
    let forms = [
        ("aren't", "are not"),
        ("couldn't", "could not"),
        ("daren't", "dare not"),
        ("didn't", "did not"),
        ("don't", "do not"),
        ("doesn't", "does not"),
        ("hadn't", "had not"),
        ("hasn't", "has not"),
        ("haven't", "have not"),
        ("isn't", "is not"),
        ("mayn't", "may not"),
        ("mightn't", "might not"),
        ("mustn't", "must not"),
        ("needn't", "need not"),
        ("oughtn't", "ought not"),
        ("shouldn't", "should not"),
        ("wasn't", "was not"),
        ("weren't", "were not"),
        ("wouldn't", "would not"),
        ("how're", "how are"),
        ("there're", "there are"),
        ("they're", "they are"),
        ("what're", "what are"),
        ("when're", "when are"),
        ("where're", "where are"),
        ("who're", "who are"),
        ("why're", "why are"),
        ("you're", "you are"),
        ("could've", "could have"),
        ("I've", "I have"),
        ("might've", "might have"),
        ("must've", "must have"),
        ("should've", "should have"),
        ("we've", "we have"),
        ("who've", "who have"),
        ("would've", "would have"),
        ("you've", "you have"),
        ("he'll", "he will"),
        ("how'll", "how will"),
        ("it'll", "it will"),
        ("she'll", "she will"),
        ("that'll", "that will"),
        ("there'll", "there will"),
        ("they'll", "they will"),
        ("we'll", "we will"),
        ("what'll", "what will"),
        ("when'll", "when will"),
        ("where'll", "where will"),
        ("who'll", "who will"),
        ("why'll", "why will"),
        ("you'll", "you will"),
    ];
    for (input, expansion) in forms {
        let (out, a) = normalize(&EnglishContractions, input);
        assert_eq!(out, expansion);
        assert_eq!(a.span_to_parent(span(0, out.len())), span(0, input.len()));
        assert_eq!(expand(&input.to_uppercase()), expansion.to_uppercase());
        assert_eq!(expand(expansion), expansion);
    }
}

/// Forms OpenNLP leaves alone that this normalizer expands: `y'all` and
/// the subjects added to the 're, 've and 'll lists. Each reading is the
/// only one the form has.
#[test]
fn additions_beyond_opennlp() {
    for (input, expansion) in [
        ("y'all", "you all"),
        ("Y'all", "You all"),
        ("Y'ALL", "YOU ALL"),
        ("here're", "here are"),
        ("these're", "these are"),
        ("those're", "those are"),
        ("may've", "may have"),
        ("there've", "there have"),
        ("these've", "these have"),
        ("those've", "those have"),
        ("what've", "what have"),
        ("where've", "where have"),
        ("here'll", "here will"),
        ("this'll", "this will"),
        ("these'll", "these will"),
        ("those'll", "those will"),
        ("which'll", "which will"),
    ] {
        assert_eq!(expand(input), expansion, "{input:?}");
    }
    let (_, a) = normalize(&EnglishContractions, "y'all");
    // "you" stands for "y" and the space for the apostrophe, each its own
    // edit; "all" is kept and maps to itself.
    assert_eq!(a.span_to_parent(span(0, 3)), span(0, 1));
    assert_eq!(a.span_to_parent(span(3, 4)), span(1, 2));
    assert_eq!(a.span_to_parent(span(4, 7)), span(2, 5));
}

#[test]
fn rejected_and_accepted_candidates_together() {
    for unsupported in ["_can't", "can't1", "\u{E9}can't", "can't've", "he's", ""] {
        let input = format!("{unsupported} won't {unsupported} 'can't' {unsupported}");
        let expected = format!("{unsupported} will not {unsupported} 'can not' {unsupported}");
        let (out, a) = normalize(&EnglishContractions, &input);
        assert_eq!(out, expected);
        let start = expected.find("will not").unwrap();
        let u = unsupported.len();
        assert_eq!(a.span_to_parent(span(start, start + 8)), span(u + 1, u + 6));
    }
}

#[test]
fn shared_across_threads() {
    let handles: Vec<_> = (0..8)
        .map(|i| {
            std::thread::spawn(move || {
                let prefix = format!("item {i} ");
                let (out, a) = normalize(&EnglishContractions, &format!("{prefix}'can't' _won't"));
                assert_eq!(out, format!("{prefix}'can not' _won't"));
                let p = prefix.len();
                assert_eq!(a.span_to_parent(span(p + 1, p + 8)), span(p + 1, p + 6));
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

/// Runs the POS-aware step on `text` with whitespace tokens split before
/// an apostrophe (Penn Treebank style) and the given tags.
fn with_pos(text: &str, tags: &[&str]) -> String {
    let doc = Document::new(text);
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(view.id());
    let mut at = 0;
    for word in text.split(' ') {
        let cut = word.find('\'').filter(|&k| k > 0).unwrap_or(word.len());
        spans.push(span(at, at + cut));
        if cut < word.len() {
            spans.push(span(at + cut, at + word.len()));
        }
        at += word.len() + 1;
    }
    let tokens = Tokens { spans, sentence: None };
    assert_eq!(tokens.spans.len(), tags.len(), "a tag per token of {text:?}");
    let tagset = TagSet::new("ptb", &["PRP", "VBZ", "VBN", "VBG", "VB", "MD", "VBD", "POS", "NN", "JJ", "RB", "NNP"]);
    let tags =
        PosTags { tagset: tagset.clone(), tags: tags.iter().map(|t| tagset.tag(t).unwrap()).collect(), prob: None };
    let input: Refs<'_, EnglishContractionsWithPos> = (view, Some(&tokens), Some(&tags));
    let (out,) = EnglishContractionsWithPos.annotate(input).unwrap();
    scanners_support::check_alignment(text, out.text(), out.alignment());
    out.text().to_string()
}

#[test]
fn pos_tags_resolve_s_and_d() {
    assert_eq!(with_pos("he's here", &["PRP", "VBZ", "RB"]), "he is here");
    assert_eq!(with_pos("he's gone", &["PRP", "VBZ", "VBN"]), "he has gone");
    assert_eq!(with_pos("HE'S GONE", &["PRP", "VBZ", "VBN"]), "HE HAS GONE");
    assert_eq!(with_pos("John's book", &["NNP", "POS", "NN"]), "John's book");
    assert_eq!(with_pos("I'd go", &["PRP", "MD", "VB"]), "I would go");
    assert_eq!(with_pos("I'd gone", &["PRP", "VBD", "VBN"]), "I had gone");
    assert_eq!(with_pos("I'd rather", &["PRP", "MD", "RB"]), "I'd rather");
    // The unambiguous forms expand as before, tags or not.
    assert_eq!(with_pos("we're sure it's done", &["PRP", "VBZ", "JJ", "PRP", "VBZ", "VBN"]), "we are sure it has done");
}

#[test]
fn pos_step_without_tags_is_the_plain_normalizer() {
    let doc = Document::new("he's sure we're done");
    let view = doc.get::<View>().unwrap();
    let input: Refs<'_, EnglishContractionsWithPos> = (view, None, None);
    let (out,) = EnglishContractionsWithPos.annotate(input).unwrap();
    assert_eq!(out.text(), "he's sure we are done");
}

#[test]
fn pos_step_refuses_tokens_of_another_view() {
    let doc = Document::new("he's here");
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(turbo_analysis::ViewId(3));
    spans.push(span(0, 4));
    let tokens = Tokens { spans, sentence: None };
    let tagset: Arc<TagSet> = TagSet::new("ptb", &["VBZ"]);
    let tags = PosTags { tagset: tagset.clone(), tags: vec![tagset.tag("VBZ").unwrap()], prob: None };
    let input: Refs<'_, EnglishContractionsWithPos> = (view, Some(&tokens), Some(&tags));
    assert!(EnglishContractionsWithPos.annotate(input).is_err());
}

#[test]
#[ignore = "timing; run with --release -- --ignored --nocapture"]
fn cost_per_megabyte() {
    let mb = 1_000_000;
    let piece = "We can't and they won't, though 'I'm' sure she'd say y'all shouldn't wait at caf\u{E9} \u{2018}don\u{2019}t\u{2019}. ";
    let text = piece.repeat(mb / piece.len() + 1);
    let mut out_len = normalize(&EnglishContractions, &text).0.len();
    let rounds = 20;
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        out_len += normalize(&EnglishContractions, &text).0.len();
    }
    let per_round = start.elapsed().as_secs_f64() / f64::from(rounds);
    let mbs = text.len() as f64 / 1_000_000.0;
    println!("EnglishContractions {:>8.2} ms per MB ({out_len} bytes written)", per_round * 1000.0 / mbs);
}
