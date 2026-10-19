use super::*;
use turbo_analysis::{Analyzer, Document, Provenance, Spans, ViewId};

/// All 28 test sentences from Python `vaderSentiment.py`'s `__main__`
/// block (`sentences` and `tricky_sentences`), with their exact
/// `(neg, neu, pos, compound)` scores from `SentimentIntensityAnalyzer`.
const VADER_REFERENCE_SUITE: &[(&str, f64, f64, f64, f64)] = &[
    ("VADER is smart, handsome, and funny.", 0.000, 0.254, 0.746, 0.8316),
    ("VADER is smart, handsome, and funny!", 0.000, 0.248, 0.752, 0.8439),
    ("VADER is very smart, handsome, and funny.", 0.000, 0.299, 0.701, 0.8545),
    ("VADER is VERY SMART, handsome, and FUNNY.", 0.000, 0.246, 0.754, 0.9227),
    ("VADER is VERY SMART, handsome, and FUNNY!!!", 0.000, 0.233, 0.767, 0.9342),
    ("VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!", 0.000, 0.294, 0.706, 0.9469),
    ("VADER is not smart, handsome, nor funny.", 0.646, 0.354, 0.000, -0.7424),
    ("The book was good.", 0.000, 0.508, 0.492, 0.4404),
    ("At least it isn't a horrible book.", 0.000, 0.678, 0.322, 0.4310),
    ("The book was only kind of good.", 0.000, 0.697, 0.303, 0.3832),
    (
        "The plot was good, but the characters are uncompelling and the dialog is not great.",
        0.327,
        0.579,
        0.094,
        -0.7042,
    ),
    ("Today SUX!", 0.779, 0.221, 0.000, -0.5461),
    ("Today only kinda sux! But I'll get by, lol", 0.127, 0.556, 0.317, 0.5249),
    ("Make sure you :) or :D today!", 0.000, 0.294, 0.706, 0.8633),
    ("Not bad at all", 0.000, 0.513, 0.487, 0.4310),
    ("Sentiment analysis has never been good.", 0.325, 0.675, 0.000, -0.3412),
    ("Sentiment analysis has never been this good!", 0.000, 0.621, 0.379, 0.5672),
    ("Most automated sentiment analysis tools are shit.", 0.375, 0.625, 0.000, -0.5574),
    ("With VADER, sentiment analysis is the shit!", 0.000, 0.583, 0.417, 0.6476),
    ("Other sentiment analysis tools can be quite bad.", 0.351, 0.649, 0.000, -0.5849),
    ("On the other hand, VADER is quite bad ass", 0.000, 0.423, 0.577, 0.8020),
    ("VADER is such a badass!", 0.000, 0.598, 0.402, 0.4003),
    ("Without a doubt, excellent idea.", 0.000, 0.341, 0.659, 0.7013),
    ("Roger Dodger is one of the most compelling variations on this theme.", 0.000, 0.834, 0.166, 0.2944),
    ("Roger Dodger is at least compelling as a variation on the theme.", 0.000, 0.853, 0.147, 0.2263),
    ("Roger Dodger is one of the least compelling variations on this theme.", 0.132, 0.868, 0.000, -0.1695),
    ("Not such a badass after all.", 0.289, 0.711, 0.000, -0.2584),
    ("Without a doubt, an excellent idea.", 0.000, 0.408, 0.592, 0.7013),
];

/// Sentences outside the upstream suite where an earlier port diverged,
/// scored with vaderSentiment 3.3.2. "no" before a lexicon word keeps its
/// 0.0 valence through the ALLCAPS and booster rules; a dampener before
/// "no" and a leading idiom are both scored by the preceding-word rules.
const VADER_EXTRA_SUITE: &[(&str, f64, f64, f64, f64)] = &[
    ("NO good", 1.000, 0.000, 0.000, -0.4835),
    ("very no good", 0.539, 0.201, 0.260, -0.3381),
    ("yeah right bad ass", 0.686, 0.098, 0.216, -0.7003),
    ("screwer NO bravely favoritisms", 1.000, 0.000, 0.000, -0.7313),
    ("no no good", 0.454, 0.189, 0.357, -0.1326),
    ("no not good", 0.420, 0.191, 0.389, -0.0412),
];

fn make_sentences_and_tokens(text: &str, words: &[&str]) -> (Sentences, Tokens) {
    let mut s_spans = Spans::new(ViewId::SOURCE);
    s_spans.push(Span::new(0, text.len() as u32));
    let sentences = Sentences { spans: s_spans };

    let mut t_spans = Spans::new(ViewId::SOURCE);
    let mut cursor = 0usize;
    for &w in words {
        let pos = text[cursor..].find(w).unwrap() + cursor;
        t_spans.push(Span::new(pos as u32, (pos + w.len()) as u32));
        cursor = pos + w.len();
    }
    let tokens = Tokens { spans: t_spans, sentence: None };
    (sentences, tokens)
}

#[test]
fn matches_python_vader_sentiment_on_all_reference_sentences() {
    let scorer = LexiconSentiment::new();
    assert_eq!(scorer.lexicon().len(), 7506);

    for &(sentence, exp_neg, exp_neu, exp_pos, exp_compound) in VADER_REFERENCE_SUITE.iter().chain(VADER_EXTRA_SUITE) {
        let s = scorer.score_text(sentence);
        assert_eq!(
            (s.neg_rounded(), s.neu_rounded(), s.pos_rounded(), s.compound_rounded()),
            (exp_neg, exp_neu, exp_pos, exp_compound),
            "mismatch on sentence: {sentence:?}"
        );
    }
}

#[test]
fn digest_follows_entries_not_source_bytes() {
    let a = VaderLexicon::from_tsv("# comment\ngood\t1.9\nbad\t-2.5\n").unwrap();
    let b = VaderLexicon::from_tsv("bad\t-2.5\t0.5\t[-3, -2]\ngood\t1.9\n").unwrap();
    assert_eq!(a.sha256_hex(), b.sha256_hex());
    let mut c = a.clone();
    c.insert("meh", -0.5);
    assert_ne!(a.sha256_hex(), c.sha256_hex());
    assert!(VaderLexicon::from_tsv("good\tnot-a-number\n").is_err());
}

#[test]
fn a_zeroed_no_that_still_scores_keeps_its_evidence_row() {
    let text = "NO good";
    let words = ["NO", "good"];
    let (_s, tokens) = make_sentences_and_tokens(text, &words);
    let scorer = LexiconSentiment::new().with_negation_mode(NegationMode::Window);
    let res = scorer.score_sentence(text, Span::new(0, 7), &tokens, IxRange::new(0, 2), None, None);
    assert_eq!(res.opinions.len(), 2);
    let no = res.opinions[0];
    assert_eq!(no.tokens, IxRange::new(0, 1));
    assert!((no.effective_polarity() + C_INCR).abs() < 1e-12);
    let good = res.opinions[1];
    assert_eq!(good.tokens, IxRange::new(1, 2));
    assert!((good.effective_polarity() - 1.9 * N_SCALAR).abs() < 1e-12);
    let sum: f64 = res.opinions.iter().map(|o| o.effective_polarity()).sum();
    assert!((normalize_score(sum, DEFAULT_ALPHA) - res.score.compound).abs() < 1e-12);
}

#[test]
fn double_negation_is_reported_as_not_negated() {
    // "no" two tokens back applies the "no" rule, then "not" applies the
    // negation window: the valence flips twice and ends positive.
    let text = "no not good";
    let words = ["no", "not", "good"];
    let (_s, tokens) = make_sentences_and_tokens(text, &words);
    let scorer = LexiconSentiment::new().with_negation_mode(NegationMode::Window);
    let res = scorer.score_sentence(text, Span::new(0, 11), &tokens, IxRange::new(0, 3), None, None);
    // "no" keeps its own lexicon valence here because "not" is not a lexicon word.
    assert_eq!(res.opinions.len(), 2);
    let op = res.opinions[1];
    assert_eq!(op.tokens, IxRange::new(2, 3));
    assert!(!op.negated);
    assert!((op.effective_polarity() - 1.9 * N_SCALAR * N_SCALAR).abs() < 1e-12);
}

#[test]
fn evidence_spans_point_at_exact_scored_tokens_and_preserve_effective_polarity() {
    let text = "The plot was good, but the dialog is not great.";
    let word_slices = ["The", "plot", "was", "good", ",", "but", "the", "dialog", "is", "not", "great", "."];
    let (sentences, tokens) = make_sentences_and_tokens(text, &word_slices);

    let mut doc = Document::new(text);
    doc.add(&Key::<Sentences>::DEFAULT, sentences, Provenance::by("test")).unwrap();
    doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(LexiconSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = &docs[0];

    let opinions = doc.get::<Opinions>().unwrap();
    let toks = doc.get::<Tokens>().unwrap();
    assert_eq!(opinions.len(), 2);

    // First opinion: "good" at token index 3 (before "but", scaled by 0.5).
    let op0 = opinions.get(Ix::new(0));
    assert_eq!(op0.tokens, IxRange::new(3, 4));
    assert_eq!(&text[opinions.span(Ix::new(0), toks).range()], "good");
    assert!(!op0.negated);
    assert!((op0.polarity - 1.9).abs() < 1e-12);
    assert!((op0.effective_polarity() - 1.9 * 0.5).abs() < 1e-12);

    // Second opinion: "great" at token index 10 (negated by "not" and after "but", scaled by -0.74 * 1.5).
    let op1 = opinions.get(Ix::new(1));
    assert_eq!(op1.tokens, IxRange::new(10, 11));
    assert_eq!(&text[opinions.span(Ix::new(1), toks).range()], "great");
    assert!(op1.negated);
    assert!((op1.polarity - 3.1).abs() < 1e-12);
    assert!((op1.effective_polarity() - 3.1 * N_SCALAR * 1.5).abs() < 1e-12);

    let cats = doc.get::<Categories<Sentences>>().unwrap();
    assert_eq!(cats.len(), 1);
    assert_eq!(cats.best_category(0), "negative");
}

#[test]
fn dependency_negation_scope_handles_long_distance_and_clause_boundaries() {
    // Sentence where negation "not" is 4 tokens before "good" ("is not in any way good"):
    // in a 3-token window "not" is too far to negate "good", whereas in the dependency parse
    // "not" is a direct `neg` dependent of "good".
    let text = "The service is not in any way good.";
    let words = ["The", "service", "is", "not", "in", "any", "way", "good", "."];
    let (sentences, tokens) = make_sentences_and_tokens(text, &words);

    // Window mode: "not" is at index 3, "good" is at index 7 (distance 4 > 3), so window mode misses negation.
    let scorer_win = LexiconSentiment::new().with_negation_mode(NegationMode::Window);
    let win_res = scorer_win.score_sentence(
        text,
        Span::new(0, text.len() as u32),
        &tokens,
        IxRange::new(0, words.len() as u32),
        None,
        None,
    );
    assert!(!win_res.used_dependencies);
    assert!(win_res.score.compound > 0.0);
    assert!(!win_res.opinions[0].negated);

    // Dependency mode: "not" (token 3) has head "good" (token 7) with relation "neg".
    let rel_tags = TagSet::new("ud", &["root", "nsubj", "cop", "neg", "case", "det", "obl", "punct"]);
    let deps = Dependencies {
        head: vec![
            Some(Ix::new(1)), // The -> service
            Some(Ix::new(7)), // service -> good
            Some(Ix::new(7)), // is -> good
            Some(Ix::new(7)), // not -> good
            Some(Ix::new(6)), // in -> way
            Some(Ix::new(6)), // any -> way
            Some(Ix::new(7)), // way -> good
            None,             // good (root)
            Some(Ix::new(7)), // . -> good
        ],
        relation: vec![Tag(5), Tag(1), Tag(2), Tag(3), Tag(4), Tag(5), Tag(6), Tag(0), Tag(7)],
        relations: rel_tags,
    };

    let mut doc = Document::new(text);
    doc.add(&Key::<Sentences>::DEFAULT, sentences, Provenance::by("test")).unwrap();
    doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test")).unwrap();
    doc.add(&Key::<Dependencies>::DEFAULT, deps, Provenance::by("test")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .add(LexiconSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = &docs[0];

    let opinions = doc.get::<Opinions>().unwrap();
    assert_eq!(opinions.len(), 1);
    let op = opinions.get(Ix::new(0));
    assert_eq!(op.tokens, IxRange::new(7, 8));
    assert!(op.negated);
    assert!(op.effective_polarity() < 0.0);

    let cats = doc.get::<Categories<Sentences>>().unwrap();
    assert_eq!(cats.best_category(0), "negative");

    // Provenance records that `opennlp:dependencies` was read.
    let prov = doc.provenance(Key::<Opinions>::DEFAULT.id()).unwrap();
    assert!(prov.reads.contains(&Key::<Dependencies>::DEFAULT.id()));
}

#[test]
fn dependency_negation_avoids_false_negation_across_clauses() {
    // "Not" negates "sure", not "good" inside the complement clause ("Not sure the food is good"):
    // in 3-token window mode ("Not sure the good"), "not" is 3 tokens before "good" and negates it,
    // whereas in dependency mode "not" attaches to "sure" and "good" is the head of "ccomp" without
    // negation on "good" (when "sure" is not in the lexicon, or when "not" attaches to an unrelated verb).
    let text = "Not knowing the food is good.";
    let words = ["Not", "knowing", "the", "food", "is", "good", "."];
    // And in a 3-token distance case: "Not saying food is good" -> "Not"(0) "saying"(1) "it"(2) "good"(3):
    let text2 = "Not saying it is good.";
    let words2 = ["Not", "saying", "it", "is", "good", "."];
    let (_s2, tokens2) = make_sentences_and_tokens(text2, &words2);

    // In window mode, "Not" is 4 tokens before "good" in text2, or 3 tokens before "good" in "Not saying anything good.":
    let text3 = "Not saying things are good.";
    let words3 = ["Not", "saying", "things", "are", "good", "."];
    let _ = (text, words, tokens2, text3, words3);

    // 3-token window case: "Not"(0) "saying"(1) "everything"(2) "good"(3):
    // Window mode sees "Not" at i - 3 and negates "good", whereas in dependency mode "Not" modifies
    // "saying" (root) and "good" modifies "everything" (obj -> amod on "everything", whose head is "saying",
    // so "good"'s head is "everything", which has NO negation dependent!).
    let sent_text = "Not saying everything good.";
    let sent_words = ["Not", "saying", "everything", "good", "."];
    let (_sents, toks) = make_sentences_and_tokens(sent_text, &sent_words);

    let win_scorer = LexiconSentiment::new().with_negation_mode(NegationMode::Window);
    let win_res = win_scorer.score_sentence(
        sent_text,
        Span::new(0, sent_text.len() as u32),
        &toks,
        IxRange::new(0, sent_words.len() as u32),
        None,
        None,
    );
    assert!(win_res.opinions[0].negated);
    assert!(win_res.score.compound < 0.0);

    let rel_tags = TagSet::new("ud", &["root", "neg", "obj", "amod", "punct"]);
    let deps = Dependencies {
        head: vec![
            Some(Ix::new(1)), // Not -> saying
            None,             // saying (root)
            Some(Ix::new(1)), // everything -> saying
            Some(Ix::new(2)), // good -> everything
            Some(Ix::new(1)), // . -> saying
        ],
        relation: vec![Tag(1), Tag(0), Tag(2), Tag(3), Tag(4)],
        relations: rel_tags,
    };
    let dep_scorer = LexiconSentiment::new().with_negation_mode(NegationMode::Dependency);
    let dep_res = dep_scorer.score_sentence(
        sent_text,
        Span::new(0, sent_text.len() as u32),
        &toks,
        IxRange::new(0, sent_words.len() as u32),
        Some(&deps),
        None,
    );
    assert!(dep_res.used_dependencies);
    assert!(!dep_res.opinions[0].negated);
    assert!(dep_res.score.compound > 0.0);
}

#[test]
fn opennlp_emoji_annotations_score_unicode_emoji_tokens() {
    let text = "Great update \u{1F600}";
    let words = ["Great", "update", "\u{1F600}"];
    let (sentences, tokens) = make_sentences_and_tokens(text, &words);

    let mut doc = Document::new(text);
    doc.add(&Key::<Sentences>::DEFAULT, sentences, Provenance::by("test")).unwrap();
    doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(LexiconSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = &docs[0];

    let opinions = doc.get::<Opinions>().unwrap();
    // Both "Great" (token 0) and grinning face U+1F600 (token 2) produce positive opinions.
    assert_eq!(opinions.len(), 2);
    assert_eq!(opinions.get(Ix::new(0)).tokens, IxRange::new(0, 1));
    assert_eq!(opinions.get(Ix::new(1)).tokens, IxRange::new(2, 3));
    assert!(opinions.get(Ix::new(1)).polarity > 0.0);
}

#[test]
fn config_hash_is_deterministic_and_pinned() {
    let base = LexiconSentiment::new().config_hash();
    let win = LexiconSentiment::new().with_negation_mode(NegationMode::Window).config_hash();
    let no_emoji = LexiconSentiment::new().with_opennlp_emoji(false).config_hash();
    assert_eq!(base, 0x0af1_182c_9c89_0240, "got 0x{base:016x}");
    assert_ne!(base, win);
    assert_ne!(base, no_emoji);
    assert_ne!(win, no_emoji);
}
