//! The spelling-correcting normalizer: OpenNLP's
//! `SpellCheckingCharSequenceNormalizerTest` and
//! `SpellCheckingUnicodeCaseTest`, parity with OpenNLP's output over
//! generated phrases, the alignment of every output both ways, and the
//! additions (placeholder skipping, proper nouns, the corrections layer).

mod folds_support;

use std::fs;
use std::sync::Arc;

use folds_support::check_alignment;
use turbo_analysis::{Annotator, Document, Layer, Refs, View};
use turbo_normalize::spellcheck::{
    FrequencyDictionary, SpellCorrect, SpellCorrectMode, SymSpell, SymSpellConfig, SymSpellModel,
};
use turbo_normalize::{Chain, Normalizer, Shrink, SocialMedia, Urls, normalize};
use turbo_normalize_layers::{Corrections, Placeholders};
use turbo_unicode::props::white_space;

fn data(name: &str) -> String {
    format!("{}/tests/data/spellcheck/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// OpenNLP's `TinyDictionary`.
fn tiny() -> Arc<SymSpell> {
    let mut engine = SymSpell::new(SymSpellConfig::builder().max_dictionary_edit_distance(2).build().unwrap());
    FrequencyDictionary::load_unigrams(
        &mut engine,
        &fs::read_to_string(data("frequency_dictionary_tiny.txt")).unwrap(),
    )
    .unwrap();
    FrequencyDictionary::load_bigrams(
        &mut engine,
        &fs::read_to_string(data("frequency_bigramdictionary_tiny.txt")).unwrap(),
    )
    .unwrap();
    Arc::new(engine)
}

fn one_word(word: &str) -> Arc<SymSpell> {
    let mut engine = SymSpell::default();
    engine.add(word, 1000);
    Arc::new(engine)
}

/// An edit this normalizer may make: whitespace to "" or one space (the
/// compound joins), or a stretch with text to another stretch with text.
fn edit_ok(parent: &str, child: &str) -> bool {
    let blank = |s: &str| s.chars().all(white_space);
    if blank(parent) {
        return child.is_empty() || child == " ";
    }
    !child.is_empty() || parent.chars().all(white_space)
}

/// Normalizes `src`, checks the alignment both ways, runs the annotator
/// form and checks that it writes the same text and a corrections layer
/// whose every row maps onto its suggestion in the output.
fn run(n: &SpellCorrect, src: &str) -> String {
    let (out, a) = normalize(n, src);
    // An empty dictionary word can make a correction empty; the check
    // accepts that only for such words, and the tests here have none.
    check_alignment(src, &out, &a, edit_ok);
    let (view, corrections) = annotate(n, src, None);
    assert_eq!(view.text(), out, "annotator and normalizer disagree on {src:?}");
    check_corrections(src, &view, &corrections);
    out
}

fn annotate(n: &SpellCorrect, src: &str, placeholders: Option<&Placeholders>) -> (View, Corrections) {
    let doc = Document::new(src);
    let view = doc.get::<View>().expect("the source view");
    let input: Refs<'_, SpellCorrect> = (view, placeholders);
    n.annotate(input).expect("annotate")
}

fn check_corrections(src: &str, view: &View, c: &Corrections) {
    c.check().expect("corrections layer invariants");
    let a = view.alignment();
    for (ix, s) in c.spans.iter() {
        let i = ix.get();
        let child = a.span_to_child(s);
        let landed = &view.text()[child.range()];
        let suggestion = c.suggestion.get(i);
        assert!(landed.contains(suggestion), "{:?} became {suggestion:?}, landed on {landed:?}", &src[s.range()]);
        assert_ne!(&src[s.range()], suggestion);
    }
}

fn spell(engine: &Arc<SymSpell>) -> SpellCorrect {
    SpellCorrect::new(engine.clone())
}

// ---------------------------------------------------------------------
// SpellCheckingCharSequenceNormalizerTest.
// ---------------------------------------------------------------------

#[test]
fn per_token_corrects_typos() {
    let t = tiny();
    assert_eq!(run(&spell(&t).min_token_length(3), "teh quikc broen fxo"), "the quick brown fox");
    // With the default minimum of 4 only the longer typos are corrected.
    assert_eq!(run(&spell(&t), "quikc broen wrold"), "quick brown world");
    assert_eq!(run(&spell(&t), "teh quikc"), "teh quick");
}

#[test]
fn per_token_leaves_known_words_and_keeps_whitespace() {
    let t = tiny();
    assert_eq!(run(&spell(&t), "the quick brown fox"), "the quick brown fox");
    assert_eq!(run(&spell(&t), "  quikc   wrold  "), "  quick   world  ");
}

#[test]
fn per_token_keeps_punctuation_and_casing() {
    let t = tiny();
    assert_eq!(run(&spell(&t), "\"wrold\","), "\"world\",");
    assert_eq!(run(&spell(&t), "(quikc)"), "(quick)");
    assert_eq!(run(&spell(&t), "Wrold"), "World");
    assert_eq!(run(&spell(&t), "WROLD"), "WORLD");
    assert_eq!(run(&spell(&t), "wrold"), "world");
}

#[test]
fn per_token_cases_a_supplementary_initial() {
    let e = one_word("\u{10428}word");
    assert_eq!(run(&spell(&e), "\u{10428}wrod"), "\u{10428}word");
    assert_eq!(run(&spell(&e), "\u{10400}wrod"), "\u{10400}word");
}

#[test]
fn min_token_length_skips_short_tokens() {
    let t = tiny();
    assert_eq!(run(&spell(&t).min_token_length(1), "adn"), "and");
    assert_eq!(run(&spell(&t).min_token_length(5), "adn"), "adn");
}

#[test]
fn numbers_and_urls_are_skipped() {
    let t = tiny();
    for s in ["12345", "3.14", "http://example.com/teh", "user@example.com"] {
        assert_eq!(run(&spell(&t), s), s);
    }
}

#[test]
fn punctuation_peels_by_unicode_letter_and_number_categories() {
    let t = tiny();
    assert_eq!(run(&spell(&t), "\u{AB}quikc\u{BB}"), "\u{AB}quick\u{BB}");
    assert_eq!(run(&spell(&t), "quikc\u{301}"), "quick\u{301}");
    assert_eq!(run(&spell(&t), "!!!"), "!!!");
}

#[test]
fn number_and_url_guards_at_their_boundaries() {
    let t = tiny();
    for s in ["+3,14%", "-42%", "1.2.3", "quikc.com", "www.quikc.com/broen"] {
        assert_eq!(run(&spell(&t), s), s);
    }
}

#[test]
fn compound_mode_repairs_merges_and_phrases() {
    let t = tiny();
    let c = spell(&t).mode(SpellCorrectMode::Compound);
    assert_eq!(run(&c, "helloworld"), "hello world");
    assert_eq!(run(&c, "teh quikc broen fxo"), "the quick brown fox");
}

#[test]
fn composes_in_a_chain() {
    let t = tiny();
    let chain = Chain::new().then(Urls).then(Shrink).then(spell(&t));
    let (out, a) = normalize(&chain, "quikc    broen http://example.com fox");
    assert_eq!(out, "quick brown fox");
    check_alignment("quikc    broen http://example.com fox", &out, &a, |_, _| true);
}

#[test]
fn empty_input_and_a_clamped_distance() {
    let t = tiny();
    assert_eq!(run(&spell(&t), ""), "");
    assert_eq!(run(&spell(&t).mode(SpellCorrectMode::Compound), ""), "");
    assert_eq!(t.max_edit_distance(), 2);
    assert_eq!(run(&spell(&t).min_token_length(3).max_edit_distance(5), "teh quikc broen fxo"), "the quick brown fox");
}

#[test]
fn per_token_boundaries_are_unicode_white_space() {
    let t = tiny();
    let n = spell(&t).min_token_length(3);
    for sep in ['\u{A0}', '\u{202F}', '\u{85}', '\u{2028}'] {
        assert_eq!(run(&n, &format!("teh{sep}fxo")), format!("the{sep}fox"), "{sep:?}");
    }
    // U+001C is not White_Space: the joined token has no correction.
    assert_eq!(run(&n, "teh\u{1C}fxo"), "teh\u{1C}fxo");
}

// ---------------------------------------------------------------------
// SpellCheckingUnicodeCaseTest.
// ---------------------------------------------------------------------

#[test]
fn casing_of_corrections() {
    for (word, input, expected) in [
        ("world", "wrold", "world"),
        ("world", "Wrold", "World"),
        ("world", "WROLD", "WORLD"),
        ("𐐨𐐩𐐪𐐫", "𐐨𐐩𐐪𐐭", "𐐨𐐩𐐪𐐫"),
        ("𐐨𐐩𐐪𐐫", "𐐀𐐩𐐪𐐭", "𐐀𐐩𐐪𐐫"),
        ("𐐨𐐩𐐪𐐫", "𐐀𐐁𐐂𐐅", "𐐀𐐁𐐂𐐃"),
        ("𐐨world", "𐐨WROLD", "𐐨world"),
        ("𐐨world", "𐐀WROLD", "𐐀WORLD"),
        ("a𐐩cda", "A𐐩CDE", "A𐐩cda"),
        ("a𐐩cda", "A𐐁CDE", "A𐐁CDA"),
        ("word𐐨", "WRLD𐐨", "Word𐐨"),
        ("word𐐨", "WRLD𐐀", "WORD𐐀"),
        ("𠮷𠮸𠮹𠮻", "𠮷𠮸𠮹𠮺", "𠮷𠮸𠮹𠮻"),
        ("ab", "A", "Ab"),
        ("ab", "𐐀", "Ab"),
        ("ab", "𐐨", "ab"),
    ] {
        let e = one_word(word);
        let mut lower = String::new();
        turbo_unicode::java::to_lowercase(input, &mut lower);
        assert_eq!(e.lookup_top(&lower)[0].term, word);
        let n = spell(&e).min_token_length(1);
        assert_eq!(run(&n, input), expected, "{input}");
        assert_eq!(run(&n, &format!("\"{input}!\"")), format!("\"{expected}!\""));
        assert_eq!(run(&n, &format!("\t{input}\u{A0}")), format!("\t{expected}\u{A0}"));
    }
}

#[test]
fn known_words_and_guarded_tokens_stay() {
    for (word, input) in [
        ("𐐨𐐩𐐪𐐫", "𐐨𐐩𐐪𐐫"),
        ("𐐨𐐩𐐪𐐫", "𐐀𐐁𐐂𐐃"),
        ("𐐨𐐩𐐪𐐫", "𐐀𐐩𐐂𐐫"),
        ("word𐐨", "WORD𐐨"),
        ("12346", "12345"),
        ("😀😀😀😁", "😀😀😀😀"),
        ("www.world.com", "www.wrold.com"),
    ] {
        assert_eq!(run(&spell(&one_word(word)).min_token_length(1), input), input);
    }
    assert_eq!(run(&spell(&one_word("\u{10428}word")), "\u{10428}WROLD"), "\u{10428}word");
}

// ---------------------------------------------------------------------
// Parity with OpenNLP over generated phrases (expected.txt).
// ---------------------------------------------------------------------

fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("\\u{") {
        out.push_str(&rest[..i]);
        let end = rest[i..].find('}').unwrap() + i;
        out.push(char::from_u32(u32::from_str_radix(&rest[i + 3..end], 16).unwrap()).unwrap());
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

#[test]
fn output_matches_opennlp() {
    let model = SymSpellModel::read(&fs::read(data("model-tiny.bin")).unwrap()).unwrap();
    let engine = Arc::new(model.build());
    let base = spell(&engine);
    let configs = [
        ("default", base.clone()),
        ("min1", base.clone().min_token_length(1)),
        ("max1", base.clone().max_edit_distance(1)),
        ("noguards", base.clone().min_token_length(1).skip_numbers(false).skip_urls(false)),
        ("compound", base.clone().mode(SpellCorrectMode::Compound)),
        ("compound1", base.clone().mode(SpellCorrectMode::Compound).max_edit_distance(1)),
    ];
    let text = fs::read_to_string(data("expected.txt")).unwrap();
    let mut checked = 0;
    for line in text.lines().filter(|l| l.starts_with("normalize\t")) {
        let f: Vec<&str> = line.split('\t').collect();
        let n = &configs.iter().find(|(name, _)| *name == f[1]).unwrap().1;
        let input = unescape(f[2]);
        assert_eq!(run(n, &input), unescape(f[3]), "{} {input:?}", f[1]);
        checked += 1;
    }
    assert_eq!(checked, 6 * 703);
}

// ---------------------------------------------------------------------
// The alignment of compound mode.
// ---------------------------------------------------------------------

#[test]
fn compound_alignment_maps_merges_splits_and_whitespace() {
    let t = tiny();
    let c = spell(&t).mode(SpellCorrectMode::Compound);
    let src = "  hel lo  spellchecker teh\t";
    let (out, a) = normalize(&c, src);
    assert_eq!(out, "hello spell checker the");
    check_alignment(src, &out, &a, edit_ok);
    let (view, corrections) = annotate(&c, src, None);
    assert_eq!(view.text(), out);
    let rows: Vec<(&str, &str, u8)> = corrections
        .spans
        .iter()
        .map(|(ix, s)| (&src[s.range()], corrections.suggestion.get(ix.get()), corrections.edit_distance[ix.get()]))
        .collect();
    // The merge covers both tokens and the space between them; the split
    // keeps its one token.
    assert_eq!(rows, [("hel lo", "hello", 1), ("spellchecker", "spell checker", 1), ("teh", "the", 1)]);
    // Each output word maps back to its input and at most the whitespace
    // deleted next to it: a space of each run is kept, so the edits of
    // neighbouring words do not touch.
    let back = |w: &str| {
        let i = out.find(w).unwrap();
        let s = a.span_to_parent(turbo_analysis::Span::new(i as u32, (i + w.len()) as u32));
        &src[s.range()]
    };
    assert_eq!(back("hello").trim_start(), "hel lo");
    assert_eq!(back("spell checker").trim_matches(white_space), "spellchecker");
    assert_eq!(back("the").trim_matches(white_space), "teh");
    // A run with no space has to become one. The edits of the two words
    // and of the run touch, but each stays its own edit.
    let src = "teh\u{A0}wrold";
    let (out, a) = normalize(&c, src);
    assert_eq!(out, "the world");
    let edits: Vec<(&str, &str)> = a.edits().iter().map(|e| (&src[e.parent.range()], &out[e.child.range()])).collect();
    assert_eq!(edits, [("teh", "the"), ("\u{A0}", " "), ("wrold", "world")]);
    let (_, corrections) = annotate(&c, src, None);
    assert_eq!(corrections.spans.len(), 2);
}

#[test]
fn compound_keeps_a_blank_text_and_one_already_correct() {
    let t = tiny();
    let c = spell(&t).mode(SpellCorrectMode::Compound);
    // Blank as Java's isBlank sees it: left as it is.
    for s in ["   ", "\t\n", "\u{1C}\u{1F}"] {
        let (out, a) = normalize(&c, s);
        assert_eq!(out, s);
        assert!(a.edits().is_empty());
    }
    // Only White_Space that Java does not count as blank: the correction
    // is empty, so the text stays.
    assert_eq!(run(&c, "\u{A0}\u{85}"), "\u{A0}\u{85}");
    // Correct words with single spaces make no edit at all.
    let (_, a) = normalize(&c, "the quick brown fox");
    assert!(a.edits().is_empty());
    assert_eq!(run(&c, " the  quick\tbrown fox\n"), "the quick brown fox");
}

// ---------------------------------------------------------------------
// Differences from OpenNLP.
// ---------------------------------------------------------------------

#[test]
fn token_length_counts_code_points() {
    // "𐐨wr" is three code points and four UTF-16 code units. OpenNLP
    // compares UTF-16 units with the minimum, so it would correct this
    // token at the default minimum of 4; here it is too short.
    let e = one_word("\u{10428}wro");
    assert_eq!(run(&spell(&e), "\u{10428}wr"), "\u{10428}wr");
    assert_eq!(run(&spell(&e).min_token_length(3), "\u{10428}wr"), "\u{10428}wro");
}

#[test]
fn compound_splits_only_between_code_points() {
    // OpenNLP splits a token at every UTF-16 index, so it also looks up
    // the halves of a supplementary character cut between its surrogates,
    // and it counts "😀" as two characters, so it tries to split even a
    // token of that one character. Here the split points are the
    // boundaries between code points and "😀" is one character.
    let mut e = SymSpell::default();
    e.add("ab", 100);
    e.add("\u{1F600}", 100);
    e.add_bigram("ab", "\u{1F600}", 1000);
    let r = e.lookup_compound("ab\u{1F600}", 2).unwrap();
    assert_eq!(r.term, "ab \u{1F600}");
    assert_eq!(r.parts.len(), 1);
    assert_eq!(r.parts[0].source, 0..6);
    let r = e.lookup_compound("\u{1F601}", 2).unwrap();
    assert_eq!((r.term.as_str(), r.edit_distance), ("\u{1F600}", 1));
}

#[test]
fn proper_nouns_two_edits_from_a_word_can_be_kept() {
    let t = tiny();
    let opennlp = spell(&t);
    let keep = spell(&t).keep_unknown_proper_nouns(true);
    // Two edits from "quick" and "brown": OpenNLP rewrites them.
    for (name, rewritten) in [("Quinn", "Quick"), ("Brian", "Brown")] {
        assert_eq!(run(&opennlp, name), rewritten);
        assert_eq!(run(&keep, name), name);
    }
    // One edit away, or not capitalized: corrected as before.
    assert_eq!(run(&keep, "Wrold"), "World");
    assert_eq!(run(&keep, "quinn"), "quick");
    assert_eq!(run(&keep, "QUINN"), "QUINN");
    assert_eq!(run(&opennlp, "QUINN"), "QUICK");
}

#[test]
fn placeholders_replace_the_url_guess() {
    let t = tiny();
    let src = "#wrold and #helloworld at wrold@exampel.co quikc";
    // The social media annotator marks hashtags and mentions, not
    // the e-mail address.
    let doc = Document::new(src);
    let view = doc.get::<View>().unwrap();
    let (_, placeholders) = SocialMedia.annotate((view,)).unwrap();
    // OpenNLP corrects the core of a hashtag.
    let n = spell(&t);
    assert_eq!(run(&n, src), "#world and #helloworld at wrold@exampel.co quick");
    // With the placeholders the hashtag stays, and the address, which no
    // placeholder covers, is checked like any token (its core has no
    // correction within two edits).
    let n = spell(&t).use_placeholders(true);
    let (out, corrections) = annotate(&n, src, Some(&placeholders));
    assert_eq!(out.text(), "#wrold and #helloworld at wrold@exampel.co quick");
    check_alignment(src, out.text(), out.alignment(), edit_ok);
    check_corrections(src, &out, &corrections);
    // Without the option, or without the layer, OpenNLP's guards apply.
    let (out, _) = annotate(&spell(&t), src, Some(&placeholders));
    assert_eq!(out.text(), "#world and #helloworld at wrold@exampel.co quick");
    let (out, _) = annotate(&n, src, None);
    assert_eq!(out.text(), "#world and #helloworld at wrold@exampel.co quick");
}

#[test]
fn placeholders_split_compound_phrases() {
    let t = tiny();
    let src = " teh #helloworld  quikc broen ";
    let doc = Document::new(src);
    let view = doc.get::<View>().unwrap();
    let (_, placeholders) = SocialMedia.annotate((view,)).unwrap();
    let c = spell(&t).mode(SpellCorrectMode::Compound).use_placeholders(true);
    let (out, corrections) = annotate(&c, src, Some(&placeholders));
    assert_eq!(out.text(), "the #helloworld quick brown");
    check_alignment(src, out.text(), out.alignment(), edit_ok);
    check_corrections(src, &out, &corrections);
    // OpenNLP splits the hashtag's word.
    let (plain, _) = annotate(&spell(&t).mode(SpellCorrectMode::Compound), src, Some(&placeholders));
    assert_ne!(plain.text(), out.text());
}

#[test]
fn placeholders_of_another_view_are_refused() {
    let t = tiny();
    let mut p = Placeholders::new(turbo_analysis::ViewId(3));
    p.spans.push(turbo_analysis::Span::new(0, 1));
    p.kind.push(turbo_normalize_layers::PlaceholderKind::Url);
    let doc = Document::new("x");
    let view = doc.get::<View>().unwrap();
    assert!(spell(&t).use_placeholders(true).annotate((view, Some(&p))).is_err());
}

#[test]
fn the_corrections_layer_records_each_change() {
    let t = tiny();
    let src = "Teh QUIKC broen (fxo), the dog";
    let n = spell(&t).min_token_length(3);
    let (view, c) = annotate(&n, src, None);
    assert_eq!(view.text(), "The QUICK brown (fox), the dog");
    let rows: Vec<(&str, &str, u8)> = c
        .spans
        .iter()
        .map(|(ix, s)| (&src[s.range()], c.suggestion.get(ix.get()), c.edit_distance[ix.get()]))
        .collect();
    assert_eq!(rows, [("Teh", "The", 1), ("QUIKC", "QUICK", 1), ("broen", "brown", 1), ("fxo", "fox", 1)]);
    assert_eq!(<Corrections as Layer>::ID, "turbo:corrections");
    // Undoing every correction gives the source back.
    let mut undone = view.text().to_owned();
    for (_, s) in c.spans.iter().collect::<Vec<_>>().into_iter().rev() {
        let child = view.alignment().span_to_child(s);
        undone.replace_range(child.range(), &src[s.range()]);
    }
    assert_eq!(undone, src);
}

#[test]
fn the_configuration_hash_covers_every_option_and_the_dictionary() {
    let t = tiny();
    let base = spell(&t);
    let variants = [
        base.clone().mode(SpellCorrectMode::Compound),
        base.clone().min_token_length(3),
        base.clone().max_edit_distance(1),
        base.clone().skip_numbers(false),
        base.clone().skip_urls(false),
        base.clone().keep_unknown_proper_nouns(true),
        base.clone().use_placeholders(true),
        spell(&one_word("other")),
    ];
    let h = |n: &SpellCorrect| <SpellCorrect as Normalizer>::config_hash(n);
    let mut seen = vec![h(&base)];
    for v in &variants {
        assert!(!seen.contains(&h(v)), "{v:?}");
        seen.push(h(v));
    }
    assert_eq!(h(&base), h(&spell(&tiny())));
    assert_eq!(h(&base), <SpellCorrect as Annotator>::config_hash(&base));
    assert_eq!(<SpellCorrect as Normalizer>::ID, "turbo:normalize:spell-correct");
}

#[test]
#[should_panic(expected = "minTokenLength must be >= 1")]
fn a_zero_minimum_length_is_refused() {
    let _ = spell(&tiny()).min_token_length(0);
}
