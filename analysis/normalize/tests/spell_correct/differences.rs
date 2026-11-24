//! Where the spelling normalizer differs from OpenNLP.

use super::*;

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
