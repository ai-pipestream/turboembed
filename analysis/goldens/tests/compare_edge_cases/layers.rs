//! Aligned layers, the language layer, and golden layers that are empty or
//! inconsistent.

use super::*;

// ---------------------------------------------------------------------
// Aligned layers report the token they are for

#[test]
fn a_mismatch_in_an_aligned_layer_shows_the_anchor_token_on_both_sides() {
    let g = golden();
    let mut p = Parts::matching();
    p.prob.as_mut().unwrap()[4] = 0.5;
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(1e-9));
    says(
        &m.to_string(),
        &[
            "rust:   bytes 18..25 \"Zürich\", tag \"PROPN\", prob 3fe0000000000000 (0.5)",
            "golden: bytes 18..25 \"Zürich\", tag \"PROPN\", prob 3fe9fe5c91d14e3c (0.8123)",
        ],
    );
    // The confidence of a sentence shows the sentence.
    let mut p = Parts::matching();
    p.sentence_conf[1] = 0.5;
    let m = fail::<Confidence<Sentences>>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 1, what: "value", .. }), "{m:?}");
    says(
        &m.to_string(),
        &[
            "turbo:confidence<turbo:sentences> item 1: the value differs",
            "rust:   bytes 27..37 \"It rained!\", value 3fe0000000000000 (0.5)",
            "golden: bytes 27..37 \"It rained!\", value 3fee666666666666 (0.95)",
        ],
    );
    // The lemma of the first token, which has a period in it.
    let mut p = Parts::matching();
    p.lemmas[0] = "Dr";
    let m = fail::<Lemmas>(&p, &g, Tolerance::exact());
    says(&m.to_string(), &["turbo:lemmas item 0", "bytes 0..3 \"Dr.\", lemma \"Dr\"", "lemma \"Dr.\""]);
    // The last one, at the end of the text.
    let mut p = Parts::matching();
    p.lemmas[8] = "?";
    let m = fail::<Lemmas>(&p, &g, Tolerance::exact());
    says(&m.to_string(), &["item 8", "bytes 36..37 \"!\""]);
}

#[test]
fn an_aligned_report_takes_its_token_from_the_rust_side_of_the_document() {
    // The Rust tokens differ from the golden's, but the tags are compared
    // by index: the tag differs, and each side shows its own token.
    let g = golden();
    let mut p = Parts::matching();
    p.tokens[4] = (18, 24);
    p.tags[4] = "NOUN";
    let m = fail::<PosTags>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 4, what: "tag", .. }), "{m:?}");
    says(&m.to_string(), &["rust:   bytes 18..24 \"Züric\"", "golden: bytes 18..25 \"Zürich\""]);
}

/// The fixture's token layers cut to `n` tokens (and no entities, which
/// refer to tokens).
fn cut_tokens(p: &mut Parts, n: usize) {
    p.tokens.truncate(n);
    p.token_conf.truncate(n);
    p.tags.truncate(n);
    p.prob.as_mut().unwrap().truncate(n);
    p.lemmas.truncate(n);
    p.entities.clear();
}

#[test]
fn counts_report_the_first_item_past_the_shorter_side() {
    let g = golden();
    // Rust has one token too many.
    let mut p = Parts::matching();
    p.tokens.push((37, 37));
    p.token_conf.push(0.5);
    p.tags.push("PUNCT");
    p.prob.as_mut().unwrap().push(0.5);
    p.lemmas.push("");
    let m = fail::<Tokens>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 10, golden: 9, .. }), "{m:?}");
    says(
        &m.to_string(),
        &["turbo:tokens: 10 items in Rust, 9 in the golden", "Rust item 9:", "Rust: bytes 37..37 \"\""],
    );
    // The aligned layers report their counts the same way.
    let m = fail::<PosTags>(&p, &g, Tolerance::exact());
    says(
        &m.to_string(),
        &["turbo:pos: 10 items in Rust, 9 in the golden", "Rust item 9:", "bytes 37..37 \"\", tag \"PUNCT\""],
    );
    // Rust has fewer; the first golden row past it is shown.
    let mut p = Parts::matching();
    cut_tokens(&mut p, 3);
    let m = fail::<Tokens>(&p, &g, Tolerance::exact());
    says(&m.to_string(), &["3 items in Rust, 9 in the golden", "golden item 3:", "golden: bytes 15..17 \"to\""]);
    let m = fail::<Lemmas>(&p, &g, Tolerance::exact());
    says(
        &m.to_string(),
        &["turbo:lemmas: 3 items in Rust, 9 in the golden", "golden: bytes 15..17 \"to\", lemma \"to\""],
    );
    // Rust has none at all.
    let mut p = Parts::matching();
    p.sentences.clear();
    p.sentence_conf.clear();
    let m = fail::<Sentences>(&p, &g, Tolerance::exact());
    says(&m.to_string(), &["0 items in Rust, 2 in the golden", "golden item 0:", "golden: bytes 0..26"]);
    // A difference before the end wins over a count difference.
    let mut p = Parts::matching();
    cut_tokens(&mut p, 3);
    p.tokens[1] = (4, 8);
    assert!(matches!(fail::<Tokens>(&p, &g, Tolerance::exact()), Mismatch::Item { index: 1, .. }));
}

#[test]
fn a_golden_span_outside_character_boundaries_is_reported_without_a_panic() {
    // The golden's token 4 cuts the two bytes of "ü" (bytes 19..21) in the middle.
    let g = golden_with("18\t25\n25\t26", "18\t20\n25\t26");
    let m = fail::<Tokens>(&Parts::matching(), &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 4, what: "span", .. }), "{m:?}");
    says(
        &m.to_string(),
        &["rust:   bytes 18..25 \"Zürich\"", "golden: bytes 18..20 (outside the text or not on character boundaries)"],
    );
}

// ---------------------------------------------------------------------
// Language

fn language_golden(rows: &[&str]) -> Golden {
    let text = fixture_text();
    let start = text.find("layer\tturbo:language").unwrap();
    let mut out = text[..start].to_string();
    out.push_str(&format!("layer\tturbo:language\tdocument\t{}\tcode:str\tconfidence:f64\n", rows.len()));
    for r in rows {
        out.push_str(r);
        out.push('\n');
    }
    out.push_str("end\tturbo:language\n");
    Golden::parse(&out).unwrap()
}

#[test]
fn language_compares_every_row_in_rank_order() {
    let mut p = Parts::matching();
    let eng = "eng\t3fedccac6fbfc4c9 0.9312345678901234";
    let deu = "deu\t3fa5182a9930be0e 0.0412";
    let nld = "nld\t3f889374bc6a7efa 0.012";
    let fra = "fra\t3ee4f8b588e368f1 1.0E-5";
    assert_eq!(golden().layer("turbo:language").unwrap().len(), 4);
    assert_eq!(matches_golden::<Language>(&p.doc(), &golden(), Tolerance::exact()), Ok(()));
    assert_eq!(
        matches_golden::<Language>(&p.doc(), &language_golden(&[eng, deu, nld, fra]), Tolerance::exact()),
        Ok(())
    );
    // Two rows swapped: the ranking differs at the first swapped row.
    let g = language_golden(&[eng, nld, deu, fra]);
    let m = fail::<Language>(&p, &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 1, what: "code", .. }), "{m:?}");
    says(&m.to_string(), &["turbo:language item 1", "rust:   code \"deu\"", "golden: code \"nld\""]);
    // A later row's confidence is compared, with the tolerance.
    p.language[3].1 += 1e-12;
    assert_eq!(matches_golden::<Language>(&p.doc(), &golden(), Tolerance::abs(1e-9)), Ok(()));
    let m = fail::<Language>(&p, &golden(), Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 3, what: "confidence", .. }), "{m:?}");
    // A golden with fewer rows: a count difference naming the first extra Rust row.
    let m = fail::<Language>(&Parts::matching(), &language_golden(&[eng, deu]), Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 4, golden: 2, .. }), "{m:?}");
    says(&m.to_string(), &["turbo:language: 4 items in Rust, 2 in the golden", "Rust item 2:", "Rust: code \"nld\""]);
    // A golden with no rows.
    let m = fail::<Language>(&Parts::matching(), &language_golden(&[]), Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 4, golden: 0, .. }), "{m:?}");
}

#[test]
fn language_with_unusual_values() {
    let eng = |bits: &str| format!("eng\t{bits} x");
    let mut p = Parts::matching();
    p.language.truncate(1);
    // A NaN confidence matches only the same bits at exact, and any NaN at a tolerance.
    p.language[0].1 = f64::from_bits(0x7ff8000000000000);
    let g = language_golden(&[&eng("7ff8000000000000")]);
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::exact()), Ok(()));
    let g = language_golden(&[&eng("fff8000000000000")]);
    assert!(matches!(fail::<Language>(&p, &g, Tolerance::exact()), Mismatch::Item { what: "confidence", .. }));
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::abs(1e-9)), Ok(()));
    // Infinities: the same sign matches, the other does not, and a finite value is not near one.
    p.language[0].1 = f64::INFINITY;
    let g = language_golden(&[&eng("7ff0000000000000")]);
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::abs(1e-9)), Ok(()));
    let g = language_golden(&[&eng("fff0000000000000")]);
    assert!(matches!(fail::<Language>(&p, &g, Tolerance::abs(f64::MAX)), Mismatch::Item { what: "confidence", .. }));
    // -0.0 and 0.0.
    p.language[0].1 = 0.0;
    let g = language_golden(&[&eng("8000000000000000")]);
    assert!(matches!(fail::<Language>(&p, &g, Tolerance::exact()), Mismatch::Item { what: "confidence", .. }));
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::abs(0.0)), Ok(()));
    // An empty code is a code.
    p.language[0] = ("", f64::from_bits(0x3fedccac6fbfc4c9));
    let g = language_golden(&["\t3fedccac6fbfc4c9 x"]);
    assert_eq!(matches_golden::<Language>(&p.doc(), &g, Tolerance::exact()), Ok(()));
}

// ---------------------------------------------------------------------
// Golden layers that are empty or inconsistent

#[test]
fn an_empty_golden_layer_is_a_count_difference_not_a_pass() {
    let text = fixture_text();
    let sentences = "layer\tturbo:sentences\ttext\t2\tstart:u32\tend:u32\n0\t26\n27\t37\nend\tturbo:sentences\n";
    assert!(text.contains(sentences));
    // The confidence layer is aligned to it, so empty both.
    let conf = "layer\tturbo:confidence<turbo:sentences>\taligned:turbo:sentences\t2\tvalue:f64\n3fef9add3c1be9b8 0.9876543210987654\n3fee666666666666 0.95\nend\tturbo:confidence<turbo:sentences>\n";
    assert!(text.contains(conf));
    let emptied = text
        .replace(sentences, "layer\tturbo:sentences\ttext\t0\tstart:u32\tend:u32\nend\tturbo:sentences\n")
        .replace(
            conf,
            "layer\tturbo:confidence<turbo:sentences>\taligned:turbo:sentences\t0\tvalue:f64\nend\tturbo:confidence<turbo:sentences>\n",
        );
    let g = Golden::parse(&emptied).unwrap();
    let m = fail::<Sentences>(&Parts::matching(), &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 2, golden: 0, .. }), "{m:?}");
    says(
        &m.to_string(),
        &["2 items in Rust, 0 in the golden", "Rust item 0:", "Rust: bytes 0..26 \"Dr. Smith went to Zürich.\""],
    );
    let m = fail::<Confidence<Sentences>>(&Parts::matching(), &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 2, golden: 0, .. }), "{m:?}");
    // A Rust layer that is empty too matches the empty golden.
    let mut p = Parts::matching();
    p.sentences.clear();
    p.sentence_conf.clear();
    assert_eq!(matches_golden::<Sentences>(&p.doc(), &g, Tolerance::exact()), Ok(()));
    assert_eq!(matches_golden::<Confidence<Sentences>>(&p.doc(), &g, Tolerance::exact()), Ok(()));
}

#[test]
fn a_golden_whose_columns_have_different_lengths_is_a_columns_mismatch() {
    let mut g = golden();
    let pos = g.layers.iter_mut().find(|l| l.id == "turbo:pos").unwrap();
    pos.columns[1].data = ColumnData::F64(vec![0.5; 8]);
    let m = fail::<PosTags>(&Parts::matching(), &g, Tolerance::exact());
    assert!(matches!(m, Mismatch::Columns { .. }), "{m:?}");
    says(
        &m.to_string(),
        &["turbo:pos: the golden layer has the columns [tag:str, prob:f64], expected [tag:str, prob:f64]"],
    );
}

#[test]
fn a_golden_layer_with_extra_or_missing_columns() {
    let mut g = golden();
    let toks = g.layers.iter_mut().find(|l| l.id == "turbo:tokens").unwrap();
    toks.columns.push(Column { name: "extra".into(), data: ColumnData::U32(vec![0; 9]) });
    let m = fail::<Tokens>(&Parts::matching(), &g, Tolerance::exact());
    says(&m.to_string(), &["[start:u32, end:u32, extra:u32], expected [start:u32, end:u32]"]);
    let mut g = golden();
    let toks = g.layers.iter_mut().find(|l| l.id == "turbo:tokens").unwrap();
    toks.columns.pop();
    let m = fail::<Tokens>(&Parts::matching(), &g, Tolerance::exact());
    says(&m.to_string(), &["[start:u32], expected [start:u32, end:u32]"]);
    // Swapped types.
    let mut g = golden();
    let toks = g.layers.iter_mut().find(|l| l.id == "turbo:tokens").unwrap();
    toks.columns[0].data = ColumnData::F64(vec![0.0; 9]);
    let m = fail::<Tokens>(&Parts::matching(), &g, Tolerance::exact());
    says(&m.to_string(), &["[start:f64, end:u32]"]);
}

const POS0: &str = "3fef141205bc01a3 0.9712";

#[test]
fn a_golden_with_a_nan_value_compares_by_bits_unless_given_a_tolerance() {
    let g = golden_with(POS0, "7ff8000000000000 NaN");
    let mut p = Parts::matching();
    p.prob.as_mut().unwrap()[0] = f64::NAN;
    assert_eq!(f64::NAN.to_bits(), 0x7ff8000000000000);
    assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, Tolerance::exact()), Ok(()));
    // A NaN with another sign or payload differs at exact bits.
    for bits in [0xfff8000000000000u64, 0x7ff8000000000001, 0x7ff0000000000001] {
        p.prob.as_mut().unwrap()[0] = f64::from_bits(bits);
        let m = fail::<PosTags>(&p, &g, Tolerance::exact());
        assert!(matches!(m, Mismatch::Item { index: 0, what: "prob", .. }), "{bits:x}: {m:?}");
        assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, Tolerance::abs(1e-9)), Ok(()), "{bits:x}");
    }
    // A number is not a NaN, within any distance, and the report shows the golden's NaN.
    p.prob.as_mut().unwrap()[0] = 0.9712;
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(f64::MAX));
    assert!(matches!(m, Mismatch::Item { index: 0, what: "prob", .. }), "{m:?}");
    says(&m.to_string(), &["golden: bytes 0..3 \"Dr.\", tag \"PROPN\", prob 7ff8000000000000 (NaN)"]);
    // A golden number against a Rust NaN.
    let g = golden();
    p.prob.as_mut().unwrap()[0] = f64::NAN;
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(f64::MAX));
    assert!(matches!(m, Mismatch::Item { index: 0, what: "prob", .. }), "{m:?}");
}

#[test]
fn a_golden_with_an_infinite_value() {
    let g = golden_with(POS0, "7ff0000000000000 Infinity");
    let mut p = Parts::matching();
    p.prob.as_mut().unwrap()[0] = f64::INFINITY;
    assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, Tolerance::exact()), Ok(()));
    assert_eq!(matches_golden::<PosTags>(&p.doc(), &g, Tolerance::abs(1e-9)), Ok(()));
    p.prob.as_mut().unwrap()[0] = f64::MAX;
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(f64::MAX));
    assert!(matches!(m, Mismatch::Item { index: 0, .. }), "{m:?}");
    p.prob.as_mut().unwrap()[0] = f64::NEG_INFINITY;
    let m = fail::<PosTags>(&p, &g, Tolerance::abs(1e300));
    assert!(matches!(m, Mismatch::Item { index: 0, .. }), "{m:?}");
}
