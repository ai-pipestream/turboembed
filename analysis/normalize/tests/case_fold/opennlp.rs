use super::*;

#[test]
fn opennlp_lowercase_expectations() {
    // CaseFoldCharSequenceNormalizerTest and
    // UnicodeCharSequenceNormalizerTest (Locale.ROOT).
    let lower = CaseFold::lowercase();
    assert_eq!(run_fold(lower, "HeLLo WORLD", lower_matches), "hello world");
    assert_eq!(run_fold(lower, "I", lower_matches), "i");
    assert_eq!(run_fold(lower, "ABC", lower_matches), "abc");
    assert_eq!(run_fold(lower, "CAF\u{C9}", lower_matches), "caf\u{E9}");
    // The Turkish locale's dotless i is the Turkic fold here.
    assert_eq!(normalize(&CaseFold::turkic_simple(), "I").0, "\u{131}");
}

#[test]
fn opennlp_full_fold_expectations() {
    // FullCaseFoldCharSequenceNormalizerTest, with UTF-16 offsets rewritten
    // as UTF-8 byte offsets.
    let rows = case_folding();
    let full = CaseFold::full();
    let reference = reference_fold(&rows, "FC");
    assert_eq!(normalize(&full, "").0, "");
    assert_eq!(run_fold(full, "ABC", fold_matches(&reference)), "abc");
    assert_eq!(run_fold(full, "\u{DF}", fold_matches(&reference)), "ss");
    assert_eq!(run_fold(full, "\u{FB00}", fold_matches(&reference)), "ff");
    assert_eq!(run_fold(full, "Ma\u{DF}e", fold_matches(&reference)), "masse");
    for unchanged in ["already lower", "\u{1F600}"] {
        assert!(matches!(apply(&full, unchanged), Cow::Borrowed(_)), "{unchanged:?}");
    }
    // alignmentMapsExpansionBackToTheSource: "ss" maps back to the sharp s.
    let (out, a) = normalize(&full, "Ma\u{DF}e");
    assert_eq!(out, "masse");
    assert_eq!(back(&a, 2, 4), (2, 4));
    // supplementaryPlaneFoldRoundTripsThroughAlignment.
    let (out, a) = normalize(&full, "\u{10400}");
    assert_eq!(out, "\u{10428}");
    assert_eq!(back(&a, 0, 4), (0, 4));
    // tripleCodePointExpansionMapsAsOneBlock.
    let (out, a) = normalize(&full, "x\u{390}y");
    assert_eq!(out, "x\u{3B9}\u{308}\u{301}y");
    assert_eq!(back(&a, 1, 7), (1, 3));
    assert_eq!(back(&a, 3, 5), (1, 3));
    // consecutiveExpansionsEachMapToTheirOwnSource. Each sharp s is
    // replaced on its own and touching edits stay separate, so each
    // half maps back to exactly its own source character.
    let (out, a) = normalize(&full, "\u{DF}\u{DF}");
    assert_eq!(out, "ssss");
    assert_eq!(back(&a, 0, 2), (0, 2));
    assert_eq!(back(&a, 2, 4), (2, 4));
    // tableLoadsEveryCommonAndFullFolding counts 1585 C and F rows in
    // CaseFolding.txt 17.0.0; 18.0.0 has 1606
    // (`every_case_folding_row_through_full_simple_and_turkic`).
    assert_eq!(rows[&'C'].len() + rows[&'F'].len(), 1606);
}

#[test]
fn opennlp_whitespace_then_full_fold_pipeline() {
    // composesIntoTheOffsetAwarePipeline and
    // identityPipelinePreservesEveryOffset.
    let chain = Chain::new().then(Whitespace::collapse()).then(CaseFold::full());
    // OpenNLP maps "ss" back to the sharp s alone (3..5 in UTF-8), and
    // touching edits stay separate under `compose` and `ViewBuilder`, so
    // the span comes back as exactly the sharp s.
    let (out, a) = normalize(&chain, "A  \u{DF}");
    assert_eq!(out, "a ss");
    assert_eq!(back(&a, 2, 4), (3, 5));
    // Apart from each other, each change keeps its own source span.
    let (out, a) = normalize(&chain, "A x  y \u{DF}");
    assert_eq!(out, "a x y ss");
    assert_eq!(back(&a, 6, 8), (7, 9));
    assert_eq!(back(&a, 0, 1), (0, 1));
    let (out, a) = normalize(&chain, "plain text");
    assert_eq!(out, "plain text");
    assert_eq!(back(&a, 6, 10), (6, 10));
    assert!(a.edits().is_empty());
}
