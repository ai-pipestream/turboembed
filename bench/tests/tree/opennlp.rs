//! No unlisted case-insensitive `opennlp` mentions in `docs/` or READMEs.
//! Every allowlisted file or directory prefix carries a stated reason.

use std::path::Path;

/// Single allowlist of exact `docs/` and `README.md` file paths permitted to
/// mention `opennlp`, paired with the reason for each entry. Directory prefixes
/// are not permitted so newly added files under `docs/receipts/` or
/// `docs/beyond-opennlp/` are still checked.
pub const ALLOWLIST: &[(&str, &str)] = &[
    (
        "docs/receipts/long-document-1486/chunk-gate.txt",
        "Existing native semantic baseline identifier in frozen #1486 gate output",
    ),
    (
        "docs/receipts/long-document-1486/paired.json",
        "Existing native semantic baseline identifier in frozen #1486 paired scores",
    ),
    ("docs/licences.md", "Upstream Apache-2.0 and corpus licence attributions"),
    ("docs/reference-inventory.md", "Pinned reference inventory table"),
    ("docs/analysis-features.md", "Feature checklist mapping table (#1536)"),
    ("docs/beyond-opennlp/other-languages.md", "External library survey notes"),
    ("docs/beyond-opennlp/rust-crates.md", "External Rust crate survey notes"),
    ("docs/receipts/crf-726/README.md", "Historical CRF #726 benchmark receipt"),
    ("docs/receipts/progressive-snapshot-adoption-1249.md", "Historical progressive snapshot #1249 receipt"),
    ("docs/receipts/unigram-984/README.md", "Historical unigram #984 benchmark receipt"),
    ("docs/receipts/core-708/tests.log", "Historical core #708 test receipt log"),
    ("docs/receipts/crf-923/report.json", "Historical CRF #923 benchmark receipt JSON"),
    ("docs/receipts/depparse-688/compact-tags-tests.log", "Historical depparse #688 compact-tags test receipt log"),
    ("docs/receipts/depparse-688/cycle-tests.log", "Historical depparse #688 cycle test receipt log"),
    ("docs/receipts/depparse-688/forest-split-tests.log", "Historical depparse #688 forest-split test receipt log"),
    ("docs/receipts/depparse-688/formats-tests.log", "Historical depparse #688 formats test receipt log"),
    ("docs/receipts/depparse-688/tests.log", "Historical depparse #688 test receipt log"),
    ("docs/receipts/depparse-689/tests.log", "Historical depparse #689 test receipt log"),
    ("docs/receipts/document-trace/main.rs", "Historical document-trace receipt harness"),
    ("docs/receipts/heads-levelzero-989/tests.log", "Historical Level Zero heads #989 test receipt log"),
    ("docs/receipts/heads-metal-989/tests.log", "Historical Metal heads #989 test receipt log"),
    ("docs/receipts/maxent-embed-660/tests.log", "Historical maxent-embed #660 test receipt log"),
    ("docs/receipts/parser-506/green-tests.log", "Historical parser #506 green test receipt log"),
    ("docs/receipts/parser-506/inference-tests.log", "Historical parser #506 inference test receipt log"),
    ("docs/receipts/parser-postag-649/dependents.log", "Historical parser-postag #649 dependents receipt log"),
    ("docs/receipts/parser-postag-649/tests.log", "Historical parser-postag #649 test receipt log"),
    ("docs/receipts/sentiment-647/split-parity.log", "Historical sentiment #647 split receipt log"),
    ("docs/receipts/sentiment-647/tests.log", "Historical sentiment #647 test receipt log"),
    ("docs/receipts/unigram-984/1-blocked.log", "Historical unigram #984 run 1 blocked receipt log"),
    ("docs/receipts/unigram-984/1-fixed.log", "Historical unigram #984 run 1 fixed receipt log"),
    ("docs/receipts/unigram-984/2-blocked.log", "Historical unigram #984 run 2 blocked receipt log"),
    ("docs/receipts/unigram-984/2-fixed.log", "Historical unigram #984 run 2 fixed receipt log"),
    ("docs/receipts/unigram-984/3-blocked.log", "Historical unigram #984 run 3 blocked receipt log"),
    ("docs/receipts/unigram-984/3-fixed.log", "Historical unigram #984 run 3 fixed receipt log"),
    ("docs/receipts/unigram-984/4-blocked.log", "Historical unigram #984 run 4 blocked receipt log"),
    ("docs/receipts/unigram-984/4-fixed.log", "Historical unigram #984 run 4 fixed receipt log"),
    ("docs/receipts/unigram-984/5-blocked.log", "Historical unigram #984 run 5 blocked receipt log"),
    ("docs/receipts/unigram-984/5-fixed.log", "Historical unigram #984 run 5 fixed receipt log"),
    ("docs/receipts/unigram-984/6-blocked.log", "Historical unigram #984 run 6 blocked receipt log"),
    ("docs/receipts/unigram-984/6-fixed.log", "Historical unigram #984 run 6 fixed receipt log"),
    ("docs/receipts/unigram-984/7-blocked.log", "Historical unigram #984 run 7 blocked receipt log"),
    ("docs/receipts/unigram-984/7-fixed.log", "Historical unigram #984 run 7 fixed receipt log"),
    ("docs/receipts/unigram-984/8-blocked.log", "Historical unigram #984 run 8 blocked receipt log"),
    ("docs/receipts/unigram-984/8-fixed.log", "Historical unigram #984 run 8 fixed receipt log"),
    ("docs/receipts/unigram-984/9-blocked.log", "Historical unigram #984 run 9 blocked receipt log"),
    ("docs/receipts/unigram-984/9-fixed.log", "Historical unigram #984 run 9 fixed receipt log"),
    ("docs/receipts/unigram-984/10-blocked.log", "Historical unigram #984 run 10 blocked receipt log"),
    ("docs/receipts/unigram-984/10-fixed.log", "Historical unigram #984 run 10 fixed receipt log"),
    ("docs/receipts/unigram-984/11-blocked.log", "Historical unigram #984 run 11 blocked receipt log"),
    ("docs/receipts/unigram-984/11-fixed.log", "Historical unigram #984 run 11 fixed receipt log"),
    ("docs/receipts/unigram-984/12-blocked.log", "Historical unigram #984 run 12 blocked receipt log"),
    ("docs/receipts/unigram-984/12-fixed.log", "Historical unigram #984 run 12 fixed receipt log"),
    ("docs/receipts/unigram-984/rejected/byte-scan/1-blocked.log", "Historical unigram #984 byte-scan 1 blocked log"),
    ("docs/receipts/unigram-984/rejected/byte-scan/1-fixed.log", "Historical unigram #984 byte-scan 1 fixed log"),
    ("docs/receipts/unigram-984/rejected/byte-scan/2-blocked.log", "Historical unigram #984 byte-scan 2 blocked log"),
    ("docs/receipts/unigram-984/rejected/byte-scan/2-fixed.log", "Historical unigram #984 byte-scan 2 fixed log"),
    ("docs/receipts/unigram-984/rejected/byte-scan/3-blocked.log", "Historical unigram #984 byte-scan 3 blocked log"),
    ("docs/receipts/unigram-984/rejected/byte-scan/3-fixed.log", "Historical unigram #984 byte-scan 3 fixed log"),
    ("docs/receipts/unigram-984/rejected/byte-scan/4-blocked.log", "Historical unigram #984 byte-scan 4 blocked log"),
    ("docs/receipts/unigram-984/rejected/byte-scan/4-fixed.log", "Historical unigram #984 byte-scan 4 fixed log"),
    ("docs/receipts/unigram-984/rejected/byte-scan/summary.json", "Historical unigram #984 byte-scan summary JSON"),
    (
        "docs/receipts/unigram-984/rejected/prepare-rescan/1-blocked.log",
        "Historical unigram #984 prepare-rescan 1 blocked log",
    ),
    (
        "docs/receipts/unigram-984/rejected/prepare-rescan/1-fixed.log",
        "Historical unigram #984 prepare-rescan 1 fixed log",
    ),
    (
        "docs/receipts/unigram-984/rejected/prepare-rescan/2-blocked.log",
        "Historical unigram #984 prepare-rescan 2 blocked log",
    ),
    (
        "docs/receipts/unigram-984/rejected/prepare-rescan/2-fixed.log",
        "Historical unigram #984 prepare-rescan 2 fixed log",
    ),
    (
        "docs/receipts/unigram-984/rejected/prepare-rescan/summary.json",
        "Historical unigram #984 prepare-rescan summary JSON",
    ),
    ("docs/receipts/unigram-984/screens/range-screen.log", "Historical unigram #984 range-screen receipt log"),
    ("docs/receipts/unigram-984/summarize.mjs", "Historical unigram #984 summarize script receipt"),
    ("docs/receipts/unigram-984/summary.json", "Historical unigram #984 summary receipt JSON"),
    ("analysis/depparse/tests/apache/README.md", "Apache-2.0 dependency fixture attribution"),
    ("analysis/depparse/tests/data/README.md", "Dependency test corpus attribution"),
    ("analysis/depparse/tests/feedforward/data/README.md", "Feedforward dependency fixture attribution"),
    ("analysis/geo/data/README.md", "GeoNames gazetteer data attribution"),
    ("analysis/goldens/data/README.md", "Legacy golden fixture README slated for removal under #1511/#1529"),
    ("analysis/goldens-tool/README.md", "Legacy goldens-tool crate slated for removal under #1529"),
    ("analysis/maxent/tests/fixtures/README.md", "Legacy maxent fixture README slated for removal under #1530"),
    ("analysis/ngram/README.md", "Legacy ngram crate slated for removal under #1532"),
    ("analysis/normalize/tests/data/spellcheck/README.md", "Spellcheck evaluation corpus attribution"),
    ("analysis/parser/README.md", "Legacy constituency parser crate slated for removal under #1530"),
    ("analysis/phrase-chunker/README.md", "Legacy phrase-chunker crate slated for removal under #1530"),
    ("analysis/sentdetect/tests/fixtures/README.md", "Sentence detector fixture attribution"),
    ("analysis/sentdetect/tests/models/README.md", "Sentence detector test model attribution"),
    ("analysis/sequence/README.md", "Legacy sequence crate slated for removal under #1530"),
    ("analysis/stem/snowball/README.md", "Snowball stemmer generator source attribution"),
    ("analysis/stopword/data/README.md", "Bundled stopword list licence attribution"),
    ("analysis/tokenize/data/ud-cjk/README.md", "UD CJK tokenizer data attribution"),
    ("analysis/tokenize/tests/models/README.md", "Tokenizer test model attribution"),
    ("analysis/train/README.md", "Legacy train crate slated for removal under #1530"),
    ("analysis/train-components/README.md", "Legacy train-components crate slated for removal under #1530"),
    ("analysis/train-recipes/README.md", "Training recipes documentation owned by #1529"),
    ("analysis/eval/records/v1-172/metal/README.md", "Committed Metal chunker v1 evaluation record README (#172)"),
    ("clients/java-panama/README.md", "Java Panama adapter package ai.pipestream.turboembed.opennlp README (#1470)"),
    ("clients/java-jni/README.md", "Java JNI adapter package ai.pipestream.turboembed.jni.opennlp README (#1800)"),
    ("trainer/recipes/records/masc-pos-1649/README.md", "Committed MASC Penn POS training record README (#1649)"),
];

/// Wire format identifiers in source code that carry an `opennlp` prefix and
/// may be cited in backticks in contract documentation.
const WIRE_IDENTIFIERS: &[&str] = &["`opennlp-dictionary-xml`"];

fn is_allowlisted(rel: &str) -> bool {
    ALLOWLIST.iter().any(|&(entry, _)| rel == entry)
}

fn contains_opennlp(bytes: &[u8]) -> bool {
    let mut text = String::from_utf8_lossy(bytes).into_owned();
    for &wire_id in WIRE_IDENTIFIERS {
        text = text.replace(wire_id, "");
    }
    const NEEDLE: &[u8; 7] = b"opennlp";
    text.as_bytes().windows(NEEDLE.len()).any(|w| w.eq_ignore_ascii_case(NEEDLE))
}

fn collect_candidates(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(root) else { continue };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if path.is_dir() {
            if matches!(rel_str.as_str(), ".git" | "target") {
                continue;
            }
            collect_candidates(root, &path, out);
        } else if path.is_file()
            && (rel_str.starts_with("docs/") || rel_str == "README.md" || rel_str.ends_with("/README.md"))
        {
            out.push(rel_str);
        }
    }
}

/// Scans `docs/` and all `README.md` files under `root` for case-insensitive
/// `opennlp` mentions outside [`ALLOWLIST`]. Returns `(files_scanned, violations)`.
pub fn violations(root: &Path) -> (usize, Vec<String>) {
    let mut candidates = Vec::new();
    collect_candidates(root, root, &mut candidates);
    candidates.sort();
    let mut found = Vec::new();
    for rel in &candidates {
        if is_allowlisted(rel) {
            continue;
        }
        let Ok(bytes) = std::fs::read(root.join(rel)) else { continue };
        if contains_opennlp(&bytes) {
            found.push(rel.clone());
        }
    }
    (candidates.len(), found)
}

#[test]
fn allowlist_entries_all_carry_a_non_empty_reason() {
    for &(entry, reason) in ALLOWLIST {
        assert!(!entry.trim().is_empty(), "empty allowlist path entry");
        assert!(!entry.ends_with('/'), "directory prefix {entry} is not permitted in ALLOWLIST");
        assert!(!reason.trim().is_empty(), "allowlist entry {entry} has no reason");
    }
}

#[test]
fn docs_and_readmes_have_no_unlisted_opennlp_mentions() {
    let root = crate::common::workspace();
    let (seen, found) = violations(&root);
    assert!(seen > 40, "only {seen} docs and READMEs scanned");
    assert!(
        found.is_empty(),
        "unlisted case-insensitive `opennlp` mentions in docs/ or READMEs:\n  {}",
        found.join("\n  ")
    );
}

#[test]
fn opennlp_check_fails_on_a_fixture_with_one_unlisted_mention() {
    let dir = crate::common::scratch("opennlp-mentions");
    std::fs::create_dir_all(dir.join("docs/receipts")).unwrap();
    std::fs::create_dir_all(dir.join("analysis/sample")).unwrap();
    std::fs::write(dir.join("README.md"), "# Workspace\nClean root README.\n").unwrap();
    std::fs::write(dir.join("docs/licences.md"), "Apache OpenNLP attribution is allowlisted.\n").unwrap();
    std::fs::write(dir.join("docs/guide.md"), "Wire contract format `opennlp-dictionary-xml` is permitted.\n").unwrap();
    std::fs::write(dir.join("docs/current-api.txt"), "Unlisted OpenNLP compatibility claim in a text file.\n").unwrap();
    std::fs::write(dir.join("docs/receipts/new-api.md"), "Unlisted OpenNLP compatibility claim.\n").unwrap();
    std::fs::write(
        dir.join("analysis/sample/README.md"),
        "This fixture mentions OpenNLP without being on the allowlist.\n",
    )
    .unwrap();

    let (seen, found) = violations(&dir);
    assert_eq!(seen, 6);
    assert_eq!(
        found,
        vec![
            "analysis/sample/README.md".to_owned(),
            "docs/current-api.txt".to_owned(),
            "docs/receipts/new-api.md".to_owned(),
        ]
    );
}

#[test]
fn documented_analysis_segmentation_and_normalizer_apis_match_crate_exports() {
    let root = crate::common::workspace();
    let analysis_doc = std::fs::read_to_string(root.join("docs/analysis.md")).unwrap();
    let seg_doc = std::fs::read_to_string(root.join("docs/segmentation.md")).unwrap();
    let norm_doc = std::fs::read_to_string(root.join("docs/normalizers.md")).unwrap();
    let lang_doc = std::fs::read_to_string(root.join("docs/langdetect.md")).unwrap();

    for &(doc_name, text) in &[
        ("docs/analysis.md", analysis_doc.as_str()),
        ("docs/segmentation.md", seg_doc.as_str()),
        ("docs/normalizers.md", norm_doc.as_str()),
        ("docs/langdetect.md", lang_doc.as_str()),
    ] {
        for forbidden in [
            "SatSentenceDetector",
            "Uax29Tokenizer",
            "EditTreeLemmatizer",
            "BiLstmCrfPosTagger",
            "GeoLocations",
            "turbo:geo-locations",
            "turbo:text-quality",
            "ParagraphSplitter",
            "turbo:paragraphs",
            "InventorySentenceDetector",
            "InventoryLanguage",
            "NewlineSentenceDetector",
            "SentenceDetectorME",
            "TokenReadings",
            "turbo:token-readings",
            "TokenizerME",
            "UnigramTokenizer",
            "ComplexScriptTokenizer",
            "DictionaryDetokenizer",
            "NormalizeChain",
            "UnicodeFormNormalizer",
            "CaseFoldNormalizer",
            "ConfusablesNormalizer",
            "SpanDetector",
            "CalibratedSpanDetector",
            "AcceptFirst",
            "analysis/formats/recipes/",
            "LanguageDetector",
            "ProbingContext",
            "ProbingConfig",
        ] {
            assert!(!text.contains(forbidden), "{doc_name} still contains nonexistent API or path {forbidden:?}");
        }
    }

    for required in [
        "SatAnnotator",
        "ParagraphDetector",
        "NewlineDetector",
        "SentenceDetector",
        "WordTokenizer",
        "TokenizerMe",
        "UnigramSegmenter",
        "ComplexSegmenter",
        "Detokenizer",
        "BilstmPosTagger",
        "MaxentLemmatizer",
        "ScriptDetector",
        "Locations",
        "EntityLocationLinks",
        "turbo:text_quality",
        "LicenceClass",
    ] {
        assert!(analysis_doc.contains(required), "docs/analysis.md missing exported symbol {required:?}");
    }
    for required in ["ParagraphDetector", "SatAnnotator", "Readings", "turbo:readings", "TokenizerMe"] {
        assert!(seg_doc.contains(required), "docs/segmentation.md missing exported symbol {required:?}");
    }
    for required in ["NgramModel", "NgramDetector", "ScriptDetector", "with_ngram_detector", "langid.ngram"] {
        assert!(lang_doc.contains(required), "docs/langdetect.md missing exported symbol {required:?}");
    }
    for required in [
        "Chain",
        "UnicodeForm",
        "CaseFold",
        "AccentFold",
        "ConfusableSkeleton",
        "HtmlTagStripNormalizer",
        "Corrections",
        "Artifacts",
        "Noise",
        "Placeholders",
    ] {
        assert!(norm_doc.contains(required), "docs/normalizers.md missing exported symbol {required:?}");
    }
}
