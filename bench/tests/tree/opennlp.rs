//! No unlisted case-insensitive `opennlp` mentions in `docs/` or READMEs.
//! Every allowlisted file or directory prefix carries a stated reason.

use std::path::Path;

/// Single allowlist of `docs/` and `README.md` paths or directory prefixes
/// permitted to mention `opennlp`, paired with the reason for each entry.
pub const ALLOWLIST: &[(&str, &str)] = &[
    ("docs/licences.md", "Upstream Apache-2.0 and corpus licence attributions"),
    ("docs/reference-inventory.md", "Pinned reference inventory table"),
    ("docs/analysis-features.md", "Feature checklist mapping table (#1536)"),
    ("docs/beyond-opennlp/", "External library survey notes"),
    ("docs/receipts/", "Historical benchmark and parity receipts"),
    ("analysis/depparse/tests/apache/README.md", "Apache-2.0 dependency fixture attribution"),
    ("analysis/depparse/tests/data/README.md", "Dependency test corpus attribution"),
    ("analysis/depparse/tests/feedforward/data/README.md", "Feedforward dependency fixture attribution"),
    ("analysis/geo/data/README.md", "GeoNames gazetteer data attribution"),
    ("analysis/goldens/data/README.md", "Legacy golden fixture README slated for removal under #1511/#1529"),
    ("analysis/goldens-tool/README.md", "Legacy goldens-tool crate slated for removal under #1529"),
    ("analysis/jvm/README.md", "Legacy jvm crate slated for removal under #1529"),
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
];

fn is_allowlisted(rel: &str) -> bool {
    ALLOWLIST.iter().any(|&(entry, _)| if entry.ends_with('/') { rel.starts_with(entry) } else { rel == entry })
}

fn contains_opennlp(bytes: &[u8]) -> bool {
    const NEEDLE: &[u8; 7] = b"opennlp";
    bytes.windows(NEEDLE.len()).any(|w| w.eq_ignore_ascii_case(NEEDLE))
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
    std::fs::create_dir_all(dir.join("docs")).unwrap();
    std::fs::create_dir_all(dir.join("analysis/sample")).unwrap();
    std::fs::write(dir.join("README.md"), "# Workspace\nClean root README.\n").unwrap();
    std::fs::write(dir.join("docs/licences.md"), "Apache OpenNLP attribution is allowlisted.\n").unwrap();
    std::fs::write(dir.join("docs/guide.md"), "Pure Rust guide with no external parity claim.\n").unwrap();
    std::fs::write(
        dir.join("analysis/sample/README.md"),
        "This fixture mentions OpenNLP without being on the allowlist.\n",
    )
    .unwrap();

    let (seen, found) = violations(&dir);
    assert_eq!(seen, 4);
    assert_eq!(found, vec!["analysis/sample/README.md".to_owned()]);
}
