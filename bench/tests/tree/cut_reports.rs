//! The chunker v1 backend READMEs under analysis/eval/records/v1-172
//! describe the CPU-vs-backend cut reports that are committed: a README
//! whose backend has `reports/cuts-cpu-<backend>-*.txt` names each of
//! them and does not call the cut report deferred or missing.

use crate::common::workspace;

/// Each committed `cuts-cpu-<backend>-*.txt` report for one backend.
fn cut_reports(reports: &std::path::Path, backend: &str) -> Vec<String> {
    let prefix = format!("cuts-cpu-{backend}-");
    let mut out: Vec<String> = std::fs::read_dir(reports)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.starts_with(&prefix) && n.ends_with(".txt"))
        .collect();
    out.sort();
    out
}

/// What a README must say, given its text and the reports that exist.
fn stale_claims(readme: &str, reports: &[String]) -> Vec<String> {
    let mut bad = Vec::new();
    if reports.is_empty() {
        return bad;
    }
    let lower = readme.to_lowercase();
    for phrase in ["deferred", "dumps are missing"] {
        if lower.contains(phrase) {
            bad.push(format!("says \"{phrase}\" though cut reports exist"));
        }
    }
    for r in reports {
        if !readme.contains(r.as_str()) {
            bad.push(format!("does not name reports/{r}"));
        }
    }
    bad
}

#[test]
fn stale_claims_flag_a_deferred_note_beside_a_committed_report() {
    let reports = vec!["cuts-cpu-npu-x.txt".to_owned()];
    assert_eq!(stale_claims("The cut report is deferred.", &reports).len(), 2);
    assert!(stale_claims("See `reports/cuts-cpu-npu-x.txt`.", &reports).is_empty());
    assert!(stale_claims("The cut report is deferred.", &[]).is_empty());
}

#[test]
fn the_npu_readme_describes_its_committed_cut_reports() {
    let root = workspace().join("analysis/eval/records/v1-172");
    let readme = std::fs::read_to_string(root.join("npu/README.md")).unwrap();
    let reports = cut_reports(&root.join("reports"), "npu");
    assert_eq!(reports.len(), 2, "expected the hybrid and cohesion-1000 NPU cut reports, found {reports:?}");
    let bad = stale_claims(&readme, &reports);
    assert!(bad.is_empty(), "npu/README.md:\n  {}", bad.join("\n  "));
}
