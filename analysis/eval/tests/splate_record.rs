//! Check documentation against the original, unmodified trained receipt.
use serde_json::Value;

const DOC: &str = include_str!("../../../docs/splate.md");
const RECORD: &str = include_str!("../records/splate-trained-913/splate-trained-913.json");

fn cells(line: &str) -> Vec<&str> {
    line.trim_matches('|').split('|').map(str::trim).collect()
}

fn arms(table: &Value) -> Vec<Vec<f64>> {
    table["markdown"]
        .as_str()
        .unwrap()
        .lines()
        .filter(|line| line.starts_with("| Arm ") && !line.starts_with("| Arm |"))
        .map(|line| cells(line)[1..].iter().map(|v| v.parse().unwrap()).collect())
        .collect()
}

#[test]
fn trained_summary_cells_match_the_original_record() {
    let record: Value = serde_json::from_str(RECORD).unwrap();
    for set in record["sets"].as_array().unwrap() {
        let name = set["set"].as_str().unwrap();
        let heading = format!("Trained run on `{name}`");
        let section = DOC.split_once(&heading).unwrap().1.split("### ").next().unwrap();
        let summaries: Vec<_> =
            section.lines().filter(|s| s.starts_with("| Arm ") && !s.starts_with("| Arm |")).collect();
        assert_eq!(summaries.len(), 4);
        for (depth, table) in set["tables"].as_array().unwrap().iter().enumerate() {
            let rows = arms(table);
            assert_eq!(rows.len(), 4);
            for (arm, row) in rows.iter().enumerate() {
                let summary = cells(summaries[arm]);
                let metrics: Vec<f64> = summary[depth + 1].split('/').map(|s| s.trim().parse().unwrap()).collect();
                // Receipt columns: C, R@1, R@5, R@10, R@100, nDCG@10, ...
                assert_eq!(metrics, [row[5], row[4]], "{name}, arm {}, depth {depth}", arm + 1);
                if depth == 3 {
                    let ms: f64 = summary[5].trim_end_matches(" ms").parse().unwrap();
                    assert!((ms - row[9] / 1000.0).abs() <= 0.005, "search latency rounding");
                }
            }
        }
    }
}

#[test]
fn decision_discloses_each_arm_two_quality_win() {
    let record: Value = serde_json::from_str(RECORD).unwrap();
    let decision = DOC.split_once("### 5.3").unwrap().1;
    for set in record["sets"].as_array().unwrap() {
        let name = set["set"].as_str().unwrap();
        for (metric, column) in [("nDCG@10", 5), ("Recall@100", 4)] {
            let winning_depths: Vec<_> = set["tables"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|table| {
                    let rows = arms(table);
                    rows[1][column] > rows[2][column]
                })
                .map(|table| format!("C={}", table["c"]))
                .collect();
            if !winning_depths.is_empty() {
                let disclosure = format!("`{name}` {metric}: Arm 2 leads at {}.", winning_depths.join(", "));
                assert!(decision.contains(&disclosure), "missing receipt evidence: {disclosure}");
            }
        }
    }
}
