use std::path::Path;

use turbo::TURBO_PRECISION_MODEL;

use super::*;
use crate::measure::RowKind;

fn r(row: u32, start: u32, end: u32) -> turbo_token_range {
    turbo_token_range { row, start, end }
}

#[test]
fn chunks_cover_the_text_tokens_of_each_row() {
    let got: Vec<(u32, u32, u32)> = chunk_ranges(2, 10, 3).unwrap().iter().map(|x| (x.row, x.start, x.end)).collect();
    // Positions 1 to 8 of each row: 1..4, 4..7, 7..9.
    assert_eq!(got, [(0, 1, 4), (0, 4, 7), (0, 7, 9), (1, 1, 4), (1, 4, 7), (1, 7, 9)]);
    assert!(chunk_ranges(1, 10, 0).is_err());
    assert!(chunk_ranges(1, 2, 1).is_err());
}

#[test]
fn a_chunk_row_is_its_tokens_between_the_row_s_ends() {
    let rows = Rows {
        kind: RowKind::Dense,
        batch: 1,
        seq: 6,
        ids: vec![101, 5, 6, 7, 8, 102],
        mask: vec![1; 6],
        types: vec![0; 6],
        cases: vec![3],
    };
    let c = chunk_rows(&rows, &[r(0, 1, 4), r(0, 4, 5)], 0);
    assert_eq!((c.batch, c.seq), (2, 5));
    assert_eq!(c.ids, [101, 5, 6, 7, 102, 101, 8, 102, 0, 0]);
    assert_eq!(c.mask, [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
    assert_eq!(c.cases, [3, 3]);
}

#[test]
fn the_three_ways_run_on_the_small_bert() {
    let plan = RangePlan {
        bundle: Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle"),
        device: "cpu".into(),
        precision: TURBO_PRECISION_MODEL,
        batch: Some(2),
        seq: Some(32),
        chunk: 8,
        warmup: 1,
        iterations: 3,
    };
    let p = Provenance { top: String::new(), commit: "a".repeat(40), pushed_to: vec!["origin/main".into()] };
    let r = measure(&plan, &p, "2026-10-22T00:00:00Z".into()).unwrap();
    // 30 text tokens a row: chunks of 8, 8, 8 and 6.
    assert_eq!((r.shape.chunks_per_row, r.rows.vectors, r.ranges.vectors, r.chunk_rows.vectors), (4, 2, 8, 8));
    // A range over a whole row is the row's vector, to the bit.
    assert_eq!(r.whole_row_max_abs_diff, 0.0);
    assert!(r.chunk_min_cosine > 0.0 && r.chunk_min_cosine <= 1.0 + 1e-12, "{}", r.chunk_min_cosine);
    assert!(r.ranges_over_rows > 0.0 && r.chunk_rows_over_ranges > 0.0);
    let arch = r.machine.arch.to_lowercase().replace(|c: char| !c.is_ascii_alphanumeric(), "-");
    assert_eq!(
        file_name(&r),
        format!("{arch}.cpu.ranges.model.all-minilm-l6-v2-{}.aaaaaaaaaaaa.json", &r.bundle.manifest_sha256[..8])
    );
    // The record passes its own check; a home directory, a ratio that is
    // not the p50s' quotient, or an unpushed commit does not.
    check(&r).unwrap();
    let mut bad = r.clone();
    bad.device.name = "/home/someone/gpu".into();
    assert!(check(&bad).unwrap_err().contains("host path"));
    let mut bad = r.clone();
    bad.ranges_over_rows *= 1.01;
    assert!(check(&bad).is_err());
    let mut bad = r.clone();
    bad.library.pushed_to.clear();
    assert!(check(&bad).is_err());
    let dir = std::env::temp_dir().join(format!("turbo-bench-ranges-{}", std::process::id()));
    assert!(write(&bad, &dir).is_err());
}
