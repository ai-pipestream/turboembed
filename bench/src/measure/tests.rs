use super::*;

fn padded_row() -> Rows {
    // One row of 128, four live tokens at the start. A packed count
    // stops at the last live token. The npu frame runs all 128.
    let mut mask = vec![0; 128];
    mask[..4].fill(1);
    Rows { kind: RowKind::Mixed, batch: 1, seq: 128, ids: vec![1; 128], mask, types: vec![0; 128], cases: vec![0] }
}

#[test]
fn npu_counts_the_compiled_frame_and_other_backends_pack() {
    let rows = padded_row();
    assert_eq!(rows.packed_tokens(), 4);
    assert_eq!(library_computed_tokens("npu", &rows), 128);
    assert_eq!(library_computed_tokens("cpu", &rows), 4);
    assert_eq!(library_computed_tokens("levelzero", &rows), 4);
}

#[test]
fn the_published_npu_cell_is_the_compiled_frame() {
    assert!(require_compiled_frame(Some(1), Some(128), 1, 128).is_ok());
    let omitted = require_compiled_frame(None, None, 1, 128).unwrap_err();
    assert!(omitted.contains("--batch 1 --seq 128"), "{omitted}");
    assert!(omitted.contains("openvino-f16"), "{omitted}");
    assert!(omitted.contains("[1, 128]"), "{omitted}");
    let other = require_compiled_frame(Some(32), Some(128), 1, 128).unwrap_err();
    assert!(other.contains("--batch 1 --seq 128"), "{other}");
    assert!(require_compiled_frame(Some(1), Some(64), 1, 128).is_err());
}

#[test]
fn npu_graph_format_comes_from_the_session_choices() {
    let mut session: turbo_session_info = unsafe { std::mem::zeroed() };
    session.tuned = turbo::TURBO_TUNED_DEFAULT;
    for (i, b) in b"NGRAPH_LITE".iter().enumerate() {
        session.choices[i] = *b as _;
    }
    assert_eq!(library_settings("npu", &session), ["TURBO_NPU_GRAPH_FORMAT=NGRAPH_LITE"]);
    session.choices = [0; turbo::TURBO_CHOICES_LEN];
    for (i, b) in b"NATIVE".iter().enumerate() {
        session.choices[i] = *b as _;
    }
    assert_eq!(library_settings("npu", &session), ["TURBO_NPU_GRAPH_FORMAT=NATIVE"]);
    // Empty choices do not invent a format, and GRAPH_INPUT is not
    // taken from the environment.
    session.choices[0] = 0;
    assert!(library_settings("npu", &session).is_empty());
}

#[test]
fn embeddings_have_no_speed_ratio() {
    let settings = vec!["TURBO_NPU_GRAPH_INPUT=INPUT_EMBEDDINGS".into()];
    let why = npu_embeddings_speed("npu", &settings, true, false).unwrap();
    assert!(why.contains("INPUT_EMBEDDINGS has no speed_ratio"), "{why}");
    assert!(npu_embeddings_speed("npu", &settings, false, true).is_some());
    assert!(npu_embeddings_speed("npu", &settings, false, false).is_none());
    let token = vec!["TURBO_NPU_GRAPH_INPUT=INPUT_TOKEN_IDS".into()];
    assert!(npu_embeddings_speed("npu", &token, true, true).is_none());
    assert!(npu_embeddings_speed("levelzero", &settings, true, true).is_none());
}

fn cases_of(lengths: &[usize]) -> Reference {
    Reference {
        texts: lengths.iter().map(|_| ("t".into(), 0)).collect(),
        ids: lengths.iter().map(|&n| vec![7; n]).collect(),
        vectors: lengths.iter().map(|_| vec![1.0]).collect(),
    }
}

#[test]
fn a_batch_of_one_cycles_every_fitting_case_in_its_own_frame() {
    // Length 200 does not fit seq 128. The other three do, in case order.
    let reference = cases_of(&[2, 5, 200, 4]);
    let rows = Rows::cycle(&reference, 0, 128).unwrap();
    assert_eq!((rows.batch, rows.seq, rows.frames()), (1, 128, 3));
    assert_eq!(rows.cases, vec![0, 1, 3]);
    assert_eq!(rows.ids.len(), 3 * 128);
    assert_eq!(rows.live(0), &[7, 7]);
    assert_eq!(rows.live(1).len(), 5);
    assert_eq!(rows.live(2).len(), 4);
    assert!(rows.mask[2..128].iter().all(|&m| m == 0), "case 0 is padded out to the frame");
    for f in 0..rows.frames() {
        let frame = rows.frame(f);
        assert_eq!((frame.batch, frame.seq, frame.frames(), frame.ids.len()), (1, 128, 1, 128));
        assert_eq!(frame.cases, vec![rows.cases[f]]);
        assert_eq!(frame.ids, rows.frame_batch(f).ids);
    }
    // One fitting case is the single frame build already made.
    let only = cases_of(&[2]);
    assert_eq!(Rows::cycle(&only, 9, 16).unwrap(), Rows::build(&only, 9, 1, 16).unwrap());
    // A wider batch still fills one frame and cycles inside it.
    let wide = Rows::build(&reference, 0, 4, 128).unwrap();
    assert_eq!(wide.frames(), 1);
    assert_eq!(wide.cases, vec![0, 1, 3, 0]);
    assert_eq!(wide.padded_tokens(), 4 * 128);
    // The cycle does not pretend the graph's batch is the case count.
    assert_eq!(rows.padded_tokens(), 3 * 128);
    assert_eq!(library_computed_tokens("npu", &rows), 3 * 128);
    assert_eq!(library_computed_tokens("cpu", &rows), 2 + 5 + 4);
    assert_ne!(rows.sha256(), Rows::build(&reference, 0, 1, 128).unwrap().sha256());
    assert!(Rows::cycle(&reference, 0, 1).is_err(), "no case is one token");
    assert_eq!(Rows::frame_dir(std::path::Path::new("/w"), 2), std::path::Path::new("/w/frame-2"));
}

#[test]
fn redact_user_paths_preserves_trailing_text_after_root_and_bare_user_paths() {
    assert_eq!(
        crate::openvino::redact_user_paths("/root/model.xml: permission denied"),
        "<home>/model.xml: permission denied"
    );
    assert_eq!(
        crate::openvino::redact_user_paths("/home/alice: I/O error on /Users/bob/run"),
        "<home>: I/O error on <home>/run"
    );
}
