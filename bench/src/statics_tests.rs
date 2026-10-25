use super::*;

#[test]
fn a_peers_rates_are_read_from_its_lines() {
    let out = "statembed+rayon: loaded in 3.1 ms\nstatembed+rayon batch 1: 51234 texts/s\n\
               statembed+rayon batch 32: 120000 texts/s\nsomething else\nx batch y: 3 texts/s\n";
    let r = rates("t60", out, &[1, 32]);
    assert_eq!(r.len(), 2);
    assert_eq!(r[0], Rate { set: "t60".into(), batch: 1, texts_per_second: 51234.0 });
    assert_eq!(r[1].batch, 32);
    assert!(rates("golden", "model2vec-zig: not run", &[1]).is_empty());
    // One line a batch, in order; `all` stands for any batch above 1.
    let zig = "model2vec-zig batch 1: 900 texts/s\nmodel2vec-zig batch all (32 threads): 4000 texts/s\n\
               model2vec-zig batch all (32 threads): 4100 texts/s\n";
    let r = rates("golden", zig, &[1, 32, 256]);
    assert_eq!(
        r.iter().map(|r| (r.batch, r.texts_per_second)).collect::<Vec<_>>(),
        [(1, 900.0), (32, 4000.0), (256, 4100.0)]
    );
    assert!(rates("golden", "x batch all (2 threads): 5 texts/s\n", &[1]).is_empty());
}

#[test]
fn vectors_agree_by_bits_cosine_and_refusals() {
    let want = [1.0f32, 0.0, 0.6, 0.8, 0.0, 0.0, 0.5, 0.5];
    let got: Vec<f32> = vec![1.0, 0.0, 0.8, 0.6, 0.0, 0.0, f32::NAN, f32::NAN];
    let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
    let a = agreement("golden", &want, &bytes, 2).unwrap();
    assert_eq!((a.rows, a.bit_equal, a.refused), (4, 2, 1));
    assert!((a.min_cosine - 0.96).abs() < 1e-6, "{}", a.min_cosine);
    assert!(agreement("golden", &want, &bytes[4..], 2).is_err());
}

#[test]
fn one_zero_row_has_zero_agreement_and_two_zero_rows_agree() {
    let want = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
    let got = [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0];
    let bytes: Vec<_> = got.into_iter().flat_map(f32::to_le_bytes).collect();
    let a = agreement("golden", &want, &bytes, 2).unwrap();
    assert_eq!((a.rows, a.bit_equal, a.refused, a.min_cosine), (3, 1, 0, 0.0));
    let zeros = agreement("golden", &[0.0, 0.0], &[0; 8], 2).unwrap();
    assert_eq!((zeros.bit_equal, zeros.refused, zeros.min_cosine), (1, 0, 1.0));
}

#[test]
fn nonfinite_peer_rows_are_refused() {
    let want = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let got = [f32::INFINITY, 0.0, f32::NEG_INFINITY, 0.0, f32::NAN, 0.0, 0.0, 0.0];
    let bytes: Vec<_> = got.into_iter().flat_map(f32::to_le_bytes).collect();
    let a = agreement("golden", &want, &bytes, 2).unwrap();
    assert_eq!((a.rows, a.bit_equal, a.refused, a.min_cosine), (4, 1, 3, 1.0));
}

/// Each committed record parses and passes `check`, and is filed under
/// the name `file_name` gives it.
#[test]
fn committed_records_pass_their_check() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(STATIC_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    for e in entries {
        let path = e.unwrap().path();
        let r: StaticRecord = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        check(&r).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), file_name(&r));
    }
}
