//! The learned head on its own: its arithmetic, its context matrix, its
//! binary format and its trainer.

use super::*;

/// A head over two values: label `l` reads value `l % 2` with weight 4.
fn two_wide() -> IncivilityVectorHead {
    let w: Vec<f32> = (0..6).flat_map(|l| if l % 2 == 0 { [4.0, 0.0] } else { [0.0, 4.0] }).collect();
    IncivilityVectorHead::new(2, w, [-2.0; 6]).unwrap()
}

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn each_label_is_the_sigmoid_of_its_row() {
    let h = two_wide();
    let s = h.score(&[1.0, 0.0], None).unwrap();
    assert!((s.insult - sig(2.0)).abs() < 1e-12, "{}", s.insult);
    assert!((s.profanity - sig(-2.0)).abs() < 1e-12);
    assert!((s.threat - sig(2.0)).abs() < 1e-12);
    assert!((s.severe_toxicity - sig(-2.0)).abs() < 1e-12);
    // Wrong width, or a value that is not finite: an error.
    assert!(h.score(&[1.0], None).is_err());
    assert!(h.score(&[f32::NAN, 0.0], None).is_err());
    // A head without a context matrix ignores a context vector.
    assert_eq!(h.score(&[1.0, 0.0], Some(&[9.0, 9.0])).unwrap(), s);
    // Through the scorer trait, with and without context.
    assert_eq!(h.score_incivility(&[1.0, 0.0]).unwrap(), s);
}

#[test]
fn the_context_matrix_moves_the_score_with_the_context() {
    // Insult reads the context's first value: a reply that is mild alone
    // reads as an insult after a hostile parent.
    let ctx: Vec<f32> = (0..6).flat_map(|l| if l == 0 { [5.0, 0.0] } else { [0.0, 0.0] }).collect();
    let h = two_wide().with_context(ctx).unwrap();
    let reply = [0.25, 0.0];
    let alone = h.score(&reply, None).unwrap().insult;
    let calm = h.score(&reply, Some(&[0.0, 1.0])).unwrap().insult;
    let hostile = h.score(&reply, Some(&[1.0, 0.0])).unwrap().insult;
    assert!(alone < 0.5 && calm < 0.5 && hostile > 0.5, "{alone} {calm} {hostile}");
    assert!(h.score(&reply, Some(&[1.0])).is_err());
    assert!(two_wide().with_context(vec![0.0; 3]).is_err());
}

#[test]
fn bad_weights_are_errors() {
    assert!(matches!(IncivilityVectorHead::new(0, Vec::new(), [0.0; 6]), Err(IncivilityHeadError::Value(_))));
    assert!(IncivilityVectorHead::new(2, vec![0.0; 11], [0.0; 6]).is_err());
    let mut w = vec![0.0; 12];
    w[7] = f32::INFINITY;
    assert!(IncivilityVectorHead::new(2, w, [0.0; 6]).is_err());
    assert!(IncivilityVectorHead::new(2, vec![0.0; 12], [0.0, 0.0, f32::NAN, 0.0, 0.0, 0.0]).is_err());
}

#[test]
fn the_bytes_round_trip_and_bad_bytes_are_errors() {
    let plain = two_wide();
    let ctx = two_wide().with_context((0..12).map(|i| i as f32 / 8.0).collect()).unwrap();
    for h in [&plain, &ctx] {
        let bytes = h.to_bytes();
        assert_eq!(&IncivilityVectorHead::from_bytes(&bytes).unwrap(), h);
        assert_eq!(bytes, IncivilityVectorHead::from_bytes(&bytes).unwrap().to_bytes());
        // Every cut of the bytes, and one more byte, is an error.
        for n in 0..bytes.len() {
            assert!(IncivilityVectorHead::from_bytes(&bytes[..n]).is_err(), "{n} bytes");
        }
        let mut long = bytes.clone();
        long.push(0);
        assert!(IncivilityVectorHead::from_bytes(&long).is_err());
    }
    // 16 header bytes, 6 biases, 12 weights.
    assert_eq!(plain.to_bytes().len(), 16 + 4 * 18);
    assert_eq!(ctx.to_bytes().len(), 16 + 4 * 30);
    let edit = |at: usize, v: &[u8]| {
        let mut b = plain.to_bytes();
        b[at..at + v.len()].copy_from_slice(v);
        IncivilityVectorHead::from_bytes(&b)
    };
    assert!(matches!(edit(0, b"XIVH"), Err(IncivilityHeadError::Format(_))));
    assert!(edit(4, &2u32.to_le_bytes()).is_err());
    assert!(matches!(edit(8, &0u32.to_le_bytes()), Err(IncivilityHeadError::Value(_))));
    assert!(edit(8, &u32::MAX.to_le_bytes()).is_err());
    assert!(edit(12, &2u32.to_le_bytes()).is_err());
    // A context flag with no context matrix is a length error.
    assert!(edit(12, &1u32.to_le_bytes()).is_err());
    assert!(edit(16, &f32::NAN.to_le_bytes()).is_err());
    assert!(edit(16 + 4 * 6, &f32::NEG_INFINITY.to_le_bytes()).is_err());
}

/// Labelled two-value vectors: value 0 is hostility, value 1 a neutral
/// topic. Insult and identity attack follow hostility, the rest are 0.
fn labelled() -> (Vec<[f32; 2]>, Vec<IncivilityScores>) {
    let mut v = Vec::new();
    let mut t = Vec::new();
    for k in 0..8 {
        let x = k as f32 / 8.0;
        v.push([1.0 + x, x]);
        t.push(IncivilityScores { insult: 1.0, identity_attack: 1.0, ..Default::default() });
        v.push([x / 4.0, 1.0 + x]);
        t.push(IncivilityScores::default());
    }
    (v, t)
}

#[test]
fn fitting_learns_the_labels_and_is_deterministic() {
    let (v, t) = labelled();
    let rows: Vec<&[f32]> = v.iter().map(|r| &r[..]).collect();
    let o = IncivilityFitOptions::default();
    let h = IncivilityVectorHead::fit(&rows, None, &t, &o).unwrap();
    for (r, want) in rows.iter().zip(&t) {
        let s = h.score(r, None).unwrap();
        for label in IncivilityLabel::ALL {
            assert_eq!(s.get(label) >= 0.5, want.get(label) >= 0.5, "{label} on {r:?}: {}", s.get(label));
        }
    }
    assert_eq!(h.to_bytes(), IncivilityVectorHead::fit(&rows, None, &t, &o).unwrap().to_bytes());
    // Bad inputs are errors.
    assert!(IncivilityVectorHead::fit(&[], None, &[], &o).is_err());
    assert!(IncivilityVectorHead::fit(&rows, None, &t[1..], &o).is_err());
    let bad_target = vec![IncivilityScores { threat: 1.5, ..Default::default() }; rows.len()];
    assert!(IncivilityVectorHead::fit(&rows, None, &bad_target, &o).is_err());
    let lr = IncivilityFitOptions { learning_rate: f64::NAN, ..o };
    assert!(IncivilityVectorHead::fit(&rows, None, &t, &lr).is_err());
}

#[test]
fn fitting_with_contexts_learns_what_the_parent_adds() {
    // The same reply vector, an insult only after a hostile parent.
    let reply = [0.5f32, 0.5];
    let (hostile, calm) = ([1.0f32, 0.0], [0.0f32, 1.0]);
    let mut rows: Vec<&[f32]> = Vec::new();
    let mut ctx: Vec<Option<&[f32]>> = Vec::new();
    let mut t = Vec::new();
    for _ in 0..4 {
        rows.extend([&reply[..], &reply[..], &reply[..]]);
        ctx.extend([Some(&hostile[..]), Some(&calm[..]), None]);
        t.extend([
            IncivilityScores { insult: 1.0, ..Default::default() },
            IncivilityScores::default(),
            IncivilityScores::default(),
        ]);
    }
    let h = IncivilityVectorHead::fit(&rows, Some(&ctx), &t, &IncivilityFitOptions::default()).unwrap();
    assert!(h.has_context());
    assert!(h.score(&reply, Some(&hostile)).unwrap().insult > 0.5);
    assert!(h.score(&reply, Some(&calm)).unwrap().insult < 0.5);
    assert!(h.score(&reply, None).unwrap().insult < 0.5);
}
