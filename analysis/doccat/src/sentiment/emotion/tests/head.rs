//! The emotion head on its own: its arithmetic and its binary format.

use super::*;

#[test]
fn each_label_is_the_sigmoid_of_its_row() {
    let h = head();
    let s = h.score(&[1.0, 0.0]).unwrap();
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    assert!((s.get(D::Fear) - sig(3.0)).abs() < 1e-12);
    assert!((s.get(D::Joy) - sig(-3.0)).abs() < 1e-12);
    assert!((s.get(D::Neutral) - 0.5).abs() < 1e-12);
    assert!(h.score(&[1.0]).is_err());
    assert!(h.score(&[f32::INFINITY, 0.0]).is_err());
    assert!(EmotionVectorHead::new(0, Vec::new(), [0.0; 28]).is_err());
    assert!(EmotionVectorHead::new(2, vec![0.0; 55], [0.0; 28]).is_err());
    assert!(EmotionVectorHead::new(2, vec![f32::NAN; 56], [0.0; 28]).is_err());
}

#[test]
fn the_bytes_round_trip_with_the_encoder_and_bad_bytes_are_errors() {
    for h in [head(), head().with_model(model())] {
        let bytes = h.to_bytes();
        let back = EmotionVectorHead::from_bytes(&bytes).unwrap();
        assert_eq!(back, h);
        assert_eq!(EmotionVectorScorer::model(&back), EmotionVectorScorer::model(&h));
        for n in 0..bytes.len() {
            assert!(EmotionVectorHead::from_bytes(&bytes[..n]).is_err(), "{n} bytes");
        }
        let mut long = bytes;
        long.push(0);
        assert!(EmotionVectorHead::from_bytes(&long).is_err());
    }
    // 16 header bytes, 28 biases, 56 weights.
    assert_eq!(head().to_bytes().len(), 16 + 4 * 84);
    let edit = |at: usize, v: &[u8]| {
        let mut b = head().to_bytes();
        b[at..at + v.len()].copy_from_slice(v);
        EmotionVectorHead::from_bytes(&b)
    };
    assert!(matches!(edit(0, b"TIVH"), Err(EmotionHeadError::Format(_))));
    assert!(edit(4, &2u32.to_le_bytes()).is_err());
    assert!(matches!(edit(8, &0u32.to_le_bytes()), Err(EmotionHeadError::Value(_))));
    assert!(edit(8, &u32::MAX.to_le_bytes()).is_err());
    assert!(edit(12, &1u32.to_le_bytes()).is_err());
    assert!(edit(12, &2u32.to_le_bytes()).is_err());
    assert!(edit(16, &f32::NAN.to_le_bytes()).is_err());
    let mut bad = head().with_model(model()).to_bytes();
    let at = 16 + 4 * 84 + 4;
    bad[at] = 0xFF;
    assert!(EmotionVectorHead::from_bytes(&bad).is_err());
}
