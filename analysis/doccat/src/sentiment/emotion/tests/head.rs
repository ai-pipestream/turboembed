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

/// Labelled two-value vectors: value 0 separates fear, value 1 separates joy,
/// and all unactivated rows are neutral.
fn labelled() -> (Vec<[f32; 2]>, Vec<EmotionScores>) {
    let mut v = Vec::new();
    let mut t = Vec::new();
    for k in 0..8 {
        let x = k as f32 / 8.0;
        v.push([1.0 + x, 0.0]);
        let mut fear = EmotionScores::default();
        fear.set(D::Fear, 1.0);
        t.push(fear);

        v.push([0.0, 1.0 + x]);
        let mut joy = EmotionScores::default();
        joy.set(D::Joy, 1.0);
        t.push(joy);

        v.push([0.0, 0.0]);
        let mut neutral = EmotionScores::default();
        neutral.set(D::Neutral, 1.0);
        t.push(neutral);
    }
    (v, t)
}

#[test]
fn fitting_learns_the_labels_and_is_deterministic() {
    let (v, t) = labelled();
    let rows: Vec<&[f32]> = v.iter().map(|r| &r[..]).collect();
    let o = EmotionFitOptions { epochs: 80, learning_rate: 1.0, l2: 1e-4 };
    let h = EmotionVectorHead::fit(&rows, &t, &o).unwrap();
    for (r, want) in rows.iter().zip(&t) {
        let s = h.score(r).unwrap();
        for d in D::ALL {
            assert_eq!(s.get(d) >= 0.5, want.get(d) >= 0.5, "{d} on {r:?}: {}", s.get(d));
        }
    }
    assert_eq!(h.to_bytes(), EmotionVectorHead::fit(&rows, &t, &o).unwrap().to_bytes());
    assert!(EmotionVectorHead::fit(&[], &[], &o).is_err());
    assert!(EmotionVectorHead::fit(&rows, &t[1..], &o).is_err());
    let mut bad_target = t.clone();
    bad_target[0].set(D::Fear, 1.5);
    assert!(EmotionVectorHead::fit(&rows, &bad_target, &o).is_err());
    let none = EmotionFitOptions { epochs: 0, ..o };
    assert!(EmotionVectorHead::fit(&rows, &t, &none).is_err());
    let lr = EmotionFitOptions { learning_rate: f64::NAN, ..o };
    assert!(EmotionVectorHead::fit(&rows, &t, &lr).is_err());
}

#[test]
fn zoo_loader_resolves_bound_head_bytes_and_rejects_non_head_artifacts() {
    use turbo_zoo::{Acceptances, Index, Offline, Policy, Sha256, Sha512, UserId, Zoo};
    let dir = std::env::temp_dir().join(format!("turbo-emotion-zoo-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bound = head().with_model(model());
    let bytes = bound.to_bytes();
    let sha256 = Sha256::of(&bytes);
    let sha512 = Sha512::of(&bytes);
    let art_dir = dir.join(GOEMOTIONS_MINILM_HEAD);
    std::fs::create_dir_all(&art_dir).unwrap();
    std::fs::write(art_dir.join("head.tevh"), &bytes).unwrap();
    let index_json = format!(
        r#"{{
  "zoo_version": 1,
  "artifacts": [
    {{
      "id": "{id}",
      "kind": "KIND_HEAD",
      "component": "COMPONENT_DOCUMENT_CATEGORIZER",
      "language": "en",
      "format": "FORMAT_HEAD",
      "version": "1",
      "license": "Apache-2.0",
      "license_terms": "https://www.apache.org/licenses/LICENSE-2.0",
      "trained_on": [
        {{
          "name": "GoEmotions",
          "license": "CC-BY-4.0",
          "license_terms": "https://creativecommons.org/licenses/by/4.0/"
        }}
      ],
      "class": "CLASS_PERMISSIVE",
      "attribution": "Trained on GoEmotions (Demszky et al., ACL 2020, CC-BY-4.0).",
      "files": [
        {{
          "path": "head.tevh",
          "url": "https://example.org/models/{id}.tevh",
          "size": {len},
          "sha256": "{sha256}",
          "sha512": "{sha512}"
        }}
      ]
    }}
  ]
}}"#,
        id = GOEMOTIONS_MINILM_HEAD,
        len = bytes.len(),
        sha256 = sha256,
        sha512 = sha512,
    );
    let index = Index::parse(index_json.as_bytes()).unwrap();
    let zoo = Zoo::new(index, dir.clone(), Policy::default(), &Offline);
    let user: UserId = "test".parse().unwrap();
    let loaded =
        EmotionVectorHead::from_zoo(&zoo, GOEMOTIONS_MINILM_HEAD, &user, || Ok(Acceptances::default())).unwrap();
    assert_eq!(loaded, bound);
    assert!(matches!(
        EmotionVectorHead::from_zoo(&zoo, "missing-head", &user, || Ok(Acceptances::default())),
        Err(EmotionZooError::Zoo(_))
    ));
    let builtin_index = Index::builtin();
    let builtin = Zoo::new(builtin_index.clone(), dir.clone(), Policy::default(), &Offline);
    assert!(matches!(
        EmotionVectorHead::from_zoo(&builtin, "goemotions", &user, || Ok(Acceptances::default())),
        Err(EmotionZooError::NotAHead { .. })
    ));
    for id in [GOEMOTIONS_MINILM_HEAD, GOEMOTIONS_BGE_SMALL_HEAD] {
        let entry = builtin_index.get(id).unwrap();
        assert_eq!(entry.kind, turbo_zoo::Kind::Head);
        assert_eq!(entry.format, turbo_zoo::Format::Head);
        assert_eq!(entry.class, turbo_zoo::LicenceClass::Permissive);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
