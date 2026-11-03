//! Native export coverage uses constructed parameters and explicit precision rules.
mod postag_support;
use postag_support::*;
use std::fs;
use turbo_bundle::postag;
use turbo_postag::neural::{BilstmPosTagger, Numerics};

#[test]
fn native_variants_preserve_vocabularies_and_explicitly_narrow_every_tensor() {
    for flags in [
        (false, false, false, false),
        (true, false, false, false),
        (false, true, false, false),
        (false, false, true, false),
        (false, false, true, true),
        (true, true, true, true),
    ] {
        let native = model(flags.0, flags.1, flags.2, flags.3);
        assert!(native.named_tensors().iter().flat_map(|t| &t.values).all(|x| *x != 0.0 && x.is_finite()));
        let dir = export(&native);
        let manifest = postag::verify(&dir.0).unwrap();
        assert_eq!(manifest.version, 2);
        assert_eq!(manifest.native, source());
        assert_eq!(manifest.license, "Apache-2.0");
        assert_eq!(manifest.gate_order, "i,f,g,o");
        assert_eq!((manifest.stacked, manifest.crf, manifest.adapter), (flags.0, flags.1, flags.3));
        assert_eq!(manifest.files.len(), 3);
        for (file, sha) in &manifest.files {
            assert_eq!(*sha, turbo::deberta::package::digest(&fs::read(dir.0.join(file)).unwrap()));
        }
        for f16 in [false, true] {
            let loaded = postag::load(&dir.0, f16).unwrap();
            let rounded =
                map_values(&native, |_, x| if f16 { half::f16::from_f64(x).to_f64() } else { f64::from(x as f32) });
            assert_eq!(loaded.vocabularies(), native.vocabularies());
            assert_eq!(loaded.named_tensors(), rounded.named_tensors());
            assert_ne!(native.named_tensors(), rounded.named_tensors());
            let expected = BilstmPosTagger::with_numerics(rounded, Numerics::Packed).unwrap();
            let actual = BilstmPosTagger::with_numerics(loaded, Numerics::Packed).unwrap();
            for tokens in [vec!["zoë", "李明", "unseen"], vec!["李明"], vec!["unseen", "zoë", "zoë", "李明"]] {
                let emissions = actual.emissions(&tokens).unwrap();
                assert_eq!(emissions.len(), tokens.len());
                assert!(emissions.iter().all(|row| row.len() == 3 && row.iter().all(|x| x.is_finite())));
                assert_eq!(emissions, expected.emissions(&tokens).unwrap());
                assert_eq!(actual.tag(&tokens).unwrap(), expected.tag(&tokens).unwrap());
            }
        }
    }
}

#[test]
fn f16_overflow_is_refused_before_creating_a_bundle() {
    for value in [70_000.0, -70_000.0, 65_520.0, -65_520.0] {
        let native =
            map_values(&model(false, false, false, false), |name, x| if name == "output_bias" { value } else { x });
        let dir = Scratch::new();
        let err = postag::export(&native, &source(), "Apache-2.0", &dir.0).unwrap_err();
        assert!(err.contains("output_bias") && err.contains("not a finite F16"), "{err}");
        assert!(!dir.0.exists());
    }
}

#[test]
fn f32_overflow_is_refused_before_creating_a_bundle() {
    for value in [f64::MAX, -f64::MAX] {
        let native =
            map_values(&model(false, false, false, false), |name, x| if name == "output_bias" { value } else { x });
        let dir = Scratch::new();
        let err = postag::export(&native, &source(), "Apache-2.0", &dir.0).unwrap_err();
        assert!(err.contains("output_bias") && err.contains("not a finite F32"), "{err}");
        assert!(!dir.0.exists());
    }
}

#[test]
fn finite_f16_boundary_values_remain_supported() {
    for value in [65_504.0, -65_504.0, 65_519.0, -65_519.0] {
        let native =
            map_values(&model(false, false, false, false), |name, x| if name == "output_bias" { value } else { x });
        let dir = export(&native);
        let loaded = postag::load(&dir.0, true).unwrap();
        let bias = loaded.named_tensors().into_iter().find(|t| t.name == "output_bias").unwrap();
        assert_eq!(bias.values, vec![value.signum() * 65_504.0; 3]);
        let f32 = postag::load(&dir.0, false).unwrap();
        assert_eq!(f32.named_tensors().into_iter().find(|t| t.name == "output_bias").unwrap().values, vec![value; 3]);
    }
}

#[test]
fn existing_directory_is_not_overwritten() {
    let dir = Scratch::new();
    fs::create_dir(&dir.0).unwrap();
    fs::write(dir.0.join("keep"), b"existing contents").unwrap();
    let err = postag::export(&model(false, false, false, false), &source(), "Apache-2.0", &dir.0).unwrap_err();
    assert!(err.contains("new directory"), "{err}");
    assert_eq!(fs::read(dir.0.join("keep")).unwrap(), b"existing contents");
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
}

#[test]
fn digests_are_checked_for_every_bundle_file() {
    for file in ["weights.safetensors", "weights-f16.safetensors", "vocab.json"] {
        let dir = export(&model(false, false, false, false));
        let path = dir.0.join(file);
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(path, bytes).unwrap();
        let err = postag::verify(&dir.0).unwrap_err();
        assert!(err.contains(file) && err.contains("SHA-256"), "{err}");
        assert!(postag::load(&dir.0, false).is_err());
        assert!(postag::load(&dir.0, true).is_err());
    }
}

#[test]
fn resealed_wrong_tensor_dtype_and_shape_are_refused() {
    for (f16, file, dtype) in [(false, "weights.safetensors", "I32"), (true, "weights-f16.safetensors", "I16")] {
        let dir = export(&model(false, false, false, false));
        edit_tensor_header(&dir.0, file, |h| h["output_bias"]["dtype"] = dtype.into());
        postag::verify(&dir.0).unwrap();
        let err = postag::load(&dir.0, f16).err().unwrap();
        assert!(err.contains("output_bias") && err.contains(dtype), "{err}");
        let dir = export(&model(false, false, false, false));
        edit_tensor_header(&dir.0, file, |h| h["output_bias"]["shape"] = serde_json::json!([1, 3]));
        postag::verify(&dir.0).unwrap();
        let err = postag::load(&dir.0, f16).err().unwrap();
        assert!(err.contains("output_bias") && err.contains("shape"), "{err}");
    }
}

#[test]
fn manifest_flags_must_match_optional_tensors() {
    for enabled in [false, true] {
        for flag in ["crf", "stacked", "adapter"] {
            let dir = export(&model(enabled, enabled, enabled, enabled));
            edit_manifest(&dir.0, |m| m[flag] = (!enabled).into());
            for f16 in [false, true] {
                let err = postag::load(&dir.0, f16).err().unwrap();
                assert!(err.contains("model flags do not match"), "{err}");
            }
        }
    }
}

#[test]
fn native_identity_and_license_are_required_before_export() {
    for (id, revision, license) in [
        (" ", "v1", "Apache-2.0"),
        ("native:test", " ", "Apache-2.0"),
        ("opennlp:retired", "v1", "Apache-2.0"),
        ("native:test", "v1", " "),
    ] {
        let dir = Scratch::new();
        let source = postag::NativeSource { id: id.into(), revision: revision.into() };
        assert!(postag::export(&model(false, false, false, false), &source, license, &dir.0).is_err());
        assert!(!dir.0.exists());
    }
}

#[test]
fn version_one_archive_manifests_are_refused() {
    let dir = export(&model(false, false, false, false));
    edit_manifest(&dir.0, |m| m["version"] = 1.into());
    let err = postag::verify(&dir.0).unwrap_err();
    assert!(err.contains("version 2 native bundle"), "{err}");
}

#[test]
fn native_directory_verification_works_without_an_archive_argument() {
    let dir = export(&model(true, true, true, true));
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-bundle"))
        .arg("postag-verify")
        .arg(&dir.0)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn archive_bytes_cannot_create_a_bundle() {
    for bytes in [b"".as_slice(), b"ONLP-BLPT-5", b"arbitrary input"] {
        let dir = Scratch::new();
        let err = postag::convert(bytes, "Apache-2.0", &dir.0).unwrap_err();
        assert!(err.contains("retired"), "{err}");
        assert!(!dir.0.exists());
    }
}
