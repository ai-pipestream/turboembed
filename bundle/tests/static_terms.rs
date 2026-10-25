//! A static model distilled from the small BERT bundle with no recipe
//! file, its vocabulary grown by learned terms: each term one more entry
//! after the base's ids, which the sealed bundle's own tokenizer gives
//! for the word and the library serves.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_bundle::distill::from_base::{self, Options};
use turbo_bundle::distill::{Progress, Refusal};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-bundle-terms-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn options() -> Options {
    Options {
        id: "turbo-testdata/tiny-bert-static-terms".into(),
        revision: "1".into(),
        pca_dims: 16,
        sif_coefficient: 1e-4,
        dtype: "F32".into(),
        max_length: 32,
    }
}

const BASE_VOCAB: usize = 30522;

#[test]
fn learned_terms_become_entries_the_sealed_bundle_tokenizes_and_serves() {
    let d = scratch("terms");
    let base = root().join("testdata/tiny-bert-bundle");
    let mut recipe = from_base::recipe(&base, &options(), &d.join("work")).unwrap();
    let bundle = d.join("bundle");
    let terms: Vec<String> =
        ["neuroimaging", "turboembed", "the", "two words", "Neuroimaging", "\u{1F642}", "", "##ing"]
            .map(String::from)
            .to_vec();
    let mut seen = Vec::new();
    let staged = turbo_bundle::distill::stage_with(&recipe, &base, &bundle, &terms, &mut |p| seen.push(p)).unwrap();
    assert_eq!(staged.added, ["neuroimaging", "turboembed"]);
    let refused: Vec<(&str, Refusal)> = staged.refused.iter().map(|(t, r)| (t.as_str(), *r)).collect();
    assert_eq!(
        refused,
        [
            ("the", Refusal::Present),
            ("two words", Refusal::Words),
            ("Neuroimaging", Refusal::Duplicate),
            ("\u{1F642}", Refusal::Unknown),
            ("", Refusal::Empty),
            ("##ing", Refusal::Words),
        ]
    );
    let Progress::Kept { added, refused, vocab_size, .. } = seen[0] else { panic!("{seen:?}") };
    assert_eq!((added, refused, vocab_size), (2, 6, BASE_VOCAB));
    let Some(Progress::Embedded { done, total }) = seen.iter().rev().find(|p| matches!(p, Progress::Embedded { .. }))
    else {
        panic!("{seen:?}")
    };
    assert_eq!(done, total, "the last embedded count is every row");
    assert!(matches!(seen.last(), Some(Progress::Measured { .. })), "{seen:?}");
    assert!(seen.iter().any(|p| matches!(p, Progress::Reduced { pca_dims: 16, .. })));

    turbo_bundle::distill::seal(&mut recipe, &bundle).unwrap();
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["static_embedding"]["vocab_size"], BASE_VOCAB + 2);
    assert_eq!(m["model"]["license"], "Apache-2.0");
    let args = m["artifacts"][0]["produced_by"]["args"].as_array().unwrap();
    assert!(args.iter().any(|a| a == "learned_terms=2"), "{args:?}");

    let b = Bundle::open(&bundle).unwrap();
    let tok = Tokenizer::load(&b).unwrap();
    let e =
        Encode { add_special_tokens: true, truncation: Truncation::Right, max_tokens: 32, prompt: PromptRole::None };
    assert_eq!(tok.encode("Neuroimaging", e).unwrap(), [BASE_VOCAB as i32]);
    assert_eq!(tok.encode("turboembed rocks", e).unwrap()[0], BASE_VOCAB as i32 + 1);
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn a_recipe_made_from_a_bundle_without_an_encoder_is_refused() {
    let d = scratch("no-encoder");
    let e = from_base::recipe(&root().join("testdata/tiny-static-bundle"), &options(), &d).unwrap_err();
    assert!(e.contains("architecture"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

/// `from` copied to `to`, every file and directory.
fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dest);
        } else {
            fs::copy(e.path(), dest).unwrap();
        }
    }
}

#[test]
fn the_bases_licence_text_goes_with_the_static_model() {
    let d = scratch("licence");
    // The small BERT bundle with a licence text of its own.
    let base = d.join("base");
    copy_dir(&root().join("testdata/tiny-bert-bundle"), &base);
    let text = b"Synthetic licence text for this test.\n";
    fs::write(base.join("LICENSE"), text).unwrap();
    let mut m: Value = serde_json::from_slice(&fs::read(base.join("manifest.json")).unwrap()).unwrap();
    m["model"]["license_file"] = "LICENSE".into();
    m["files"].as_array_mut().unwrap().push(serde_json::json!({
        "path": "LICENSE",
        "size": text.len(),
        "sha256": turbo::bundle::sha256_hex(text),
    }));
    fs::write(base.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();

    let mut recipe = from_base::recipe(&base, &options(), &d.join("work")).unwrap();
    assert_eq!(recipe.manifest["model"]["license_file"], "LICENSE");
    let bundle = d.join("bundle");
    turbo_bundle::distill::stage_with(&recipe, &base, &bundle, &[], &mut |_| {}).unwrap();
    turbo_bundle::distill::seal(&mut recipe, &bundle).unwrap();
    assert_eq!(fs::read(bundle.join("LICENSE")).unwrap(), text);
    let sealed: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(sealed["model"]["license_file"], "LICENSE");
    assert!(sealed["files"].as_array().unwrap().iter().any(|f| f["path"] == "LICENSE"), "{}", sealed["files"]);
    Bundle::open(&bundle).unwrap();

    // A licence path that leaves the bundle is refused.
    m["model"]["license_file"] = "../LICENSE".into();
    fs::write(base.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    let e = from_base::recipe(&base, &options(), &d.join("work2")).unwrap_err();
    assert!(e.contains("relative path"), "{e}");
    fs::remove_dir_all(d).unwrap();
}
