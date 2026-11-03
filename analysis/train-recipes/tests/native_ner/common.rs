//! Project-authored text and generated tiny-BERT weights, not MASC text
//! or a pretrained accuracy reference. All files are local test inputs.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use turbo::head::tagger::tensors;
use turbo::manifest::HeadSpec;
use turbo_train_recipes::native_ner::head;
use turbo_trainer::Tagger;

pub const TEXT: &str = "Ánn met Bob in Paris at Acme on May 🦀 Zoë .";

pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "native-ner-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

pub fn corpus(root: &Path) {
    for n in 0..30 {
        document(root, n, "person", "0");
    }
}

pub fn document(root: &Path, n: usize, kind: &str, first_entity: &str) {
    let stem = format!("doc{n:03}");
    let mut seg = "<graph>".to_string();
    let mut penn = "<graph>".to_string();
    let mut byte = 0;
    for (i, word) in TEXT.split(' ').enumerate() {
        let start = TEXT[..byte].encode_utf16().count();
        let end = start + word.encode_utf16().count();
        seg.push_str(&format!(r#"<region xml:id="seg-r{i}" anchors="{start} {end}"/>"#));
        penn.push_str(&format!(r#"<node xml:id="penn-n{i}"><link targets="seg-r{i}"/></node>"#));
        byte += word.len() + 1;
    }
    seg.push_str("</graph>");
    penn.push_str("</graph>");
    let mut ne = "<graph>".to_string();
    for (i, (label, indices)) in
        [(kind, first_entity), ("person", "2"), ("location", "4"), ("org", "6"), ("date", "8"), ("person", "10")]
            .iter()
            .enumerate()
    {
        ne.push_str(&format!(r#"<a ref="ne-n{i}" label="{label}"/>"#));
        for token in indices.split(' ') {
            ne.push_str(&format!(r#"<edge from="ne-n{i}" to="penn-n{token}"/>"#));
        }
    }
    ne.push_str("</graph>");
    let s = format!(r#"<graph><region anchors="0 {}"/></graph>"#, TEXT.encode_utf16().count());
    let mut header = format!(r#"<documentHeader><primaryData f.id="f.text" loc="{stem}.txt"/><annotations>"#);
    for (id, suffix, data) in [("f.seg", "seg", seg), ("f.s", "s", s), ("f.penn", "penn", penn), ("f.ne", "ne", ne)] {
        header.push_str(&format!(r#"<annotation f.id="{id}" loc="{stem}-{suffix}.xml"/>"#));
        fs::write(root.join(format!("{stem}-{suffix}.xml")), data).unwrap();
    }
    header.push_str("</annotations></documentHeader>");
    fs::write(root.join(format!("{stem}.hdr")), header).unwrap();
    fs::write(root.join(format!("{stem}.txt")), TEXT).unwrap();
}

pub fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

/// Test-only standard bundle export, following the existing tagger tests.
/// Production serialization will use the shared trainer packaging API.
pub fn trained_bundle(model: &Tagger<f32>, out: &Path) {
    let base = tiny();
    for file in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(out.join(file).parent().unwrap()).unwrap();
        fs::copy(base.join(file), out.join(file)).unwrap();
    }
    let src = fs::read(base.join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    header.remove("__metadata__");
    let mut data = src[8 + n..].to_vec();
    let spec: HeadSpec = head();
    for ((suffix, shape), (name, values)) in tensors(&spec, 32).into_iter().zip(model.tensors()) {
        assert_eq!(suffix, name);
        let start = data.len();
        data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(
            format!("{}.{}", spec.tensor_prefix, suffix),
            json!({"dtype":"F32", "shape":shape, "data_offsets":[start,data.len()]}),
        );
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut weights = (h.len() as u64).to_le_bytes().to_vec();
    weights.extend(h);
    weights.extend(data);
    fs::create_dir_all(out.join("weights")).unwrap();
    fs::write(out.join("weights/model.safetensors"), &weights).unwrap();
    let mut m: serde_json::Value = serde_json::from_slice(&fs::read(base.join("manifest.json")).unwrap()).unwrap();
    m["heads"] = json!([spec]);
    // This generated tiny encoder and this generated corpus are project fixtures.
    m["model"]["license_class"] = json!(turbo::manifest::LicenseClass::Permissive);
    // The base fixture has historical source labels; this test does not
    // use those labels as pretrained-model provenance or quality evidence.
    for f in m["files"].as_array_mut().unwrap() {
        if f["path"] == "weights/model.safetensors" {
            f["size"] = json!(weights.len());
            f["sha256"] = json!(turbo::bundle::sha256_hex(&weights));
        }
    }
    fs::write(out.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
}
