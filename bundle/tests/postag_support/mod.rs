//! Small native parameters exercise bundle mechanics, not trained accuracy.
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
use turbo_bundle::postag::{self, NativeSource};
use turbo_postag::neural::{
    BilstmPosModel,
    bilstm_model::{UNKNOWN, char_vocabulary_from_strs},
    lstm::LstmLayer,
};

pub struct Scratch(pub PathBuf);
impl Scratch {
    pub fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "turbo-native-postag-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!path.exists());
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        if self.0.exists() {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
}

pub fn source() -> NativeSource {
    NativeSource { id: "turbo-native:bundle-regression".into(), revision: "parameters-v1".into() }
}
fn vector(n: usize, seed: usize) -> Vec<f64> {
    (0..n).map(|i| (((i + seed) % 19) as f64 - 9.0) * 0.017 + 0.003141592653).collect()
}
fn matrix(rows: usize, cols: usize, seed: usize) -> Vec<Vec<f64>> {
    (0..rows).map(|r| vector(cols, seed + r * cols)).collect()
}
fn layer(input: usize, hidden: usize, seed: usize) -> LstmLayer {
    LstmLayer::of_weights(
        input,
        hidden,
        matrix(4 * hidden, input, seed),
        matrix(4 * hidden, hidden, seed + 3),
        vector(4 * hidden, seed + 7),
    )
    .unwrap()
}

pub fn model(stacked: bool, crf: bool, pretrained: bool, adapter: bool) -> BilstmPosModel {
    assert!(!adapter || pretrained);
    let words = [UNKNOWN, "zoë", "李明"].into_iter().enumerate().map(|(i, s)| (s.into(), i)).collect();
    let chars =
        char_vocabulary_from_strs([UNKNOWN, "z", "o", "ë", "李", "明"].into_iter().enumerate().map(|(i, s)| (s, i)));
    BilstmPosModel::new(
        words,
        chars,
        vec!["NOUN".into(), "VERB".into(), "ADJ".into()],
        matrix(3, 3, 1),
        matrix(6, 2, 2),
        layer(2, 2, 3),
        layer(2, 2, 5),
        layer(7 + usize::from(pretrained) * 2, 3, 7),
        layer(7 + usize::from(pretrained) * 2, 3, 11),
        stacked.then(|| layer(6, 3, 13)),
        stacked.then(|| layer(6, 3, 17)),
        matrix(3, 6, 19),
        vector(3, 23),
        8,
        pretrained.then(|| vec![("zoë".into(), 0), ("李明".into(), 1)]),
        pretrained
            .then(|| matrix(2, 2, 29).into_iter().map(|row| row.into_iter().map(|x| x as f32).collect()).collect()),
        crf.then(|| matrix(3, 3, 31)),
        crf.then(|| vector(3, 37)),
        crf.then(|| vector(3, 41)),
        adapter.then(|| matrix(2, 2, 43)),
        adapter.then(|| vector(2, 47)),
    )
    .unwrap()
}

pub fn map_values(model: &BilstmPosModel, mut map: impl FnMut(&str, f64) -> f64) -> BilstmPosModel {
    let mut tensors = model.named_tensors();
    for t in &mut tensors {
        for value in &mut t.values {
            *value = map(&t.name, *value);
        }
    }
    BilstmPosModel::from_named(model.vocabularies(), |key| {
        tensors.iter().find(|t| t.name == key).map(|t| (t.shape.clone(), t.values.clone()))
    })
    .unwrap()
}

pub fn export(model: &BilstmPosModel) -> Scratch {
    let dir = Scratch::new();
    postag::export(model, &source(), "Apache-2.0", &dir.0).unwrap();
    dir
}

pub fn edit_manifest(dir: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let path = dir.join(postag::MANIFEST);
    let mut value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    edit(&mut value);
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

pub fn edit_tensor_header(dir: &Path, name: &str, edit: impl FnOnce(&mut serde_json::Value)) {
    let path = dir.join(name);
    let bytes = fs::read(&path).unwrap();
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
    edit(&mut header);
    let mut header = serde_json::to_vec(&header).unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut changed = (header.len() as u64).to_le_bytes().to_vec();
    changed.extend(header);
    changed.extend_from_slice(&bytes[8 + len..]);
    fs::write(path, &changed).unwrap();
    edit_manifest(dir, |m| m["files"][name] = turbo::deberta::package::digest(&changed).into());
}
