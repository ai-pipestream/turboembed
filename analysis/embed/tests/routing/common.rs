use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
use turbo::bundle::Bundle;

/// A real tiny transformer bundle with another identity and declared widths.
/// Artifact bytes and digests remain those of the checked-in fixture.
pub struct Variant {
    pub dir: PathBuf,
}
impl Variant {
    pub fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let source = root.join("testdata/tiny-bert-bundle");
        let dir = root.join("target/routing-tests").join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        let bundle = Bundle::open(&source).unwrap();
        for file in &bundle.manifest.files {
            let dest = dir.join(&file.path);
            fs::create_dir_all(dest.parent().unwrap()).unwrap();
            fs::write(dest, bundle.read_verified(&file.path).unwrap()).unwrap();
        }
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(source.join("manifest.json")).unwrap()).unwrap();
        manifest["model"]["id"] = "turbo-testdata/routing-variant".into();
        manifest["embed"]["output_dims"] = serde_json::json!([16, 32]);
        fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        Self { dir }
    }
}
impl Drop for Variant {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
