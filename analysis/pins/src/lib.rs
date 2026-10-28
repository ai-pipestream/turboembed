//! The pins of every model a test, gate, evaluation or record loads: the
//! SHA-256 of its bytes, written as its key in the content-addressed model
//! store of #1613 (`blobs/sha256/<hex>`). The pins are `models.txt` beside
//! this crate, every file of the zoo index, and the score-only reference
//! models the training recipes list; all three are compiled in, so a check
//! needs no file but the model's own.
//!
//! A test that loads a model from a path it is given (an environment
//! variable, a fetched directory) passes the path through [`require`],
//! which hashes the model and refuses it unless that digest is pinned, so a
//! changed or swapped file fails loudly instead of being measured. The
//! tree test of this crate checks that every such test does.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// The pin list beside this crate.
pub const MODELS: &str = include_str!("../models.txt");
/// The zoo index, whose files are pinned by size, SHA-256 and SHA-512.
const ZOO: &str = include_str!("../../zoo/index/zoo.json");
/// The score-only reference models of the training recipes.
const REFERENCES: [&str; 4] = [
    include_str!("../../train-recipes/masc-references.txt"),
    include_str!("../../train-recipes/ud-references.txt"),
    include_str!("../../train-recipes/tatoeba-references.txt"),
    include_str!("../../train-recipes/fineweb-references.txt"),
];
/// The key prefix of a pin, as the model store names a blob.
pub const BLOB_PREFIX: &str = "blobs/sha256/";
/// The manifests a directory is pinned by, in the order they are looked
/// for, and the kind each makes it.
const MANIFESTS: [(&str, Kind); 4] = [
    ("manifest.json", Kind::Bundle),
    ("gliner.json", Kind::Package),
    ("glirel.json", Kind::Package),
    ("deberta.json", Kind::Package),
];

/// What the digest of a pin is over (models.txt says the same).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A bundle's manifest.json.
    Bundle,
    /// A GLiNER, GLiREL or DeBERTa package's own manifest.
    Package,
    /// One model file.
    File,
    /// A directory with no manifest: the digest of its listing.
    Tree,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Bundle => "bundle",
            Kind::Package => "package",
            Kind::File => "file",
            Kind::Tree => "tree",
        }
    }

    fn parse(s: &str) -> Option<Kind> {
        [Kind::Bundle, Kind::Package, Kind::File, Kind::Tree].into_iter().find(|k| k.name() == s)
    }
}

/// One pinned model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pin {
    /// The SHA-256, 64 lower case hex digits.
    pub sha256: &'static str,
    pub kind: Kind,
    /// A name for people: a model id, a file name, a path.
    pub name: &'static str,
    /// Where the pin is written: models.txt, the zoo index, or a
    /// references file.
    pub source: &'static str,
}

/// Every pin, models.txt first.
pub fn pins() -> &'static [Pin] {
    static PINS: OnceLock<Vec<Pin>> = OnceLock::new();
    PINS.get_or_init(|| {
        let mut all = parse_models(MODELS).unwrap_or_else(|e| panic!("analysis/pins/models.txt: {e}"));
        all.extend(zoo_pins(ZOO));
        for text in REFERENCES {
            all.extend(reference_pins(text));
        }
        all
    })
}

/// The pin of the digest `sha256`, if any.
pub fn lookup(sha256: &str) -> Option<&'static Pin> {
    pins().iter().find(|p| p.sha256 == sha256)
}

/// The pin named `name` in models.txt, if any.
pub fn named(name: &str) -> Option<&'static Pin> {
    pins().iter().find(|p| p.source == "models.txt" && p.name == name)
}

/// The lines of models.txt as pins; an ill-formed line is an error.
pub fn parse_models(text: &'static str) -> Result<Vec<Pin>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let bad = |why: &str| format!("line {}: {why}: {line}", i + 1);
        let mut parts = line.splitn(3, "  ");
        let (Some(key), Some(kind), Some(name)) = (parts.next(), parts.next(), parts.next()) else {
            return Err(bad("not `blobs/sha256/<hex>  <kind>  <name>`"));
        };
        let sha256 = key.strip_prefix(BLOB_PREFIX).filter(|h| is_hex64(h)).ok_or_else(|| bad("not a blob key"))?;
        let kind = Kind::parse(kind).ok_or_else(|| bad("unknown kind"))?;
        if name.trim().is_empty() || name != name.trim() {
            return Err(bad("the name is empty or padded"));
        }
        out.push(Pin { sha256, kind, name, source: "models.txt" });
    }
    Ok(out)
}

/// Every `"sha256": "<hex>"` of the zoo index, named by the `"path"`
/// written before it in the same file entry.
fn zoo_pins(text: &'static str) -> Vec<Pin> {
    const KEY: &str = "\"sha256\": \"";
    const PATH: &str = "\"path\": \"";
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = text[from..].find(KEY) {
        let start = from + at + KEY.len();
        from = start;
        let Some(hex) = text.get(start..start + 64).filter(|h| is_hex64(h)) else { continue };
        let name =
            text[..start].rfind(PATH).and_then(|p| text[p + PATH.len()..].split('"').next()).unwrap_or("zoo file");
        out.push(Pin { sha256: hex, kind: Kind::File, name, source: "analysis/zoo/index/zoo.json" });
    }
    out
}

/// The `<sha256> <address>` lines of a references file.
fn reference_pins(text: &'static str) -> Vec<Pin> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (hex, url) = l.split_once(' ')?;
            is_hex64(hex).then(|| Pin {
                sha256: hex,
                kind: Kind::File,
                name: url.rsplit('/').next().unwrap_or(url),
                source: "analysis/train-recipes/*-references.txt",
            })
        })
        .collect()
}

pub fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    turbo_sha256::sha256(bytes).to_hex()
}

/// The kind and digest of the model at `path`: a file's bytes; a
/// directory's manifest (MANIFESTS) when it has one, otherwise its
/// listing, one `<sha256>  <relative path>\n` per regular file in byte
/// order of the path.
pub fn digest(path: &Path) -> Result<(Kind, String), String> {
    measure(path).map(|(kind, hex, _)| (kind, hex))
}

/// A model's kind and digest, and for a tree each file's relative path and
/// digest.
type Measured = (Kind, String, Vec<(String, String)>);

/// [`digest`], and for a tree the digest of each of its files.
fn measure(path: &Path) -> Result<Measured, String> {
    let meta = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.is_file() {
        return Ok((Kind::File, file_digest(path)?, Vec::new()));
    }
    for (name, kind) in MANIFESTS {
        let m = path.join(name);
        if m.is_file() {
            return Ok((kind, file_digest(&m)?, Vec::new()));
        }
    }
    let mut files = Vec::new();
    list(path, path, &mut files)?;
    files.sort();
    let mut listing = String::new();
    let mut each = Vec::with_capacity(files.len());
    for rel in files {
        let hex = file_digest(&path.join(&rel))?;
        listing.push_str(&format!("{hex}  {rel}\n"));
        each.push((rel, hex));
    }
    Ok((Kind::Tree, sha256_hex(listing.as_bytes()), each))
}

fn file_digest(path: &Path) -> Result<String, String> {
    fs::read(path).map(|b| sha256_hex(&b)).map_err(|e| format!("{}: {e}", path.display()))
}

/// The regular files under `dir`, as paths relative to `root` with `/`
/// between their parts.
fn list(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|e| format!("{}: {e}", dir.display()))?.path();
        if path.is_dir() {
            list(root, &path, out)?;
        } else if path.is_file() {
            let rel = path.strip_prefix(root).map_err(|e| e.to_string())?;
            let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            out.push(parts.join("/"));
        }
    }
    Ok(())
}

/// Whether the model at `path` is pinned: its digest is pinned as its
/// kind, or, for a directory with no manifest (a checkpoint the zoo
/// fetched), every file in it is pinned as a file.
pub fn check(path: &Path) -> Result<(), String> {
    let (kind, hex, each) = measure(path)?;
    match lookup(&hex) {
        Some(p) if p.kind == kind => return Ok(()),
        Some(p) => {
            return Err(format!(
                "{}: its {} digest {hex} is pinned as a {} ({}), not as a {}",
                path.display(),
                kind.name(),
                p.kind.name(),
                p.name,
                kind.name()
            ));
        }
        None => {}
    }
    let unpinned: Vec<&str> = each
        .iter()
        .filter(|(_, h)| !lookup(h).is_some_and(|p| p.kind == Kind::File))
        .map(|(rel, _)| rel.as_str())
        .collect();
    if kind == Kind::Tree && !each.is_empty() && unpinned.is_empty() {
        return Ok(());
    }
    let files =
        if unpinned.is_empty() { String::new() } else { format!(" (files not pinned: {})", unpinned.join(", ")) };
    Err(format!(
        "{}: {} {hex} is not pinned{files}. A tested model must be pinned in analysis/pins/models.txt; \
         if this is the model meant, add `{}  {}  <name>` with the record or test that uses it \
         (`cargo run -p turbo-pins -- <path>` prints the line)",
        path.display(),
        kind.name(),
        line_key(&hex),
        kind.name()
    ))
}

/// `path` itself, once [`check`] passes; panics otherwise. A path checked
/// once in this process is not hashed again.
pub fn require(path: impl AsRef<Path>) -> PathBuf {
    static CHECKED: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
    let path = path.as_ref();
    let key = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if CHECKED.lock().unwrap_or_else(|e| e.into_inner()).contains(&key) {
        return path.to_path_buf();
    }
    if let Err(e) = check(path) {
        panic!("{e}");
    }
    CHECKED.lock().unwrap_or_else(|e| e.into_inner()).push(key);
    path.to_path_buf()
}

/// `root` itself, once each of the models `names` under it passes
/// [`require`]: a directory of bundles named by one variable.
pub fn require_in(root: impl AsRef<Path>, names: &[&str]) -> PathBuf {
    let root = root.as_ref();
    for name in names {
        require(root.join(name));
    }
    root.to_path_buf()
}

/// Each path of a list of paths joined by `,` or the platform's path
/// separator, as the multi-bundle variables name them, through [`require`].
pub fn require_list(list: &str) -> Vec<PathBuf> {
    list.split([',', if cfg!(windows) { ';' } else { ':' }]).filter(|s| !s.is_empty()).map(require).collect()
}

/// The models.txt key of a digest.
pub fn line_key(hex: &str) -> String {
    format!("{BLOB_PREFIX}{hex}")
}

#[cfg(test)]
mod tests;
