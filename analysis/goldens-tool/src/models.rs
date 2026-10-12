//! Fetching the models into a directory and checking their SHA-512.

use crate::command::run;
use crate::recipe::{MODELS, Model};
use crate::sha512::{Sha512, hex};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::process::Command;

/// Makes sure every model in the recipe is in `dir` with its published
/// SHA-512. A file already there with the right hash is kept; one that is
/// missing is downloaded with `curl`. A download whose hash differs is
/// removed and reported; a file already there whose hash differs is
/// reported and left for the caller to look at.
pub fn fetch(dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for m in MODELS {
        let target = dir.join(m.file);
        if target.exists() {
            check(&target, &m)?;
            println!("{}: present, SHA-512 matches", m.file);
            continue;
        }
        let part = dir.join(format!("{}.part", m.file));
        run(Command::new("curl")
            .args(["--fail", "--silent", "--show-error", "--location", "--proto", "=https", "--output"])
            .arg(&part)
            .arg(m.url))?;
        if let Err(e) = check(&part, &m) {
            let _ = fs::remove_file(&part);
            return Err(e);
        }
        fs::rename(&part, &target).map_err(|e| format!("{}: {e}", target.display()))?;
        println!("{}: fetched, SHA-512 matches", m.file);
    }
    Ok(())
}

/// Checks that every model in the recipe is in `dir` with its published SHA-512.
pub fn check_all(dir: &Path) -> Result<(), String> {
    MODELS.iter().try_for_each(|m| check(&dir.join(m.file), m))
}

/// Fails unless the file's SHA-512 is the model's.
fn check(path: &Path, m: &Model) -> Result<(), String> {
    let actual = sha512_file(path)?;
    if actual != m.sha512 {
        return Err(format!("{}: SHA-512 is {actual}, {} publishes {}", path.display(), m.url, m.sha512));
    }
    Ok(())
}

/// The SHA-512 of a file's contents, lowercase hex, read in pieces.
pub fn sha512_file(path: &Path) -> Result<String, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha512::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_with_another_hash_is_refused() {
        let dir = std::env::temp_dir().join(format!("turbo-goldens-models-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.bin");
        fs::write(&path, b"abc").unwrap();
        let m = Model { file: "model.bin", url: "https://example.org/model.bin", sha512: MODELS[0].sha512 };
        let err = check(&path, &m).unwrap_err();
        assert!(err.contains("ddaf35a193617aba"), "{err}");
        let ok = Model {
            sha512: "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
            ..m
        };
        check(&path, &ok).unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }
}
