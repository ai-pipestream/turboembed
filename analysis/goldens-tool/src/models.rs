//! Fetching the models and the MeCab dictionaries into a directory and
//! checking their SHA-512, and checking the SHA-512 of the checked in
//! fixtures.

use crate::command::run;
use crate::recipe::{
    CORPORA, DICTIONARIES, Dictionary, DictionaryEncoding, FIXTURES, HUNSPELL_FILES, LOCAL_MODELS, MODELS, Model,
    POOLING_SOURCE, UTF8_COPY_SUFFIX,
};
use crate::sha512::{Sha512, hex};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::process::Command;

/// Which models [`fetch`] makes sure of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Models {
    /// [`MODELS`], the ones CI fetches.
    Published,
    /// [`MODELS`] and [`LOCAL_MODELS`], every model the generator loads.
    WithLocal,
}

/// Makes sure every model of `which`, every MeCab dictionary, every
/// Hunspell dictionary file and every training corpus of the recipe is in
/// `dir` with its SHA-512. A file already there with the right hash is
/// kept; one that is missing is downloaded with `curl`. A download whose
/// hash differs is removed and reported; a file already there whose hash
/// differs is reported and left for the caller to look at.
pub fn fetch(dir: &Path, which: Models) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let local: &[Model] = match which {
        Models::Published => &[],
        Models::WithLocal => &LOCAL_MODELS,
    };
    let source: &[Model] = match which {
        Models::Published => &[],
        Models::WithLocal => std::slice::from_ref(&POOLING_SOURCE),
    };
    let hunspell: Vec<Model> =
        HUNSPELL_FILES.iter().map(|f| Model { file: f.file, url: f.url, sha512: f.sha512 }).collect();
    let corpora: Vec<Model> = CORPORA.iter().map(|c| Model { file: c.file, url: c.url, sha512: c.sha512 }).collect();
    for m in MODELS.iter().chain(local).chain(source).chain(&hunspell).chain(&corpora) {
        let target = dir.join(m.file);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        if target.exists() {
            check(&target, m)?;
            println!("{}: present, SHA-512 matches", m.file);
            continue;
        }
        let part = dir.join(format!("{}.part", m.file));
        run(Command::new("curl")
            .args(["--fail", "--silent", "--show-error", "--location", "--proto", "=https", "--output"])
            .arg(&part)
            .arg(m.url))?;
        if let Err(e) = check(&part, m) {
            let _ = fs::remove_file(&part);
            return Err(e);
        }
        fs::rename(&part, &target).map_err(|e| format!("{}: {e}", target.display()))?;
        println!("{}: fetched, SHA-512 matches", m.file);
    }
    for d in DICTIONARIES {
        fetch_dictionary(dir, &d)?;
    }
    if which == Models::WithLocal {
        crate::pooled::derive(dir)?;
    }
    Ok(())
}

/// Makes sure the dictionary's archive is in `dir` with its SHA-512,
/// downloading it as a model is downloaded, then unpacks it into
/// `dir/<d.dir>` unless that is there, and for a dictionary that is not
/// UTF-8 writes the UTF-8 copy into `dir/<d.dir>.utf8` unless that is
/// there. Unpacking and recoding go through a `.part` directory that is
/// renamed when complete, so an interrupted run leaves nothing that looks
/// finished. A directory already there is not checked here: the generator
/// writes the SHA-512 of every file it reads into the goldens' headers,
/// which `verify` compares.
fn fetch_dictionary(dir: &Path, d: &Dictionary) -> Result<(), String> {
    let archive = dir.join(d.archive);
    let m = Model { file: d.archive, url: d.url, sha512: d.sha512 };
    if archive.exists() {
        check(&archive, &m)?;
        println!("{}: present, SHA-512 matches", d.archive);
    } else {
        let part = dir.join(format!("{}.part", d.archive));
        run(Command::new("curl")
            .args(["--fail", "--silent", "--show-error", "--location", "--proto", "=https", "--output"])
            .arg(&part)
            .arg(d.url))?;
        if let Err(e) = check(&part, &m) {
            let _ = fs::remove_file(&part);
            return Err(e);
        }
        fs::rename(&part, &archive).map_err(|e| format!("{}: {e}", archive.display()))?;
        println!("{}: fetched, SHA-512 matches", d.archive);
    }
    let unpacked = dir.join(d.dir);
    if !unpacked.exists() {
        let part = fresh_dir(&dir.join(format!(".{}.part", d.dir)))?;
        run(Command::new("tar")
            .arg("-xzf")
            .arg(&archive)
            .arg("-C")
            .arg(&part)
            .args(["--no-same-owner", "--no-same-permissions"]))?;
        let inner = part.join(d.dir);
        if !inner.is_dir() {
            return Err(format!("{} does not unpack to {}", d.archive, d.dir));
        }
        fs::rename(&inner, &unpacked).map_err(|e| format!("{}: {e}", unpacked.display()))?;
        fs::remove_dir_all(&part).map_err(|e| format!("{}: {e}", part.display()))?;
        println!("{}: unpacked", d.dir);
    }
    if d.encoding == DictionaryEncoding::EucJp {
        let copy = dir.join(format!("{}{UTF8_COPY_SUFFIX}", d.dir));
        if !copy.exists() {
            let part = fresh_dir(&dir.join(format!(".{}{UTF8_COPY_SUFFIX}.part", d.dir)))?;
            for name in dictionary_files(&unpacked)? {
                let out = run(Command::new("iconv").args(["-f", "EUC-JP", "-t", "UTF-8"]).arg(unpacked.join(&name)))?;
                let text = String::from_utf8(out.stdout).map_err(|e| format!("{name}: iconv wrote bad UTF-8: {e}"))?;
                let text = as_java_decodes_euc_jp(&text);
                fs::write(part.join(&name), text.as_bytes())
                    .map_err(|e| format!("{}: {e}", part.join(&name).display()))?;
            }
            fs::rename(&part, &copy).map_err(|e| format!("{}: {e}", copy.display()))?;
            println!("{}: written in UTF-8", copy.display());
        }
    }
    Ok(())
}

/// `text`, which iconv decoded from EUC-JP, as Java's `EUC-JP` decoder
/// decodes the same bytes. The two differ on one code: 0xA1BD is U+2015
/// HORIZONTAL BAR to iconv and U+2014 EM DASH to Java (IPADIC has it in
/// Noun.csv, Noun.org.csv, Noun.proper.csv and Symbol.csv). The generator
/// checks the copy against what Java decodes, file by file, so any other
/// difference stops it.
fn as_java_decodes_euc_jp(text: &str) -> std::borrow::Cow<'_, str> {
    if text.contains('\u{2015}') { text.replace('\u{2015}', "\u{2014}").into() } else { text.into() }
}

/// An empty directory at `path`, removing what an interrupted run left.
fn fresh_dir(path: &Path) -> Result<std::path::PathBuf, String> {
    if path.exists() {
        fs::remove_dir_all(path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    fs::create_dir_all(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path.to_path_buf())
}

/// The names of the files a MeCab dictionary reader reads in `dir`:
/// `matrix.def`, every `*.csv`, `char.def` and `unk.def`.
pub fn dictionary_files(dir: &Path) -> Result<Vec<String>, String> {
    let mut names = vec!["matrix.def".to_string(), "char.def".to_string(), "unk.def".to_string()];
    for e in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let e = e.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = e.file_name().to_string_lossy().into_owned();
        if name.ends_with(".csv") {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

/// Checks that every model, Hunspell dictionary file and training corpus
/// the generator loads is in `dir` with the recipe's SHA-512, the derived
/// pooled model included.
pub fn check_all(dir: &Path) -> Result<(), String> {
    MODELS.iter().chain(&LOCAL_MODELS).chain([&POOLING_SOURCE]).try_for_each(|m| check(&dir.join(m.file), m))?;
    crate::pooled::derive(dir)?;
    HUNSPELL_FILES
        .iter()
        .try_for_each(|f| check(&dir.join(f.file), &Model { file: f.file, url: f.url, sha512: f.sha512 }))?;
    CORPORA.iter().try_for_each(|c| check(&dir.join(c.file), &Model { file: c.file, url: c.url, sha512: c.sha512 }))
}

/// Fails unless `dir` holds exactly the recipe's fixtures, each with its
/// SHA-512: a fixture changed without the recipe, or a file the recipe does
/// not list, would make goldens the recipe does not describe. A fixture in
/// a subdirectory (a file of a dictionary directory) is listed by its path
/// below `dir`.
pub fn check_fixtures(dir: &Path) -> Result<(), String> {
    let mut found = Vec::new();
    list_files(dir, "", &mut found)?;
    found.sort();
    let mut expected: Vec<&str> = FIXTURES.iter().map(|f| f.file).collect();
    expected.sort();
    if found != expected {
        return Err(format!("{} holds {found:?}; the recipe's fixtures are {expected:?}", dir.display()));
    }
    for f in FIXTURES {
        let path = dir.join(f.file);
        let actual = sha512_file(&path)?;
        if actual != f.sha512 {
            return Err(format!("{}: SHA-512 is {actual}, the recipe has {}", path.display(), f.sha512));
        }
    }
    Ok(())
}

/// Every file below `dir`, by its path from `dir` with `/` between the
/// parts, prefixed with `prefix`.
fn list_files(dir: &Path, prefix: &str, out: &mut Vec<String>) -> Result<(), String> {
    for e in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let e = e.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = format!("{prefix}{}", e.file_name().to_string_lossy());
        let kind = e.file_type().map_err(|err| format!("{}: {err}", e.path().display()))?;
        if kind.is_dir() {
            list_files(&e.path(), &format!("{name}/"), out)?;
        } else {
            out.push(name);
        }
    }
    Ok(())
}

/// Fails unless the file's SHA-512 is the model's.
fn check(path: &Path, m: &Model) -> Result<(), String> {
    let actual = sha512_file(path)?;
    if actual != m.sha512 {
        return Err(format!("{}: SHA-512 is {actual}, the recipe has {} for {}", path.display(), m.sha512, m.url));
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
    fn the_horizontal_bar_of_iconv_is_the_em_dash_of_java() {
        assert_eq!(as_java_decodes_euc_jp("\u{2015},1,1,0,記号"), "\u{2014},1,1,0,記号");
        assert!(matches!(as_java_decodes_euc_jp("東京"), std::borrow::Cow::Borrowed("東京")));
    }

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

    #[test]
    fn the_checked_in_fixtures_are_the_recipes() {
        check_fixtures(&crate::paths::fixtures_dir()).unwrap();
        let dir = std::env::temp_dir().join(format!("turbo-goldens-fixtures-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("extra.bin"), b"x").unwrap();
        let err = check_fixtures(&dir).unwrap_err();
        assert!(err.contains("extra.bin"), "{err}");
        // A file in a subdirectory is listed by its path.
        fs::remove_file(dir.join("extra.bin")).unwrap();
        fs::create_dir_all(dir.join("mecab-x")).unwrap();
        fs::write(dir.join("mecab-x/extra.csv"), b"x").unwrap();
        let err = check_fixtures(&dir).unwrap_err();
        assert!(err.contains("mecab-x/extra.csv"), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
