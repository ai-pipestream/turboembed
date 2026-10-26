use std::path::PathBuf;

use turbo_train_recipes::release::ModelsRelease;

pub(crate) const USAGE: &str = "usage:
  turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
  turbo-train-recipes masc-namefind-select --corpus <dir> --references <dir> --out <dir> [--recipe <model id>]...
  turbo-train-recipes fetch-masc --home <dir>
  turbo-train-recipes ud --treebanks <dir> --references <dir> --out <dir> [--recipe <model id>]...
  turbo-train-recipes fetch-ud --home <dir> [--recipe <model id>]...
  turbo-train-recipes zoo-index --models <dir> --release <tag> [--check]
  turbo-train-recipes tatoeba --export <dir> --references <dir> --out <dir> [--unrecorded-export]
  turbo-train-recipes fineweb-ranges (--paths | --web <dir> [--files <path>] (--need | --record))
  turbo-train-recipes tatoeba-fineweb --export <dir> --web <dir> --references <dir> --out <dir>
      [--unrecorded-export] [--files <path> --ranges <path> --unrecorded-web]";

#[derive(Debug)]
pub(crate) struct Args {
    pub(crate) command: String,
    pub(crate) corpus: Option<PathBuf>,
    pub(crate) treebanks: Option<PathBuf>,
    pub(crate) references: Option<PathBuf>,
    pub(crate) out: Option<PathBuf>,
    pub(crate) home: Option<PathBuf>,
    pub(crate) models: Option<PathBuf>,
    pub(crate) release: Option<ModelsRelease>,
    pub(crate) export: Option<PathBuf>,
    pub(crate) unrecorded: bool,
    pub(crate) unrecorded_export: bool,
    pub(crate) check: bool,
    pub(crate) web: Option<PathBuf>,
    pub(crate) files: Option<PathBuf>,
    pub(crate) ranges: Option<PathBuf>,
    pub(crate) unrecorded_web: bool,
    pub(crate) paths: bool,
    pub(crate) need: bool,
    pub(crate) record: bool,
    /// The model ids `--recipe` names, resolved by the command.
    pub(crate) recipe_ids: Vec<String>,
    /// Every flag given, for the commands that refuse flags they do not
    /// read.
    pub(crate) given: Vec<String>,
}

/// The recipes of `all` that `ids` name, in the order of `all` whatever
/// the order of `ids`; all of them when `ids` is empty.
pub(crate) fn chosen<R: Copy>(
    all: &[R],
    ids: &[String],
    model_id: impl Fn(&R) -> &'static str,
) -> Result<Vec<R>, String> {
    if let Some(id) = ids.iter().find(|id| !all.iter().any(|r| model_id(r) == id.as_str())) {
        let known: Vec<&str> = all.iter().map(&model_id).collect();
        return Err(format!("no recipe trains {id}; the recipes train {}", known.join(", ")));
    }
    Ok(all.iter().filter(|r| ids.is_empty() || ids.iter().any(|id| model_id(r) == id.as_str())).copied().collect())
}

fn set_once<T>(slot: &mut Option<T>, flag: &str, value: T) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("{flag} was given more than once\n{USAGE}"));
    }
    *slot = Some(value);
    Ok(())
}

pub(crate) fn parse() -> Result<Args, String> {
    parse_from(std::env::args().skip(1))
}

pub(crate) fn parse_from(mut it: impl Iterator<Item = String>) -> Result<Args, String> {
    let command = it.next().ok_or(USAGE)?;
    let mut a = Args {
        command,
        corpus: None,
        treebanks: None,
        references: None,
        out: None,
        home: None,
        models: None,
        release: None,
        export: None,
        unrecorded: false,
        unrecorded_export: false,
        check: false,
        web: None,
        files: None,
        ranges: None,
        unrecorded_web: false,
        paths: false,
        need: false,
        record: false,
        recipe_ids: Vec::new(),
        given: Vec::new(),
    };
    while let Some(flag) = it.next() {
        a.given.push(flag.clone());
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--corpus" => set_once(&mut a.corpus, &flag, value()?.into())?,
            "--treebanks" => set_once(&mut a.treebanks, &flag, value()?.into())?,
            "--references" => set_once(&mut a.references, &flag, value()?.into())?,
            "--out" => set_once(&mut a.out, &flag, value()?.into())?,
            "--home" => set_once(&mut a.home, &flag, value()?.into())?,
            "--models" => set_once(&mut a.models, &flag, value()?.into())?,
            "--release" => {
                let rel = ModelsRelease::tagged(&value()?).map_err(|e| e.to_string())?;
                set_once(&mut a.release, &flag, rel)?;
            }
            "--unrecorded-corpus" => a.unrecorded = true,
            "--check" => a.check = true,
            "--export" => set_once(&mut a.export, &flag, value()?.into())?,
            "--unrecorded-export" => a.unrecorded_export = true,
            "--recipe" => a.recipe_ids.push(value()?),
            "--web" => set_once(&mut a.web, &flag, value()?.into())?,
            "--files" => set_once(&mut a.files, &flag, value()?.into())?,
            "--ranges" => set_once(&mut a.ranges, &flag, value()?.into())?,
            "--unrecorded-web" => a.unrecorded_web = true,
            "--paths" => a.paths = true,
            "--need" => a.need = true,
            "--record" => a.record = true,
            _ => return Err(format!("unknown argument {flag}\n{USAGE}")),
        }
    }
    Ok(a)
}

pub(crate) fn need(v: Option<PathBuf>, flag: &str) -> Result<PathBuf, String> {
    v.ok_or_else(|| format!("{flag} is required\n{USAGE}"))
}

/// Refuses a flag of `a` that `command` does not read.
pub(crate) fn only(a: &Args, command: &str, read: &[&str]) -> Result<(), String> {
    match a.given.iter().find(|f| !read.contains(&f.as_str())) {
        Some(f) => Err(format!("{command} does not read {f}\n{USAGE}")),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(raw: &[&str]) -> Result<Args, String> {
        parse_from(raw.iter().map(|s| (*s).to_owned()))
    }

    #[test]
    fn duplicate_single_value_flag_is_rejected() {
        let err = args(&["masc", "--corpus", "a", "--corpus", "b"]).unwrap_err();
        assert!(err.starts_with("--corpus was given more than once"), "{err}");
    }

    #[test]
    fn only_rejects_unrecognized_flag_for_command() {
        let parsed = args(&["fetch-masc", "--home", "dir", "--check"]).unwrap();
        let err = only(&parsed, "fetch-masc", &["--home"]).unwrap_err();
        assert!(err.starts_with("fetch-masc does not read --check"), "{err}");
    }
}
