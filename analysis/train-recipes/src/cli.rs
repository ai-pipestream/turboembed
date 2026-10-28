use std::path::PathBuf;

use turbo_train_recipes::release::ModelsRelease;

pub(crate) const USAGE: &str = "usage:
  turbo-train-recipes fetch-masc --home <dir>
  turbo-train-recipes fetch-ud --home <dir> [--recipe <model id>]...
  turbo-train-recipes fineweb-ranges (--paths | --web <dir> [--files <path>] (--need | --record))
  turbo-train-recipes static-embed --teacher <safetensors> --tokenizer <tokenizer.json> --recipes <dir>
      [--sentences <path>]... [--parquet <path>] --out <dir> --commit <40-hex> [--steps <n>] [--lr <f>] [--init-only]
Classical training, reference-model gates and publication commands are disabled by #1511.";

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
    pub(crate) teacher: Option<PathBuf>,
    pub(crate) tokenizer: Option<PathBuf>,
    pub(crate) recipes: Option<PathBuf>,
    pub(crate) sentences: Vec<PathBuf>,
    pub(crate) parquet: Option<PathBuf>,
    pub(crate) commit: Option<String>,
    pub(crate) steps: Option<String>,
    pub(crate) lr: Option<String>,
    pub(crate) init_only: bool,
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
    if matches!(
        command.as_str(),
        "masc" | "masc-namefind-select" | "ud" | "depparse" | "zoo-index" | "tatoeba" | "tatoeba-fineweb"
    ) {
        return Err(format!(
            "{command}: disabled by #1511; OpenNLP model training, reference gates and publication are disabled. Replacement trainers: #1518; models: #1522-#1525."
        ));
    }
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
        teacher: None,
        tokenizer: None,
        recipes: None,
        sentences: Vec::new(),
        parquet: None,
        commit: None,
        steps: None,
        lr: None,
        init_only: false,
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
            "--teacher" => set_once(&mut a.teacher, &flag, value()?.into())?,
            "--tokenizer" => set_once(&mut a.tokenizer, &flag, value()?.into())?,
            "--recipes" => set_once(&mut a.recipes, &flag, value()?.into())?,
            "--sentences" => a.sentences.push(value()?.into()),
            "--parquet" => set_once(&mut a.parquet, &flag, value()?.into())?,
            "--commit" => set_once(&mut a.commit, &flag, value()?)?,
            "--steps" => set_once(&mut a.steps, &flag, value()?)?,
            "--lr" => set_once(&mut a.lr, &flag, value()?)?,
            "--init-only" => a.init_only = true,
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
        let err = args(&["fetch-masc", "--home", "a", "--home", "b"]).unwrap_err();
        assert!(err.starts_with("--home was given more than once"), "{err}");
    }

    #[test]
    fn only_rejects_unrecognized_flag_for_command() {
        let parsed = args(&["fetch-masc", "--home", "dir", "--check"]).unwrap();
        let err = only(&parsed, "fetch-masc", &["--home"]).unwrap_err();
        assert!(err.starts_with("fetch-masc does not read --check"), "{err}");
    }
}
