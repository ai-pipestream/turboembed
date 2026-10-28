//! The rotary families' architecture against the upstream config.json
//! (#1461), after deberta/config.rs: FAMILY_MODERNBERT against
//! answerdotai/ModernBERT's config (modernbert.rs) and FAMILY_NOMIC_BERT
//! against nomic-embed-text's (nomic.rs). Each takes the keys of the
//! config as upstream wrote it and as transformers 5 saves it again, refuses
//! every key whose value no kernel computes, naming it, and then requires
//! every field of `architecture` to be the one the config gives.

mod modernbert;
mod nomic;
#[cfg(test)]
mod tests;

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::Result;
use crate::recipe::Recipe;

/// The recipe's architecture against `cfg`, the upstream config.json of a
/// FAMILY_MODERNBERT or FAMILY_NOMIC_BERT model. A recipe of another family
/// is not checked.
pub fn agrees(cfg: &Value, manifest: &Value) -> Result<()> {
    let a = &manifest["architecture"];
    let want = match a["family"].as_str() {
        Some("FAMILY_MODERNBERT") => modernbert::architecture(&Config(cfg))?,
        Some("FAMILY_NOMIC_BERT") => nomic::architecture(&Config(cfg))?,
        _ => return Ok(()),
    };
    for (path, want) in want {
        let got = a.pointer(path).filter(|v| !v.is_null()).cloned().or_else(|| default(path));
        if !same(got.as_ref(), want.as_ref()) {
            let shown = |v: Option<&Value>| v.map_or("absent".to_owned(), Value::to_string);
            let name = path.trim_start_matches('/').replace('/', ".");
            return Err(format!(
                "manifest.architecture.{name}: {}, the upstream config.json gives {}",
                shown(got.as_ref()),
                shown(want.as_ref())
            ));
        }
    }
    Ok(())
}

/// A rotary recipe's architecture against the upstream config.json, which
/// such a recipe must have.
pub fn check_profile(recipe: &Recipe, upstream: &Path) -> Result<()> {
    let family = &recipe.manifest["architecture"]["family"];
    if family != "FAMILY_MODERNBERT" && family != "FAMILY_NOMIC_BERT" {
        return Ok(());
    }
    let path = upstream.join("config.json");
    let bytes = fs::read(&path)
        .map_err(|e| format!("upstream config.json: {e}; a {family} recipe's architecture is checked against it"))?;
    let cfg: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    agrees(&cfg, &recipe.manifest)
}

/// The fields a manifest leaves out when they hold their default, so a
/// recipe may write them or not.
fn default(path: &str) -> Option<Value> {
    match path {
        "/norm" => Some("NORM_LAYER".into()),
        "/mlp" => Some("MLP_DENSE".into()),
        "/position_offset" => Some(0.into()),
        _ => None,
    }
}

/// Equal, numbers by value (`1000` is `1000.0`).
fn same(a: Option<&Value>, b: Option<&Value>) -> bool {
    match (a, b) {
        (Some(Value::Number(x)), Some(Value::Number(y))) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

/// The manifest fields a config gives: a path under `architecture` and the
/// value it must hold, `None` for a field that must be absent.
type Fields = Vec<(&'static str, Option<Value>)>;

/// An upstream config.json, read key by key, every refusal naming the key.
struct Config<'a>(&'a Value);

impl Config<'_> {
    /// The key's value; a JSON null is a value, as upstream reads it.
    fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }

    /// The key must be present and hold `want`.
    fn must(&self, key: &str, want: Value) -> Result<()> {
        match self.get(key) {
            Some(v) if same(Some(v), Some(&want)) => Ok(()),
            Some(v) => Err(format!("upstream config.json: {key} is {v}; only {want} is supported")),
            None => Err(format!("upstream config.json: {key} is missing; it must be {want}")),
        }
    }

    /// Absent, the key takes upstream's default, `want`; present, it must
    /// hold it.
    fn default_is(&self, key: &str, want: Value) -> Result<()> {
        match self.get(key) {
            Some(v) if !same(Some(v), Some(&want)) => {
                Err(format!("upstream config.json: {key} is {v}; only {want} is supported"))
            }
            _ => Ok(()),
        }
    }

    /// The value the keys give: at least one present, and every present one
    /// a number and the same (an old name beside its new one).
    /// The key may be absent or null; present, it must hold `want`.
    fn default_is_or_null(&self, key: &str, want: Value) -> Result<()> {
        match self.get(key) {
            Some(Value::Null) | None => Ok(()),
            Some(_) => self.default_is(key, want),
        }
    }

    fn number(&self, keys: &[&str]) -> Result<Value> {
        let mut found: Option<(&str, &Value)> = None;
        for &k in keys {
            let Some(v) = self.get(k) else { continue };
            if !v.is_number() {
                return Err(format!("upstream config.json: {k} is {v}, not a number"));
            }
            if let Some((first, f)) = found {
                if !same(Some(f), Some(v)) {
                    return Err(format!("upstream config.json: {first} is {f} and {k} is {v}; they must agree"));
                }
            } else {
                found = Some((k, v));
            }
        }
        found.map(|(_, v)| v.clone()).ok_or_else(|| format!("upstream config.json: {} is missing", keys.join(" or ")))
    }

    /// An object key's members must all be in `allowed`, so a rotary
    /// variant the kernels do not compute is refused by name.
    fn only_members(&self, key: &str, v: &Value, allowed: &[&str]) -> Result<()> {
        let obj = v.as_object().ok_or_else(|| format!("upstream config.json: {key} is {v}, not an object"))?;
        match obj.keys().find(|k| !allowed.contains(&k.as_str())) {
            Some(k) => Err(format!("upstream config.json: {key}.{k} is not supported (only {})", allowed.join(", "))),
            None => Ok(()),
        }
    }

    /// A `rope_parameters` entry: only `rope_theta` and the default
    /// `rope_type`, its theta returned when present.
    fn rope_entry(&self, key: &str, v: &Value) -> Result<Option<Value>> {
        self.only_members(key, v, &["rope_theta", "rope_type"])?;
        if let Some(t) = v.get("rope_type").filter(|t| *t != "default") {
            return Err(format!("upstream config.json: {key}.rope_type is {t}; only \"default\" is supported"));
        }
        match v.get("rope_theta") {
            Some(t) if t.is_number() => Ok(Some(t.clone())),
            Some(t) => Err(format!("upstream config.json: {key}.rope_theta is {t}, not a number")),
            None => Ok(None),
        }
    }

    /// A theta given by `legacy` and by `rope_parameters` at `entry`
    /// (absent: the object itself): at least one, and both the same.
    fn theta(&self, legacy: &str, entry: Option<&str>) -> Result<Value> {
        let name = entry.map_or("rope_parameters".to_owned(), |e| format!("rope_parameters.{e}"));
        let nested = match (self.get("rope_parameters"), entry) {
            (None, _) => None,
            (Some(p), Some(e)) => match p.get(e) {
                Some(v) => self.rope_entry(&name, v)?,
                None => None,
            },
            (Some(p), None) => self.rope_entry(&name, p)?,
        };
        let theta = format!("{name}.rope_theta");
        match (self.get(legacy), nested) {
            (Some(l), Some(n)) if !same(Some(l), Some(&n)) => {
                Err(format!("upstream config.json: {legacy} is {l} and {theta} is {n}; they must agree"))
            }
            (Some(l), _) if !l.is_number() => Err(format!("upstream config.json: {legacy} is {l}, not a number")),
            (Some(l), _) => Ok(l.clone()),
            (None, Some(n)) => Ok(n),
            (None, None) => Err(format!("upstream config.json: {legacy} or {theta} is missing")),
        }
    }

    /// `head_dim`, when the config writes it, must be hidden over heads.
    fn head_dim(&self, hidden: &Value, heads: &Value) -> Result<()> {
        let (Some(h), Some(n)) = (hidden.as_u64(), heads.as_u64()) else {
            return Err("upstream config.json: the hidden size and head count must be whole numbers".into());
        };
        if n == 0 || !h.is_multiple_of(n) {
            return Err(format!("upstream config.json: hidden size {h} is not a multiple of {n} heads"));
        }
        self.default_is("head_dim", Value::from(h / n))
    }
}
