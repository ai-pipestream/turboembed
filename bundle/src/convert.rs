//! Artifacts the bundle makes from another one: the F16 ONNX file, from
//! the upstream F32 export, for the reference programs that build F16
//! only from a strongly typed graph (TensorRT's trtexec from 10 on).
//! The conversion runs in the reference container the recipe pins, with
//! no network; the library never reads either file.
//!
//! In the recipe such an artifact's `produced_by` names only `from`, the
//! artifact it is made from. The rest comes from the run: what the
//! container reports, the container itself, the files and settings, and
//! whether a second run gave the same bytes.

use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::{Value, json};

use crate::recipe::Recipe;
use crate::reference::{current_user, present, scratch};
use crate::{Result, check_rel};

/// The script in the reference image that makes an F16 ONNX file.
pub const ONNX_F16: &str = "/onnx_f16.py";

/// One artifact to make: its name, its one file, the file it is made
/// from and that artifact's name, and the script that makes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Conversion {
    pub name: String,
    pub file: String,
    pub from: String,
    pub from_file: String,
    pub script: &'static str,
}

fn one_file(a: &Value, name: &str) -> Result<String> {
    match a["files"].as_array().map(Vec::as_slice) {
        Some([f]) => {
            let f = f.as_str().ok_or(format!("artifact {name}: files are not paths"))?;
            check_rel(f)?;
            Ok(f.to_owned())
        }
        _ => Err(format!("artifact {name}: a converted artifact and its source are one file each")),
    }
}

/// The recipe's artifacts with a `produced_by`, each one the tool can
/// make: a FORMAT_ONNX file in DTYPE_F16 from a FORMAT_ONNX file with no
/// compute_dtype (the upstream export). Anything else is refused, before
/// anything runs.
pub fn conversions(recipe: &Recipe) -> Result<Vec<Conversion>> {
    let artifacts = recipe.manifest["artifacts"].as_array().ok_or("manifest.artifacts: missing")?;
    let mut out = Vec::new();
    for a in artifacts {
        let Some(pb) = a.get("produced_by") else { continue };
        let name = a["name"].as_str().ok_or("an artifact has no name")?;
        let keys: Vec<&String> = pb.as_object().map(|o| o.keys().collect()).unwrap_or_default();
        if keys != ["from"] {
            return Err(format!(
                "artifact {name}: the recipe's produced_by names only from; the rest comes from the run, not {keys:?}"
            ));
        }
        let from = pb["from"].as_str().ok_or(format!("artifact {name}: produced_by.from is not a name"))?;
        let src = artifacts
            .iter()
            .find(|s| s["name"] == from)
            .ok_or(format!("artifact {name}: produced_by.from names no artifact {from:?}"))?;
        let onnx = |x: &Value| x["format"] == "FORMAT_ONNX";
        if !(onnx(a) && onnx(src) && a["compute_dtype"] == "DTYPE_F16" && src.get("compute_dtype").is_none()) {
            return Err(format!(
                "artifact {name}: the bundle tool makes only a DTYPE_F16 FORMAT_ONNX file from a FORMAT_ONNX file \
                 with no compute_dtype"
            ));
        }
        out.push(Conversion {
            name: name.to_owned(),
            file: one_file(a, name)?,
            from: from.to_owned(),
            from_file: one_file(src, from)?,
            script: ONNX_F16,
        });
    }
    Ok(out)
}

/// A converted artifact's `produced_by`: what the container says ran, the
/// container, the artifact it came from, the files and settings, and
/// whether two runs gave identical bytes.
pub fn produced_by(reported: &Value, container: &str, c: &Conversion, reproducible: bool) -> Result<Value> {
    let s = |k: &str| reported[k].as_str().map(str::to_owned).ok_or(format!("{}: the run reported no {k}", c.name));
    let settings = reported["settings"].as_array().ok_or(format!("{}: the run reported no settings", c.name))?;
    let mut args = vec![Value::from(c.from_file.clone()), Value::from(c.file.clone())];
    for v in settings {
        args.push(Value::from(v.as_str().ok_or(format!("{}: a setting is not a string", c.name))?));
    }
    Ok(json!({
        "tool": s("tool")?,
        "tool_version": s("tool_version")?,
        "container": container,
        "from": c.from,
        "args": args,
        "reproducible": reproducible,
    }))
}

/// Make each converted artifact in `bundle` from the staged file it
/// comes from, in the reference container, twice. Returns each one's
/// name and `produced_by`.
pub fn run(recipe: &Recipe, bundle: &Path) -> Result<Vec<(String, Value)>> {
    let todo = conversions(recipe)?;
    if todo.is_empty() {
        return Ok(Vec::new());
    }
    let container = recipe.str_at("/reference/produced_by/container")?;
    let image = present(container)?;
    let abs = |p: &Path| fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display()));
    let mut out = Vec::new();
    for c in &todo {
        let work = scratch(bundle)?;
        let mut bytes = Vec::new();
        let mut reported = Value::Null;
        for run in ["a", "b"] {
            let mut cmd = Command::new("docker");
            cmd.args(["run", "--rm", "--network", "none", "--mount"])
                .arg(format!("type=bind,src={},dst=/bundle,readonly", abs(bundle)?.display()))
                .arg("--mount")
                .arg(format!("type=bind,src={},dst=/work", abs(&work)?.display()));
            if let Some(user) = current_user() {
                cmd.args(["--user", &user]);
            }
            cmd.args(["--entrypoint", "python"]).arg(&image).arg(c.script).args([
                format!("/bundle/{}", c.from_file),
                format!("/work/{run}.out"),
                format!("/work/{run}.json"),
            ]);
            let o = cmd.output().map_err(|e| format!("docker: {e}"))?;
            if !o.status.success() {
                return Err(format!(
                    "making {} in {container} failed (an image built before {} was added to it needs \
                     building again, bundle/README.md):\n{}{}",
                    c.name,
                    c.script,
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                ));
            }
            bytes.push(fs::read(work.join(format!("{run}.out"))).map_err(|e| format!("{}: {e}", c.name))?);
            reported = serde_json::from_slice(
                &fs::read(work.join(format!("{run}.json"))).map_err(|e| format!("{}: {e}", c.name))?,
            )
            .map_err(|e| format!("{}: produced_by output: {e}", c.name))?;
        }
        let _ = fs::remove_dir_all(&work);
        crate::fetch::write_atomic(&bundle.join(&c.file), &bytes[0])?;
        out.push((c.name.clone(), produced_by(&reported, container, c, bytes[0] == bytes[1])?));
    }
    Ok(out)
}
