//! Artifacts the bundle makes from another one:
//!
//! - the F16 ONNX file, from the upstream F32 export, for the reference
//!   programs that build F16 only from a strongly typed graph (TensorRT's
//!   trtexec from 10 on), in the reference container the recipe pins;
//! - a HEF for a Hailo device, compiled from the upstream F32 export by
//!   Hailo's Dataflow Compiler (bundle/hailo), in the container the
//!   artifact's `produced_by.container` pins, calibrated on the files its
//!   `produced_by.inputs` names.
//!
//! Each runs with no network, twice. The library never reads the ONNX
//! files; it loads the HEF. In the recipe an F16 file's `produced_by`
//! names only `from`, the artifact it is made from, and a HEF's names
//! `from`, `container` and `inputs`. The rest comes from the run: what the
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

/// The script in the Dataflow Compiler image that makes a HEF.
pub const HEF_COMPILE: &str = "/hef_compile.py";

/// One artifact to make: its name, its one file, the file it is made
/// from and that artifact's name, the script that makes it, the pinned
/// container it runs in (None: the reference container), the bundle files
/// it reads besides, and the arguments that follow the files.
#[derive(Debug, Clone, PartialEq)]
pub struct Conversion {
    pub name: String,
    pub file: String,
    pub from: String,
    pub from_file: String,
    pub script: &'static str,
    pub container: Option<String>,
    pub inputs: Vec<String>,
    pub args: Vec<String>,
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
/// make: a FORMAT_ONNX file in DTYPE_F16, or a FORMAT_HEF in DTYPE_I8
/// from INPUT_EMBEDDINGS, each from a FORMAT_ONNX file with no
/// compute_dtype that starts at INPUT_TOKEN_IDS (the upstream export).
/// Anything else is refused, before anything runs.
pub fn conversions(recipe: &Recipe) -> Result<Vec<Conversion>> {
    let artifacts = recipe.manifest["artifacts"].as_array().ok_or("manifest.artifacts: missing")?;
    let mut out = Vec::new();
    for a in artifacts {
        let Some(pb) = a.get("produced_by") else { continue };
        let name = a["name"].as_str().ok_or("an artifact has no name")?;
        let mut keys: Vec<&str> = pb.as_object().map(|o| o.keys().map(String::as_str).collect()).unwrap_or_default();
        keys.sort_unstable();
        let hef = a["format"] == "FORMAT_HEF";
        let named = if hef { &["container", "from", "inputs"][..] } else { &["from"][..] };
        if keys != named {
            let names = if hef { "from, container and inputs" } else { "only from" };
            return Err(format!(
                "artifact {name}: the recipe's produced_by names {names}; the rest comes from the run, not {keys:?}"
            ));
        }
        let from = pb["from"].as_str().ok_or(format!("artifact {name}: produced_by.from is not a name"))?;
        let src = artifacts
            .iter()
            .find(|s| s["name"] == from)
            .ok_or(format!("artifact {name}: produced_by.from names no artifact {from:?}"))?;
        let upstream_export = src["format"] == "FORMAT_ONNX"
            && src.get("compute_dtype").is_none()
            && src["graph_input"] == "INPUT_TOKEN_IDS";
        let from_file = one_file(src, from)?;
        if !hef {
            if !(a["format"] == "FORMAT_ONNX" && a["compute_dtype"] == "DTYPE_F16" && upstream_export) {
                return Err(format!(
                    "artifact {name}: the bundle tool makes only a DTYPE_F16 FORMAT_ONNX file or a FORMAT_HEF \
                     from the FORMAT_ONNX export with no compute_dtype"
                ));
            }
            out.push(Conversion {
                name: name.to_owned(),
                file: one_file(a, name)?,
                from: from.to_owned(),
                from_file,
                script: ONNX_F16,
                container: None,
                inputs: Vec::new(),
                args: Vec::new(),
            });
            continue;
        }
        out.push(hef_conversion(recipe, a, name, pb, from, from_file, upstream_export)?);
    }
    Ok(out)
}

/// A FORMAT_HEF artifact: DTYPE_I8, from INPUT_EMBEDDINGS to
/// OUTPUT_HIDDEN_STATES, for one target at a fixed frame of fixed_seq
/// tokens and one row, compiled in a pinned container on one file of
/// calibration texts.
fn hef_conversion(
    recipe: &Recipe,
    a: &Value,
    name: &str,
    pb: &Value,
    from: &str,
    from_file: String,
    upstream_export: bool,
) -> Result<Conversion> {
    let refuse = |why: &str| format!("artifact {name}: {why}");
    if !upstream_export {
        return Err(refuse("a HEF is compiled from the FORMAT_ONNX export with no compute_dtype"));
    }
    if a["compute_dtype"] != "DTYPE_I8"
        || a["graph_input"] != "INPUT_EMBEDDINGS"
        || a["graph_output"] != "OUTPUT_HIDDEN_STATES"
    {
        return Err(refuse("the bundle tool compiles a DTYPE_I8 HEF from INPUT_EMBEDDINGS to OUTPUT_HIDDEN_STATES"));
    }
    let target = a["target"].as_str().filter(|t| !t.is_empty()).ok_or(refuse("a HEF names its target"))?;
    let seq = a["fixed_seq"].as_u64().filter(|&s| s > 0).ok_or(refuse("a HEF gives fixed_seq"))?;
    if a["fixed_batch"].as_u64() != Some(1) {
        return Err(refuse("the bundle tool compiles a HEF of one row a frame: fixed_batch 1"));
    }
    let container = pb["container"].as_str().ok_or(refuse("produced_by.container is not a string"))?;
    crate::reference::check_pinned(container)?;
    let inputs: Vec<String> = match pb["inputs"].as_array().map(Vec::as_slice) {
        Some([f]) => {
            let f = f.as_str().ok_or(refuse("produced_by.inputs are not paths"))?;
            check_rel(f)?;
            vec![f.to_owned()]
        }
        _ => return Err(refuse("produced_by.inputs is the one file of calibration texts")),
    };
    let heads = recipe.manifest["architecture"]["heads"].as_u64().ok_or("manifest.architecture.heads: missing")?;
    let tokenizer = recipe.str_at("/tokenizer/file")?;
    Ok(Conversion {
        name: name.to_owned(),
        file: one_file(a, name)?,
        from: from.to_owned(),
        from_file,
        script: HEF_COMPILE,
        container: Some(container.to_owned()),
        args: vec![
            "--tokenizer".into(),
            format!("/bundle/{tokenizer}"),
            "--calibration".into(),
            format!("/bundle/{}", inputs[0]),
            "--target".into(),
            target.to_owned(),
            "--seq".into(),
            seq.to_string(),
            "--heads".into(),
            heads.to_string(),
        ],
        inputs,
    })
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
    let mut pb = json!({
        "tool": s("tool")?,
        "tool_version": s("tool_version")?,
        "container": container,
        "from": c.from,
        "args": args,
        "reproducible": reproducible,
    });
    if !c.inputs.is_empty() {
        pb["inputs"] = json!(c.inputs);
    }
    Ok(pb)
}

/// Make each converted artifact in `bundle` from the staged file it
/// comes from, in the reference container, twice. Returns each one's
/// name and `produced_by`.
pub fn run(recipe: &Recipe, bundle: &Path) -> Result<Vec<(String, Value)>> {
    let todo = conversions(recipe)?;
    if todo.is_empty() {
        return Ok(Vec::new());
    }
    let reference = recipe.str_at("/reference/produced_by/container")?;
    let abs = |p: &Path| fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display()));
    let mut out = Vec::new();
    for c in &todo {
        let container = c.container.as_deref().unwrap_or(reference);
        let image = present(container)?;
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
            cmd.args(["--entrypoint", "python"])
                .arg(&image)
                .arg(c.script)
                .args([format!("/bundle/{}", c.from_file), format!("/work/{run}.out"), format!("/work/{run}.json")])
                .args(&c.args);
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
