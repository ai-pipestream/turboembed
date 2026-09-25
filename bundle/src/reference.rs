//! The reference: the upstream pipeline, fp32 on CPU, run on the upstream
//! files in the container the recipe pins. That container is the only
//! place Python runs (here and for the conversions, convert.rs), and only
//! while a bundle is made.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};

use crate::Result;
use crate::recipe::Recipe;

/// `name@sha256:<64 hex>`: the image is pinned by content, never by tag.
pub fn check_pinned(container: &str) -> Result<&str> {
    let hex = container
        .split_once("@sha256:")
        .map(|(_, h)| h)
        .filter(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()))
        .ok_or_else(|| format!("reference container {container:?} is not pinned as name@sha256:<64 hex>"))?;
    Ok(hex)
}

/// What the container reads: the cases with their prefixes resolved, and
/// the row length the model is evaluated at.
pub fn cases(recipe: &Recipe) -> Result<Value> {
    let m = &recipe.manifest;
    let max_seq = m["embed"]["max_seq"].as_u64().ok_or("manifest.embed.max_seq: missing")?;
    let max_batch = m["embed"]["max_batch"].as_u64().ok_or("manifest.embed.max_batch: missing")?;
    let prefix = |role: &str| -> Result<String> {
        Ok(match role {
            "PROMPT_NONE" => String::new(),
            "PROMPT_QUERY" => m["embed"]["prefix_query"].as_str().unwrap_or_default().to_owned(),
            "PROMPT_DOCUMENT" => m["embed"]["prefix_document"].as_str().unwrap_or_default().to_owned(),
            r => return Err(format!("reference case prompt_role {r:?} is not a PROMPT_* value")),
        })
    };
    let mut out = Vec::new();
    for (i, c) in m["reference"]["cases"].as_array().ok_or("manifest.reference.cases: missing")?.iter().enumerate() {
        let text = c["text"].as_str().ok_or(format!("reference.cases[{i}].text: missing"))?;
        let role = c["prompt_role"].as_str().ok_or(format!("reference.cases[{i}].prompt_role: missing"))?;
        out.push(json!({ "text": text, "prefix": prefix(role)? }));
    }
    Ok(json!({ "max_seq": max_seq, "max_batch": max_batch, "cases": out }))
}

/// Run the container on `upstream` and put the reference file in `bundle`.
/// Returns the reference's `produced_by`, as the container reported it.
pub fn run(recipe: &Recipe, upstream: &Path, bundle: &Path) -> Result<Value> {
    let container = recipe.str_at("/reference/produced_by/container")?;
    let image = present(container)?;

    let work = scratch(bundle)?;
    fs::write(work.join("cases.json"), serde_json::to_vec_pretty(&cases(recipe)?).unwrap())
        .map_err(|e| format!("{}: {e}", work.display()))?;
    let abs = |p: &Path| fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display()));
    let mut cmd = Command::new("docker");
    cmd.args(["run", "--rm", "--network", "none", "--mount"])
        .arg(format!("type=bind,src={},dst=/model,readonly", abs(upstream)?.display()))
        .arg("--mount")
        .arg(format!("type=bind,src={},dst=/work", abs(&work)?.display()));
    if let Some(user) = current_user() {
        cmd.args(["--user", &user]);
    }
    cmd.arg(&image).args(["/model", "/work/cases.json", "/work/reference.safetensors", "/work/produced_by.json"]);
    let out = cmd.output().map_err(|e| format!("docker: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "the reference container failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }

    let file = recipe.str_at("/reference/file")?;
    crate::check_rel(file)?;
    let bytes = fs::read(work.join("reference.safetensors")).map_err(|e| format!("reference output: {e}"))?;
    crate::fetch::write_atomic(&bundle.join(file), &bytes)?;
    let reported: Value = serde_json::from_slice(
        &fs::read(work.join("produced_by.json")).map_err(|e| format!("produced_by output: {e}"))?,
    )
    .map_err(|e| format!("produced_by output: {e}"))?;
    let _ = fs::remove_dir_all(&work);
    produced_by(&reported, container)
}

/// The reference's `produced_by`: what the container says ran, and the
/// container it ran in. Every field comes from the run, none from the recipe.
pub fn produced_by(reported: &Value, container: &str) -> Result<Value> {
    let s = |k: &str| reported[k].as_str().map(str::to_owned).ok_or(format!("produced_by output: no {k}"));
    let args = reported["args"].as_array().ok_or("produced_by output: no args")?;
    if !args.iter().all(Value::is_string) {
        return Err("produced_by output: args are not strings".into());
    }
    Ok(json!({
        "tool": s("tool")?,
        "tool_version": s("tool_version")?,
        "container": container,
        "args": args,
        // fp32 on CPU is not bit-identical across CPUs and library builds.
        "reproducible": false,
    }))
}

/// The image docker runs for a pinned container: a registry image is
/// found by its digest, one built here by its image id.
pub(crate) fn present(container: &str) -> Result<String> {
    let hex = check_pinned(container)?;
    let image = if docker_has(container) { container.to_owned() } else { format!("sha256:{hex}") };
    if !docker_has(&image) {
        return Err(format!("the reference container {container} is not present; pull or build it first"));
    }
    Ok(image)
}

pub(crate) fn docker_has(image: &str) -> bool {
    Command::new("docker")
        .args(["image", "inspect", "--format", "{{.Id}}", image])
        .output()
        .is_ok_and(|o| o.status.success())
}

pub(crate) fn current_user() -> Option<String> {
    let id = |flag: &str| {
        Command::new("id")
            .arg(flag)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    Some(format!("{}:{}", id("-u")?, id("-g")?))
}

/// Beside the bundle directory, not under /tmp, which some Docker
/// installations cannot mount.
pub(crate) fn scratch(bundle: &Path) -> Result<PathBuf> {
    let bundle = fs::canonicalize(bundle).map_err(|e| format!("{}: {e}", bundle.display()))?;
    let parent = bundle.parent().ok_or_else(|| format!("{} has no parent directory", bundle.display()))?;
    let d = parent.join(format!(".turbo-bundle-work-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    Ok(d)
}
