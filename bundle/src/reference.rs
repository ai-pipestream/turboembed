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
    let mut spec = json!({ "max_seq": max_seq, "max_batch": max_batch, "cases": out });
    if let Some(st) = m.get("static_embedding") {
        // What StaticModel is built with: the row of ids padded with the
        // pad token, or the unknown one when there is none.
        let specials =
            m["tokenizer"]["special_tokens"].as_array().ok_or("manifest.tokenizer.special_tokens: missing")?;
        let id = |role: &str| specials.iter().find(|s| s["role"] == role).and_then(|s| s["id"].as_u64());
        spec["max_length"] = st["max_length"].clone();
        spec["normalize"] = json!(m["embed"]["normalize"] == "NORMALIZE_L2");
        spec["pad_id"] = json!(id("SPECIAL_PAD").or(id("SPECIAL_UNK")).ok_or("manifest.tokenizer: no SPECIAL_UNK")?);
    }
    Ok(spec)
}

/// The upstream path of the file the bundle carries at `to`.
fn upstream_of<'a>(recipe: &'a Recipe, to: &str) -> Result<&'a str> {
    recipe
        .upstream
        .iter()
        .find(|u| u.to.as_deref() == Some(to))
        .map(|u| u.path.as_str())
        .ok_or_else(|| format!("recipe.upstream: no file is carried at {to}"))
}

/// Run the container on `upstream` and put the reference file in `bundle`.
/// Returns the reference's `produced_by`, as the container reported it.
pub fn run(recipe: &Recipe, upstream: &Path, bundle: &Path) -> Result<Value> {
    let container = recipe.str_at("/reference/produced_by/container")?;
    let image = present(container)?;

    let work = scratch(bundle)?;
    let mut spec = cases(recipe)?;
    // A static model's upstream is Model2Vec's own files: the static
    // reference builds StaticModel from them.
    let is_static = recipe.manifest.get("static_embedding").is_some();
    if is_static {
        spec["weights"] = json!(upstream_of(recipe, recipe.str_at("/artifacts/0/files/0")?)?);
        spec["tokenizer"] = json!(upstream_of(recipe, recipe.str_at("/tokenizer/file")?)?);
    }
    fs::write(work.join("cases.json"), serde_json::to_vec_pretty(&spec).unwrap())
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
    if is_static {
        cmd.args(["--entrypoint", "python", &image, "/static_reference.py"]);
    } else {
        cmd.arg(&image);
    }
    cmd.args(["/model", "/work/cases.json", "/work/reference.safetensors", "/work/produced_by.json"]);
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
/// found by its digest, one built here by its config digest, which is the
/// image id docker's classic store reports. Docker's containerd store
/// reports a manifest digest as the id instead, so there the image is found
/// among the ones carrying the pin's name by the config its saved manifest
/// names.
pub(crate) fn present(container: &str) -> Result<String> {
    let hex = check_pinned(container)?;
    if docker_has(container) {
        return Ok(container.to_owned());
    }
    let id = format!("sha256:{hex}");
    if docker_has(&id) {
        return Ok(id);
    }
    let name = container.split_once('@').map_or(container, |(n, _)| n);
    if let Some(image) = image_with_config(name, hex) {
        return Ok(image);
    }
    Err(format!("the reference container {container} is not present; pull or build it first"))
}

/// The local image tagged with `name` whose config digest is `hex`, if any.
fn image_with_config(name: &str, hex: &str) -> Option<String> {
    let out = Command::new("docker")
        .args(["image", "ls", "--format", "{{.Repository}}:{{.Tag}}", name])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let listed = String::from_utf8_lossy(&out.stdout);
    listed
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .find(|image| config_digest(image).as_deref() == Some(hex))
        .map(str::to_owned)
}

/// An image's config digest, read from the manifest `docker image save`
/// writes, which names the config the same way under either store: as
/// `<hex>.json` from the classic store, as `blobs/sha256/<hex>` from the
/// containerd store.
fn config_digest(image: &str) -> Option<String> {
    use std::process::Stdio;
    let mut save = Command::new("docker")
        .args(["image", "save", image])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let tar = Command::new("tar")
        .args(["-xO", "manifest.json"])
        .stdin(save.stdout.take()?)
        .stderr(Stdio::null())
        .output()
        .ok();
    let _ = save.wait();
    let manifest: Value = serde_json::from_slice(&tar?.stdout).ok()?;
    let config = manifest.get(0)?.get("Config")?.as_str()?;
    let hex = config.trim_end_matches(".json").rsplit('/').next()?.trim_start_matches("sha256:");
    (hex.len() == 64).then(|| hex.to_owned())
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

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "needs docker and the pinned image: TURBO_TEST_IMAGE_PIN=name@sha256:<64 hex>"]
    fn a_pinned_image_is_found_under_either_store() {
        let pin = std::env::var("TURBO_TEST_IMAGE_PIN").expect("TURBO_TEST_IMAGE_PIN");
        let image = super::present(&pin).unwrap();
        println!("{pin} -> {image}");
        assert!(super::docker_has(&image));
    }
}
