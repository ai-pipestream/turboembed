//! Putting the bundle together: the upstream files it carries, the
//! `files` list, the manifest, and the check that the core loads it.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use turbo::bundle::{Bundle, sha256_hex};
use turbo::manifest::{Manifest, Normalize};
use turbo::safetensors::{self, Dtype};
use turbo::tokenizer::Tokenizer;

use crate::recipe::Recipe;
use crate::{Result, check_rel};

/// Copy the upstream files the bundle carries from `upstream` into `bundle`.
pub fn stage(recipe: &Recipe, upstream: &Path, bundle: &Path) -> Result<()> {
    stage_named(recipe, upstream, bundle, None)
}

/// As `stage`, copying a local file only when `keep` names its bundle
/// path. `None` copies every local file.
fn stage_named(recipe: &Recipe, upstream: &Path, bundle: &Path, keep: Option<&BTreeSet<String>>) -> Result<()> {
    for u in &recipe.upstream {
        if let Some(to) = &u.to {
            if keep.is_some_and(|k| !k.contains(to)) {
                continue;
            }
            let bytes = fs::read(upstream.join(&u.path)).map_err(|e| format!("upstream {}: {e}", u.path))?;
            crate::fetch::write_atomic(&bundle.join(to), &bytes)?;
        }
    }
    // The recipe's own files, each checked against its pinned hash.
    for l in &recipe.local {
        if keep.is_some_and(|k| !k.contains(&l.to)) {
            continue;
        }
        let from = recipe.dir.join(&l.path);
        let bytes = fs::read(&from).map_err(|e| format!("local {}: {e}", from.display()))?;
        let got = sha256_hex(&bytes);
        if got != l.sha256 {
            return Err(format!("local {}: sha256 {got}, the recipe pins {}", l.path, l.sha256));
        }
        crate::fetch::write_atomic(&bundle.join(&l.to), &bytes)?;
    }
    Ok(())
}

/// Seal a bundle from files already in `bundle`, without running a
/// container. The reference file and its report must already be there.
/// An OpenVINO IR the recipe converts must already be there too, with
/// the report `onnx_to_openvino_ir.py` writes; that report names no
/// container, and the manifest records `host`. Any other conversion
/// whose files are absent is left out of the manifest, and one whose
/// files are present must bring a report that names its container.
/// A report is `<file>.report.json`, or `report.json` in the output
/// file's directory when that named report is absent. The named file
/// wins, so two IRs in one directory do not share a receipt. The
/// report is not left in the bundle.
pub fn seal_staged(recipe: &mut Recipe, upstream: &Path, bundle: &Path) -> Result<()> {
    if bundle.join("manifest.json").exists() {
        return Err(format!("{} already holds a bundle; make it into an empty directory", bundle.display()));
    }
    let reference_file = recipe.str_at("/reference/file")?;
    if !bundle.join(reference_file).is_file() {
        return Err(format!(
            "{reference_file} is not in the bundle. The loader checks the reference, and this command does not run \
             the container that writes it: copy the file, and reference/report.json holding its produced_by \
             (including container), from a bundle sealed with the pinned image (docs/npu.md)"
        ));
    }
    let reference_report = find_report(bundle, reference_file).ok_or_else(|| {
        format!(
            "{reference_file}: no report ({reference_file}.report.json, or report.json in its directory). Copy \
             reference.produced_by from a bundle sealed with the pinned image"
        )
    })?;

    let mut manifest = recipe.manifest.clone();
    let mut made = Vec::new();
    let mut drop_names = Vec::new();
    let mut reports = vec![reference_report];
    for c in crate::convert::conversions(recipe)? {
        match staged_outputs(bundle, &c)? {
            Staged::Absent => {
                // A token-id IR is the artifact a host seal of this backend
                // is expected to have. The embeddings cut is a second IR:
                // absent files omit it, the way a missing HEF is omitted,
                // so a seal of the token-id IR still finishes. A partial
                // pair is still an error, from staged_outputs.
                let embeddings_cut = c.args.iter().any(|a| a == "embeddings");
                if c.script == crate::convert::ONNX_TO_OPENVINO_IR && !embeddings_cut {
                    return Err(format!(
                        "{}: {} and its weights must already be in the bundle. This command does not run docker \
                         (docs/npu.md)",
                        c.name, c.file
                    ));
                }
                println!("seal: omitted {} ({} is not in the bundle)", c.name, c.file);
                drop_names.push(c.name);
            }
            Staged::Ready(report_path) => {
                let reported: Value = read_json(&report_path)?;
                require_ir_report(&c, &report_path, &reported)?;
                made.push((c.name.clone(), conversion_produced_by(&reported, &c)?));
                reports.push(report_path);
            }
        }
    }
    let arts = manifest["artifacts"].as_array_mut().ok_or("manifest.artifacts: missing")?;
    arts.retain(|a| a["name"].as_str().is_none_or(|n| !drop_names.iter().any(|d| d == n)));
    recipe.manifest = manifest;

    let reported = read_json(&reports[0])?;
    let container = reported["container"].as_str().ok_or(
        "reference/report.json has no container. Copy reference.produced_by from a bundle sealed with the pinned \
         image; this command does not run that image and does not fill the field in",
    )?;
    let produced_by = crate::reference::produced_by(&reported, container)?;
    for path in &reports {
        fs::remove_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    }

    let keep = named_paths(&recipe.manifest)?;
    stage_named(recipe, upstream, bundle, Some(&keep))?;
    seal(recipe, bundle, produced_by, made)
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}

enum Staged {
    Absent,
    Ready(std::path::PathBuf),
}

/// Both outputs present, or neither. One of the two is an error.
fn staged_outputs(bundle: &Path, c: &crate::convert::Conversion) -> Result<Staged> {
    let primary = bundle.join(&c.file).is_file();
    let secondary = match &c.file2 {
        Some(f) => bundle.join(f).is_file(),
        None => primary,
    };
    if primary != secondary {
        let missing = match &c.file2 {
            Some(f) if !bundle.join(f).is_file() => f.as_str(),
            _ => c.file.as_str(),
        };
        return Err(format!("{}: {missing} is not in the bundle, and the other output is", c.name));
    }
    if !primary {
        return Ok(Staged::Absent);
    }
    Ok(Staged::Ready(find_report(bundle, &c.file).ok_or_else(|| {
        format!(
            "{}: no report for {} ({}.report.json, or report.json in its directory). The host script writes it; \
             this command does not run docker",
            c.name, c.file, c.file
        )
    })?))
}

fn find_report(bundle: &Path, file: &str) -> Option<std::path::PathBuf> {
    let path = bundle.join(file);
    // `<file>.report.json` keeps the original suffix: embeddings.xml.report.json.
    // It is this output's own report. `report.json` beside it is the
    // fallback for a directory that holds one conversion, which is how
    // the token-id IR's openvino/report.json is found when
    // model.xml.report.json is absent.
    let mut named = path.clone().into_os_string();
    named.push(".report.json");
    let named = std::path::PathBuf::from(named);
    if named.is_file() {
        return Some(named);
    }
    path.parent().map(|d| d.join("report.json")).filter(|p| p.is_file())
}

/// The strings a script report or a copied produced_by carries.
fn report_strings(name: &str, reported: &Value) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for key in ["settings", "args"] {
        let Some(arr) = reported.get(key).filter(|v| !v.is_null()) else { continue };
        let arr = arr.as_array().ok_or(format!("{name}: the report's {key} is not a list"))?;
        for v in arr {
            out.push(v.as_str().ok_or(format!("{name}: a report {key} entry is not a string"))?.to_owned());
        }
    }
    if out.is_empty() {
        return Err(format!("{name}: the report has no settings or args"));
    }
    Ok(out)
}

/// Two OpenVINO IRs in one directory must not share a receipt. The
/// embeddings artifact's report carries `cut=embeddings` and
/// `cut_max_abs_diff`. The token-id IR's report carries neither.
fn require_ir_report(c: &crate::convert::Conversion, report: &Path, reported: &Value) -> Result<()> {
    if c.script != crate::convert::ONNX_TO_OPENVINO_IR {
        return Ok(());
    }
    let fields = report_strings(&c.name, reported)?;
    let embeddings = c.args.iter().any(|a| a == "embeddings");
    let has_cut = fields.iter().any(|s| s == "cut=embeddings");
    let has_diff = fields.iter().any(|s| s.starts_with("cut_max_abs_diff="));
    let path = report.display();
    if embeddings && !(has_cut && has_diff) {
        return Err(format!(
            "{}: {path} is not the embeddings cut. The report needs cut=embeddings and cut_max_abs_diff. This \
             artifact's report is {}.report.json",
            c.name, c.file
        ));
    }
    if !embeddings && (has_cut || has_diff) {
        return Err(format!(
            "{}: {path} is an embeddings cut (it has cut=embeddings or cut_max_abs_diff). The token-id IR's report \
             does not",
            c.name
        ));
    }
    Ok(())
}

/// A script report (`tool`, `tool_version`, `settings`) or a finished
/// `produced_by`. The OpenVINO host script names no container; that run
/// is recorded as `host`, and as not reproducible, because it ran once.
fn conversion_produced_by(reported: &Value, c: &crate::convert::Conversion) -> Result<Value> {
    if reported.get("settings").is_some() {
        let container = match reported["container"].as_str() {
            Some(container) => container.to_owned(),
            None if c.script == crate::convert::ONNX_TO_OPENVINO_IR => "host".to_owned(),
            None => {
                return Err(format!(
                    "{}: the report has no container. A conversion that ran in an image names that image; only the \
                     OpenVINO host script may omit it",
                    c.name
                ));
            }
        };
        let reproducible = reported["reproducible"].as_bool().unwrap_or(false);
        return crate::convert::produced_by(reported, &container, c, reproducible);
    }
    // A produced_by copied from a bundle already sealed. Its args stay as
    // that seal wrote them.
    let tool = reported["tool"].as_str().ok_or(format!("{}: the report has no tool", c.name))?;
    let tool_version =
        reported["tool_version"].as_str().ok_or(format!("{}: the report has no tool_version", c.name))?;
    let container = reported["container"].as_str().ok_or(format!(
        "{}: the report has no container. A conversion that ran in an image names that image; only the OpenVINO \
         host script may omit it",
        c.name
    ))?;
    let args = reported["args"].as_array().ok_or(format!("{}: the report has no args or settings", c.name))?;
    let mut pb = json!({
        "tool": tool,
        "tool_version": tool_version,
        "container": container,
        "args": args,
        "reproducible": reported["reproducible"].as_bool().unwrap_or(false),
    });
    if !c.from.is_empty() {
        pb["from"] = json!(c.from);
    }
    if !c.inputs.is_empty() {
        pb["inputs"] = json!(c.inputs);
    }
    Ok(pb)
}

/// Every path the manifest names, which is what `files` must list.
pub fn named_paths(manifest: &Value) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    let mut add = |v: &Value, what: &str| -> Result<()> {
        let p = v.as_str().ok_or(format!("manifest.{what}: not a path"))?;
        check_rel(p)?;
        out.insert(p.to_owned());
        Ok(())
    };
    add(&manifest["tokenizer"]["file"], "tokenizer.file")?;
    add(&manifest["reference"]["file"], "reference.file")?;
    if let Some(l) = manifest["model"].get("license_file") {
        add(l, "model.license_file")?;
    }
    for (i, a) in manifest["artifacts"].as_array().ok_or("manifest.artifacts: missing")?.iter().enumerate() {
        for f in a["files"].as_array().ok_or(format!("manifest.artifacts[{i}].files: missing"))? {
            add(f, &format!("artifacts[{i}].files"))?;
        }
        for f in a["produced_by"]["inputs"].as_array().into_iter().flatten() {
            add(f, &format!("artifacts[{i}].produced_by.inputs"))?;
        }
    }
    Ok(out)
}

/// Write `manifest.json` for the files now in `bundle`, then verify it.
/// `produced_by` is the reference's, from the run, and `converted` each
/// converted artifact's name and `produced_by`, from its run.
pub fn seal(recipe: &Recipe, bundle: &Path, produced_by: Value, converted: Vec<(String, Value)>) -> Result<()> {
    let mut m = recipe.manifest.clone();
    m["reference"]["produced_by"] = produced_by;
    let made: Vec<String> = converted.iter().map(|(n, _)| n.clone()).collect();
    let want: Vec<String> = crate::convert::conversions(recipe)?.into_iter().map(|c| c.name).collect();
    if made != want {
        return Err(format!("the recipe converts {want:?}, and the runs made {made:?}"));
    }
    for (name, pb) in converted {
        let artifacts = m["artifacts"].as_array_mut().ok_or("manifest.artifacts: missing")?;
        let a = artifacts.iter_mut().find(|a| a["name"] == name.as_str()).ok_or(format!("no artifact {name}"))?;
        a["produced_by"] = pb;
    }
    let mut files = Vec::new();
    for p in named_paths(&m)? {
        let bytes = fs::read(bundle.join(&p)).map_err(|e| format!("{p}: {e}"))?;
        files.push(json!({ "path": p, "size": bytes.len(), "sha256": sha256_hex(&bytes) }));
    }
    m["files"] = Value::Array(files);
    // Through the core's own parser, so the tool writes only what the
    // loader accepts, in the one canonical form.
    let parsed = Manifest::parse(&serde_json::to_vec(&m).unwrap()).map_err(|e| e.message)?;
    let mut bytes = serde_json::to_vec_pretty(&parsed).unwrap();
    bytes.push(b'\n');
    crate::fetch::write_atomic(&bundle.join("manifest.json"), &bytes)?;
    verify(bundle)
}

/// Check a bundle the way a machine will: loader rules 1 to 5 through the
/// core (the core's tokenizer must give the reference's ids exactly), every
/// listed file by size and hash, nothing unlisted in the directory, and a
/// reference whose vectors are real.
pub fn verify(bundle: &Path) -> Result<()> {
    let b = Bundle::open(bundle).map_err(|e| e.message)?;
    Tokenizer::load(&b).map_err(|e| e.message)?;
    let m = &b.manifest;
    let listed: BTreeSet<String> = m.files.iter().map(|f| f.path.clone()).collect();
    for p in &listed {
        b.read_verified(p).map_err(|e| e.message)?;
    }
    for p in walk(bundle, bundle)? {
        if p != "manifest.json" && !listed.contains(&p) {
            return Err(format!("{p} is in the bundle directory but not in files"));
        }
    }

    let bytes = b.read_verified(&m.reference.file).map_err(|e| e.message)?;
    let st = safetensors::File::parse(&m.reference.file, &bytes).map_err(|e| e.message)?;
    let emb = st.get("embeddings", Dtype::F32, 2).map_err(|e| e.message)?;
    let embed = m.embed();
    let (n, dim) = (m.reference.cases.len(), embed.dim as usize);
    if emb.shape != [n as u64, dim as u64] {
        return Err(format!("{}: embeddings are {:?}, not [{n}, {dim}]", m.reference.file, emb.shape));
    }
    let values = emb.f32s();
    for (i, row) in values.chunks_exact(dim).enumerate() {
        let norm = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        if !row.iter().all(|v| v.is_finite()) || norm == 0.0 {
            return Err(format!("{}: case {i} has no real vector", m.reference.file));
        }
        if embed.normalize == Normalize::L2 && (norm - 1.0).abs() > 1e-4 {
            return Err(format!("{}: case {i} has norm {norm}, and the bundle says NORMALIZE_L2", m.reference.file));
        }
    }
    Ok(())
}

fn walk(root: &Path, dir: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for e in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let path = e.map_err(|e| e.to_string())?.path();
        let meta = fs::symlink_metadata(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if meta.is_dir() {
            out.extend(walk(root, &path)?);
        } else {
            let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            out.push(rel);
        }
    }
    Ok(out)
}
