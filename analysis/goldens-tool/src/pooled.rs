//! The mean-pooled all-MiniLM-L6-v2 ONNX model the SentenceVectorsDL
//! goldens run, derived from the upstream export [`fetch`] downloaded.
//!
//! SentenceVectorsDL returns the first row of a graph's first output,
//! which in the upstream export is the [CLS] row of the last hidden
//! states. `container/mean_pool_onnx.py` writes a copy whose first output
//! is the mean over the positions the attention mask keeps, so the
//! generator gets the vector sentence-transformers pools. It runs in the
//! bundle tool's reference image ([`REFERENCE_IMAGE`]), which has the
//! `onnx` package, with no network, on the upstream file read only; the
//! copy is checked against [`POOLED_SHA512`] before it is kept.
//!
//! [`fetch`]: crate::models::fetch

use crate::command::run;
use crate::models::sha512_file;
use crate::paths;
use std::fs;
use std::path::Path;
use std::process::Command;

/// The upstream export, as `LOCAL_MODELS` names it.
pub const SOURCE: &str = "all-minilm-l6-v2.onnx";

/// The derived copy, as the generator's `Runs.SENTENCE_VECTORS_ONNX` names it.
pub const POOLED: &str = "all-minilm-l6-v2-mean.onnx";

/// The SHA-512 of the copy: the derivation writes the same bytes on every
/// run.
pub const POOLED_SHA512: &str = "893ab896c6008bf8534b109dc25544e7225fe789d33afb4b03236c796fad8599\
                                 e36b89128dc6a0803297f93e267c28555f0385cf3e6acea29355df19cc08e8c6";

/// The bundle tool's reference image (bundle/reference/Dockerfile), by
/// the image id the classic store reports for it.
pub const REFERENCE_IMAGE: &str = "sha256:8d07bd26184d6efe28889ca6a78e3d672f4eaa31a14930f1472cad243476914d";

/// Makes sure `dir` holds [`POOLED`] with [`POOLED_SHA512`], deriving it
/// from [`SOURCE`] when it is missing. A copy already there whose hash
/// differs is reported and left for the caller to look at.
pub fn derive(dir: &Path) -> Result<(), String> {
    let target = dir.join(POOLED);
    if target.exists() {
        return check(&target).map(|()| println!("{POOLED}: present, SHA-512 matches"));
    }
    let script = paths::tool_dir().join("container/mean_pool_onnx.py");
    let work = dir.join(".pooled.part");
    if work.exists() {
        fs::remove_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    }
    fs::create_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    let canonical = |p: &Path| fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display()));
    let mut cmd = Command::new("docker");
    cmd.args(["run", "--rm", "--network", "none", "--mount"])
        .arg(format!("type=bind,src={},dst=/source.onnx,readonly", canonical(&dir.join(SOURCE))?.display()))
        .arg("--mount")
        .arg(format!("type=bind,src={},dst=/mean_pool_onnx.py,readonly", canonical(&script)?.display()))
        .arg("--mount")
        .arg(format!("type=bind,src={},dst=/out", canonical(&work)?.display()));
    if let Some(user) = current_user() {
        cmd.args(["--user", &user]);
    }
    cmd.args(["--entrypoint", "python", REFERENCE_IMAGE, "/mean_pool_onnx.py", "/source.onnx"])
        .args(["/out/pooled.onnx", "/out/pooled.json"]);
    run(&mut cmd).map_err(|e| {
        format!("{e}\nderiving {POOLED} needs the bundle tool's reference image {REFERENCE_IMAGE} (bundle/README.md)")
    })?;
    let made = work.join("pooled.onnx");
    check(&made)?;
    fs::rename(&made, &target).map_err(|e| format!("{}: {e}", target.display()))?;
    fs::remove_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    println!("{POOLED}: derived from {SOURCE}, SHA-512 matches");
    Ok(())
}

fn check(path: &Path) -> Result<(), String> {
    let actual = sha512_file(path)?;
    if actual != POOLED_SHA512 {
        return Err(format!("{}: SHA-512 is {actual}, the recipe has {POOLED_SHA512}", path.display()));
    }
    Ok(())
}

/// `uid:gid` of the calling user, so the files written are theirs.
fn current_user() -> Option<String> {
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
