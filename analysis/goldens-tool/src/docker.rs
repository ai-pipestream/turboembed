//! Building the generator's image and running it with Docker.

use crate::command::run;
use crate::paths;
use crate::recipe;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Builds the image from `container/Dockerfile` with `java/` as the build
/// context, tags it [`recipe::IMAGE_TAG`] and returns its id.
pub fn build_image() -> Result<String, String> {
    let java = canonical(&paths::java_dir())?;
    let dockerfile = canonical(&paths::dockerfile())?;
    run(Command::new("docker")
        .args(["build", "--quiet", "--tag", recipe::IMAGE_TAG, "--file"])
        .arg(dockerfile)
        .arg(java))?;
    image_id()
}

/// The id of the image tagged [`recipe::IMAGE_TAG`]; fails when there is none.
pub fn image_id() -> Result<String, String> {
    let out = run(Command::new("docker").args(["image", "inspect", "--format", "{{.Id}}", recipe::IMAGE_TAG]))
        .map_err(|e| format!("{e}\nthe image {} is not present; run `turbo-goldens image` first", recipe::IMAGE_TAG))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Runs the generator in the image without a network, with `inputs` at
/// /inputs and `models` at /models read only and `out` at /out, as the
/// calling user so the files written are theirs.
pub fn generate(inputs: &Path, models: &Path, out: &Path) -> Result<(), String> {
    image_id()?;
    let mut cmd = Command::new("docker");
    cmd.args(["run", "--rm", "--network", "none"])
        .arg("--mount")
        .arg(format!("type=bind,src={},dst=/inputs,readonly", canonical(inputs)?.display()))
        .arg("--mount")
        .arg(format!("type=bind,src={},dst=/models,readonly", canonical(models)?.display()))
        .arg("--mount")
        .arg(format!("type=bind,src={},dst=/out", canonical(out)?.display()));
    if let Some(user) = current_user() {
        cmd.args(["--user", &user]);
    }
    cmd.arg(recipe::IMAGE_TAG);
    let o = run(&mut cmd)?;
    print!("{}", String::from_utf8_lossy(&o.stdout));
    Ok(())
}

/// `uid:gid` of the calling user, from `id`, or nothing where `id` is missing.
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

fn canonical(p: &Path) -> Result<std::path::PathBuf, String> {
    fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display()))
}
