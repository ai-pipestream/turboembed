//! Reference programs run in containers pinned by digest, through the
//! docker command line. Every command is kept as its argv, so the record
//! says exactly what ran.

use std::process::Command;

use crate::Result;

/// `name@sha256:<64 hex>`, or an error saying what is wrong with it: an
/// image is pinned by its content, never by a tag.
pub fn check_pinned<'a>(what: &str, image: &'a str) -> Result<&'a str> {
    turbo::record::pinned(image)
        .ok_or_else(|| format!("{what} {image:?} is not pinned as name@sha256:<64 hex>"))
        .map(|_| image)
}

/// The commands run so far, in order, as the record keeps them.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Log {
    pub commands: Vec<Vec<String>>,
}

impl Log {
    /// Run `argv`, keeping it in the log; its standard output on success,
    /// else an error with the command, its status and its output.
    pub fn run(&mut self, argv: &[String]) -> Result<String> {
        self.commands.push(argv.to_vec());
        let out = Command::new(&argv[0]).args(&argv[1..]).output().map_err(|e| format!("{}: {e}", argv.join(" ")))?;
        if !out.status.success() {
            return Err(format!(
                "{} failed ({}):\n{}{}",
                argv.join(" "),
                out.status,
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

pub fn argv(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| (*s).to_owned()).collect()
}

/// The image is present on this machine: the tool never pulls, so what
/// runs is what the digest names and was fetched on purpose.
pub fn require_image(log: &mut Log, image: &str) -> Result<()> {
    log.run(&argv(&["docker", "image", "inspect", "--format", "{{.Id}}", image]))
        .map(|_| ())
        .map_err(|e| format!("the image {image} is not present; pull it first ({e})"))
}

/// The host port `docker port` gives for a published container port:
/// the first line, `127.0.0.1:32768` or `[::1]:32768`.
pub fn parse_port(out: &str) -> Result<u16> {
    let line = out.lines().next().unwrap_or("").trim();
    line.rsplit_once(':')
        .and_then(|(_, p)| p.parse().ok())
        .ok_or_else(|| format!("docker port gave {out:?}, not host:port"))
}

/// A container started detached, removed when this goes.
pub struct Running {
    pub name: String,
}

impl Drop for Running {
    fn drop(&mut self) {
        let _ = Command::new("docker").args(["rm", "--force", &self.name]).output();
    }
}
