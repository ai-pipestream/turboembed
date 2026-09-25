//! Reference programs run in containers pinned by digest, through the
//! docker command line. Every command is kept as its argv, so the record
//! says exactly what ran, with each host path in it written as a fixed
//! placeholder: records are published, and a path on the machine that
//! made one says nothing about the measurement and may name its user.

use std::process::Command;

use crate::Result;

/// `name@sha256:<64 hex>`, or an error saying what is wrong with it: an
/// image is pinned by its content, never by a tag.
pub fn check_pinned<'a>(what: &str, image: &'a str) -> Result<&'a str> {
    turbo::record::pinned(image)
        .ok_or_else(|| format!("{what} {image:?} is not pinned as name@sha256:<64 hex>"))
        .map(|_| image)
}

/// The bundle directory, as a recorded command names it.
pub const BUNDLE: &str = "<bundle>";

/// The directory the tool writes a program's input files to.
pub const WORK: &str = "<work>";

/// The model directory TEI serves.
pub const TEI_MODEL: &str = "<tei-model>";

/// The commands run so far, in order, as the record keeps them.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Log {
    pub commands: Vec<Vec<String>>,
}

impl Log {
    /// Run `argv`, a command with no host path in it, keeping it in the
    /// log; its standard output on success, else an error with the
    /// command, its status and its output.
    pub fn run(&mut self, argv: &[String]) -> Result<String> {
        self.run_as(argv, argv.to_vec())
    }

    /// Run `argv`, keeping `recorded` in the log in its place: the same
    /// command built with the placeholders above for its host paths. The
    /// error, which is not recorded, names the command as run.
    pub fn run_as(&mut self, argv: &[String], recorded: Vec<String>) -> Result<String> {
        self.commands.push(recorded);
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

    /// Run `argv`, a `docker run` of a reference program, as `run_as`
    /// does, except that the program failing is a result and not an
    /// error: `Ran::Failed` with the line that says why (`failure`).
    /// docker failing, or the program not starting, is still an error.
    pub fn run_program(&mut self, argv: &[String], recorded: Vec<String>, tags: &[&str]) -> Result<Ran> {
        self.commands.push(recorded);
        let out = Command::new(&argv[0]).args(&argv[1..]).output().map_err(|e| format!("{}: {e}", argv.join(" ")))?;
        let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        if out.status.success() {
            return Ok(Ran::Done(stdout.into_owned()));
        }
        match failure(out.status.code(), &format!("{stdout}{stderr}"), tags) {
            Some(why) => Ok(Ran::Failed(why)),
            None => Err(format!("{} failed ({}):\n{stdout}{stderr}", argv.join(" "), out.status)),
        }
    }
}

/// How a reference program run in a container ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Ran {
    /// It exited 0: its standard output.
    Done(String),
    /// It exited non-zero: why, as `failure` gives it.
    Failed(String),
}

/// Why a program in a container failed, from the exit code `docker run`
/// gave and its output: its exit code and the first line carrying one of
/// the program's error `tags`, from after the tag, or its last line when
/// none does. None when docker failed rather than the program: killed by
/// a signal, or 125, 126 or 127, the codes docker keeps for itself.
pub fn failure(code: Option<i32>, out: &str, tags: &[&str]) -> Option<String> {
    let code = code.filter(|c| !(125..=127).contains(c))?;
    let tagged = out.lines().find_map(|l| tags.iter().find_map(|t| l.split_once(t).map(|(_, rest)| rest.trim())));
    let last = || out.lines().map(str::trim).rfind(|l| !l.is_empty());
    let why = tagged.or_else(last).unwrap_or("no output");
    Some(format!("exited with code {code}: {why}"))
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
