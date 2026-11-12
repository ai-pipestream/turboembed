//! Subcommands for the hardened zoo container (`docs/zoo-container.md`):
//! `zoo-start`, `zoo-health`, and `zoo-bucket`.
//!
//! Each runs either as `turbo-kserve <subcommand>` or through a symlink whose
//! file name is `<subcommand>`, so the runtime image needs neither a shell nor
//! `curl`.

mod bucket;
mod env;
mod health;
mod start;

use std::ffi::OsStr;
use std::path::Path;
use std::process::ExitCode;

pub(crate) const SUBCOMMAND_START: &str = "zoo-start";
pub(crate) const SUBCOMMAND_HEALTH: &str = "zoo-health";
pub(crate) const SUBCOMMAND_BUCKET: &str = "zoo-bucket";

/// Which zoo helper an invocation names, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Subcommand {
    Start,
    Health,
    Bucket,
}

impl Subcommand {
    fn from_name(name: &str) -> Option<Subcommand> {
        match name {
            SUBCOMMAND_START => Some(Subcommand::Start),
            SUBCOMMAND_HEALTH => Some(Subcommand::Health),
            SUBCOMMAND_BUCKET => Some(Subcommand::Bucket),
            _ => None,
        }
    }

    /// Detects a zoo helper from `argv[0]`'s file name (for symlink multicall
    /// dispatch) or from `argv[1]` (for `turbo-kserve zoo-*`), returning the
    /// helper and the remaining arguments.
    pub(crate) fn detect(program: Option<&OsStr>, first: Option<&str>) -> Option<(Subcommand, bool)> {
        if let Some(name) = program.and_then(|p| Path::new(p).file_name()).and_then(OsStr::to_str)
            && let Some(cmd) = Subcommand::from_name(name)
        {
            return Some((cmd, false));
        }
        first.and_then(Subcommand::from_name).map(|cmd| (cmd, true))
    }

    /// Runs the selected zoo subcommand to completion.
    pub(crate) async fn run(self, args: Vec<String>) -> ExitCode {
        match self {
            Subcommand::Start => start::run(args).await,
            Subcommand::Health => health::run(args).await,
            Subcommand::Bucket => bucket::run(args).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::{SUBCOMMAND_BUCKET, SUBCOMMAND_HEALTH, SUBCOMMAND_START, Subcommand};

    #[test]
    fn detect_matches_both_multicall_program_names_and_explicit_subcommands() {
        assert_eq!(
            Subcommand::detect(Some(OsStr::new("/opt/turbo/bin/zoo-start")), Some("--catalog-prefetch=permissive")),
            Some((Subcommand::Start, false))
        );
        assert_eq!(Subcommand::detect(Some(OsStr::new("zoo-health")), None), Some((Subcommand::Health, false)));
        assert_eq!(
            Subcommand::detect(Some(OsStr::new("/opt/turbo/bin/zoo-bucket")), None),
            Some((Subcommand::Bucket, false))
        );
        assert_eq!(
            Subcommand::detect(Some(OsStr::new("turbo-kserve")), Some(SUBCOMMAND_START)),
            Some((Subcommand::Start, true))
        );
        assert_eq!(
            Subcommand::detect(Some(OsStr::new("/opt/turbo/bin/turbo-kserve")), Some(SUBCOMMAND_HEALTH)),
            Some((Subcommand::Health, true))
        );
        assert_eq!(
            Subcommand::detect(Some(OsStr::new("turbo-kserve")), Some(SUBCOMMAND_BUCKET)),
            Some((Subcommand::Bucket, true))
        );
        assert_eq!(Subcommand::detect(Some(OsStr::new("turbo-kserve")), Some("health")), None);
        assert_eq!(Subcommand::detect(Some(OsStr::new("turbo-kserve")), Some("--zoo")), None);
    }
}
