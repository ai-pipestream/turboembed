//! Where the zoo keeps its cache and acceptance file, and who the user is,
//! as the `turbo-zoo` command finds them from the environment.

use std::path::{Path, PathBuf};

use crate::accept::UserId;

/// The zoo's home: the artifact cache and the acceptance file under one
/// directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Home {
    dir: PathBuf,
}

impl Home {
    pub fn new(dir: PathBuf) -> Home {
        Home { dir }
    }

    /// `$TURBO_ZOO_HOME`, else `$XDG_DATA_HOME/turbo-zoo`, else
    /// `$HOME/.local/share/turbo-zoo`; an empty variable counts as unset.
    /// `None` when none of the three is set.
    pub fn from_env() -> Option<Home> {
        let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        var("TURBO_ZOO_HOME")
            .or_else(|| var("XDG_DATA_HOME").map(|x| x.join("turbo-zoo")))
            .or_else(|| var("HOME").map(|h| h.join(".local/share/turbo-zoo")))
            .map(Home::new)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The artifact cache, the directory a [`crate::Zoo`] is made with.
    pub fn artifacts(&self) -> PathBuf {
        self.dir.join("artifacts")
    }

    /// The acceptance file, as [`crate::Acceptances::load`] and
    /// [`crate::Acceptances::save`] take it.
    pub fn acceptances(&self) -> PathBuf {
        self.dir.join("acceptances.json")
    }
}

impl UserId {
    /// `$USER`, else `$USERNAME`. An error when neither is set, or the
    /// name is not one [`UserId`] takes.
    pub fn from_env() -> Result<UserId, String> {
        let name = std::env::var("USER").or_else(|_| std::env::var("USERNAME")).map_err(|_| "set USER".to_string())?;
        name.parse()
    }
}
