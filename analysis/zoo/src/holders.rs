//! Holders: the sources each user registered, in the order they are tried
//! (#1744). A holder is a name the user picked, a source address
//! (`file:///dir`, `s3://bucket/prefix`, `gdrive://<folder-id>`, a zoo
//! server, or any scheme an application adds) and how it signs in.
//!
//! The record is a strict JSON file in one canonical form, keyed by user
//! like the acceptance file. It holds sign-in secrets (a Drive grant's
//! refresh token, an API key), so it is written readable by its owner
//! alone on Unix. Opening a holder into a source is the kind registry's
//! work (`turbo-zoo-remote`), not this file's.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::accept::UserId;
use crate::error::ZooError;
use crate::fetch::write_atomic_as;

/// The `holders_version` this build reads and writes.
pub const HOLDERS_VERSION: u32 = 1;

/// One registered source.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Holder {
    /// What the user calls it: 1 to 64 characters, no control character,
    /// no comma, no leading or trailing whitespace; unique per user.
    pub name: String,
    /// The source's address, `<scheme>://...`.
    pub address: String,
    #[serde(default, skip_serializing_if = "SignIn::is_environment")]
    pub sign_in: SignIn,
}

/// How a holder signs in.
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SignIn {
    /// From the process environment, as `TURBO_ZOO_SOURCE` does
    /// (`AWS_*`, `GOOGLE_OAUTH_ACCESS_TOKEN`, `GOOGLE_API_KEY`), or not at
    /// all for a source that needs none.
    #[default]
    Environment,
    /// A Google OAuth grant: the client it was made with and its refresh
    /// token.
    GoogleOauth {
        client_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        client_secret: Option<String>,
        refresh_token: String,
    },
    /// An API key, for a source that takes one (a Drive folder shared by
    /// link).
    ApiKey { key: String },
}

impl SignIn {
    pub fn is_environment(&self) -> bool {
        matches!(self, SignIn::Environment)
    }

    /// Every secret it holds, each printable ASCII and not empty.
    fn check(&self) -> Result<(), String> {
        let ok = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_graphic());
        let fine = match self {
            SignIn::Environment => true,
            SignIn::GoogleOauth { client_id, client_secret, refresh_token } => {
                ok(client_id) && ok(refresh_token) && client_secret.as_deref().is_none_or(ok)
            }
            SignIn::ApiKey { key } => ok(key),
        };
        if fine { Ok(()) } else { Err("a sign-in secret is empty or holds spaces or control characters".into()) }
    }
}

impl std::fmt::Debug for SignIn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SignIn::Environment => "Environment",
            SignIn::GoogleOauth { .. } => "GoogleOauth(..)",
            SignIn::ApiKey { .. } => "ApiKey(..)",
        })
    }
}

impl std::fmt::Debug for Holder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Holder")
            .field("name", &self.name)
            .field("address", &self.address)
            .field("sign_in", &self.sign_in)
            .finish()
    }
}

impl Holder {
    pub fn new(name: &str, address: &str, sign_in: SignIn) -> Result<Holder, ZooError> {
        let h = Holder { name: name.to_owned(), address: address.to_owned(), sign_in };
        h.check().map_err(ZooError::Holders)?;
        Ok(h)
    }

    /// The scheme of its address, before `://`.
    pub fn scheme(&self) -> &str {
        self.address.split_once("://").map_or("", |(s, _)| s)
    }

    fn check(&self) -> Result<(), String> {
        let n = &self.name;
        let named = !n.is_empty() && n.chars().count() <= 64 && n.trim() == n;
        if !named || n.chars().any(|c| c.is_control() || c == ',') {
            return Err(format!("{n:?}: a holder name is 1 to 64 characters, with no comma or control character"));
        }
        let a = &self.address;
        let scheme = self.scheme();
        let schemed = !scheme.is_empty()
            && scheme.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'+' || b == b'-');
        if !schemed || a.len() == scheme.len() + 3 || a.chars().any(|c| c.is_control() || c == ',') {
            return Err(format!("{n}: {a:?} is not <scheme>://... with no comma or control character"));
        }
        self.sign_in.check().map_err(|e| format!("{n}: {e}"))
    }
}

/// Every user's holders.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Holders {
    users: BTreeMap<UserId, Vec<Holder>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HoldersFile {
    holders_version: u32,
    users: Vec<UserHolders>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct UserHolders {
    user: UserId,
    holders: Vec<Holder>,
}

impl Holders {
    /// The holders in `path`; none when there is no file.
    pub fn load(path: &Path) -> Result<Holders, ZooError> {
        match fs::read(path) {
            Ok(b) => Holders::parse(&b).map_err(|e| match e {
                ZooError::Holders(m) => ZooError::Holders(format!("{}: {m}", path.display())),
                other => other,
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Holders::default()),
            Err(e) => Err(ZooError::Io { path: path.to_path_buf(), message: e.to_string() }),
        }
    }

    /// The file's bytes, checked as [`Holder::new`] checks each holder:
    /// one entry per user, names unique per user.
    pub fn parse(bytes: &[u8]) -> Result<Holders, ZooError> {
        let file: HoldersFile = serde_json::from_slice(bytes).map_err(|e| ZooError::Holders(e.to_string()))?;
        if file.holders_version != HOLDERS_VERSION {
            return Err(ZooError::Holders(format!(
                "holders_version {} is not {HOLDERS_VERSION}, the one this build reads",
                file.holders_version
            )));
        }
        let mut out = Holders::default();
        for entry in file.users {
            if out.users.contains_key(&entry.user) {
                return Err(ZooError::Holders(format!("{} is listed twice", entry.user)));
            }
            let mut list: Vec<Holder> = Vec::new();
            for h in entry.holders {
                h.check().map_err(|e| ZooError::Holders(format!("{}: {e}", entry.user)))?;
                if list.iter().any(|o| o.name == h.name) {
                    return Err(ZooError::Holders(format!("{}: {} is named twice", entry.user, h.name)));
                }
                list.push(h);
            }
            out.users.insert(entry.user, list);
        }
        Ok(out)
    }

    /// Writes the canonical form to `path` through a temporary name,
    /// readable by its owner alone on Unix.
    pub fn save(&self, path: &Path) -> Result<(), ZooError> {
        write_atomic_as(path, self.to_canonical_json().as_bytes(), true)
    }

    /// serde_json's pretty printer and a final newline; users in order,
    /// a user with no holder left out.
    pub fn to_canonical_json(&self) -> String {
        let users = self
            .users
            .iter()
            .filter(|(_, l)| !l.is_empty())
            .map(|(user, holders)| UserHolders { user: user.clone(), holders: holders.clone() })
            .collect();
        let file = HoldersFile { holders_version: HOLDERS_VERSION, users };
        let mut s = serde_json::to_string_pretty(&file).expect("a holders file always serializes");
        s.push('\n');
        s
    }

    /// `user`'s holders, in the order they are tried.
    pub fn list(&self, user: &UserId) -> &[Holder] {
        self.users.get(user).map_or(&[], Vec::as_slice)
    }

    /// Every user with a holder.
    pub fn users(&self) -> impl Iterator<Item = &UserId> {
        self.users.iter().filter(|(_, l)| !l.is_empty()).map(|(u, _)| u)
    }

    /// Adds `holder` last, or puts it in place of `user`'s holder of the
    /// same name, where that one was.
    pub fn register(&mut self, user: &UserId, holder: Holder) -> Result<(), ZooError> {
        holder.check().map_err(ZooError::Holders)?;
        let list = self.users.entry(user.clone()).or_default();
        match list.iter_mut().find(|h| h.name == holder.name) {
            Some(old) => *old = holder,
            None => list.push(holder),
        }
        Ok(())
    }

    /// Removes `user`'s holder `name`, and hands it back.
    pub fn remove(&mut self, user: &UserId, name: &str) -> Option<Holder> {
        let list = self.users.get_mut(user)?;
        let at = list.iter().position(|h| h.name == name)?;
        Some(list.remove(at))
    }

    /// Moves `user`'s holder `name` to place `at` (0 is tried first; past
    /// the end is last).
    pub fn move_to(&mut self, user: &UserId, name: &str, at: usize) -> Result<(), ZooError> {
        let list = self.users.get_mut(user).filter(|l| l.iter().any(|h| h.name == name));
        let Some(list) = list else {
            return Err(ZooError::Holders(format!("{user} has no holder called {name}")));
        };
        let from = list.iter().position(|h| h.name == name).expect("found above");
        let h = list.remove(from);
        list.insert(at.min(list.len()), h);
        Ok(())
    }
}
