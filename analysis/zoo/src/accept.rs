//! The acceptance record: which user accepted which licence, and when.
//!
//! The record is a strict JSON file in one canonical form, as the bundle
//! manifest is (docs/zoo.md, "Acceptance file"). The zoo reads it before
//! every fetch of an artifact whose class needs acceptance and never
//! writes it on its own: only [`Acceptances::accept`], called on the
//! user's explicit action, adds to it.
//!
//! Version 2 adds two optional fields to an acceptance: the artifact whose
//! page showed the terms, and the SHA-256 of the terms text as that page
//! displayed it. A file is written as version 1 while no acceptance holds
//! either field, so a build that reads only version 1 still reads it.

use std::fmt;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::digest::Sha256;
use crate::error::ZooError;
use crate::fetch::write_atomic;
use crate::index::{ArtifactId, Url};
use crate::licence::{Licence, LicenceClass, Requirement};

/// The newest `acceptances_version` this build reads. It reads 1 as well,
/// and writes 1 while no acceptance holds a field version 2 added.
pub const ACCEPTANCES_VERSION: u32 = 2;

/// Who accepted: a user name, 1 to 256 bytes, no control characters and
/// no leading or trailing whitespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UserId(String);

impl UserId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for UserId {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let ok = !s.is_empty() && s.len() <= 256 && !s.chars().any(char::is_control) && s.trim() == s;
        if ok { Ok(UserId(s.to_owned())) } else { Err(format!("{s:?} is not a user name")) }
    }
}

impl fmt::Display for UserId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for UserId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for UserId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
    }
}

/// One user's acceptance of one set of terms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Acceptance {
    pub user: UserId,
    pub license: Licence,
    /// The text the user was shown: `license`'s `legal_url`, or, for
    /// `LicenseRef-RestrictedPublic`, the terms page of the data's host.
    pub license_text: String,
    /// When, in seconds since the Unix epoch, UTC.
    pub accepted_at: u64,
    /// The artifact whose page showed the terms, for audit only: the
    /// acceptance covers every artifact under the same terms (version 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<ArtifactId>,
    /// The SHA-256 of `license_text` as the page displayed it (version 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_text_sha256: Option<Sha256>,
}

impl Acceptance {
    fn requirement(&self) -> Requirement {
        Requirement { licence: self.license, text: self.license_text.clone() }
    }

    /// Whether the acceptance holds a field version 1 does not have.
    fn needs_version_2(&self) -> bool {
        self.artifact.is_some() || self.license_text_sha256.is_some()
    }
}

/// What the page that asked for an acceptance showed: the artifact it was
/// for, and the SHA-256 of the terms text as it displayed it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Shown {
    pub artifact: Option<ArtifactId>,
    pub text_sha256: Option<Sha256>,
}

/// Check that `shown` names the text of `r`.
fn check_shown(r: &Requirement, shown: &Shown) -> Result<(), String> {
    match shown.text_sha256 {
        Some(h) if h != Sha256::of(r.text.as_bytes()) => {
            Err(format!("license_text_sha256 {h} is not the SHA-256 of {:?}", r.text))
        }
        _ => Ok(()),
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptancesFile {
    acceptances_version: u32,
    accepted: Vec<Acceptance>,
}

/// Every acceptance in one file, in the order they were made.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Acceptances {
    accepted: Vec<Acceptance>,
}

/// Check that `r` is terms a user can accept: its licence's class needs
/// acceptance under some policy (permissive licences need none; refused
/// ones cannot be accepted at all), and its text is the licence's own
/// address, or, for a licence without one, an `https://` address.
fn check(r: &Requirement) -> Result<(), String> {
    let class = r.licence.class();
    if !matches!(class, LicenceClass::Lgpl | LicenceClass::ShareAlike | LicenceClass::Gpl | LicenceClass::LocalOnly) {
        return Err(format!("{} is {class}: there is nothing to accept", r.licence));
    }
    match r.licence.legal_url() {
        Some(u) if u == r.text => Ok(()),
        Some(u) => Err(format!("{:?} is not the text of {}, which is at {u}", r.text, r.licence)),
        None => r.text.parse::<Url>().map(|_| ()).map_err(|e| format!("{}: {e}", r.licence)),
    }
}

impl Acceptances {
    /// Parse and check an acceptance file. An unknown field, a version
    /// other than 1 or 2, a version 2 field in a version 1 file, terms
    /// that cannot be accepted (a permissive or refused licence, a
    /// `license_text` that is not the licence's own address), a
    /// `license_text_sha256` that is not the digest of `license_text`, or
    /// the same user and terms twice is `ZooError::Acceptances`.
    pub fn parse(bytes: &[u8]) -> Result<Acceptances, ZooError> {
        let file: AcceptancesFile = serde_json::from_slice(bytes).map_err(|e| ZooError::Acceptances(e.to_string()))?;
        let version = file.acceptances_version;
        if !(1..=ACCEPTANCES_VERSION).contains(&version) {
            return Err(ZooError::Acceptances(format!(
                "acceptances_version {version} is not 1 to {ACCEPTANCES_VERSION}"
            )));
        }
        let mut out = Acceptances::default();
        for (i, a) in file.accepted.into_iter().enumerate() {
            let r = a.requirement();
            let shown = Shown { artifact: a.artifact.clone(), text_sha256: a.license_text_sha256 };
            check(&r)
                .and_then(|()| check_shown(&r, &shown))
                .map_err(|e| ZooError::Acceptances(format!("accepted[{i}] ({}): {e}", a.user)))?;
            if version == 1 && a.needs_version_2() {
                return Err(ZooError::Acceptances(format!(
                    "accepted[{i}] ({}): artifact and license_text_sha256 need acceptances_version 2",
                    a.user
                )));
            }
            if out.has(&a.user, &r) {
                return Err(ZooError::Acceptances(format!("accepted[{i}]: {} accepted {r} twice", a.user)));
            }
            out.accepted.push(a);
        }
        Ok(out)
    }

    /// The file at `path`, or no acceptances when there is no file.
    pub fn load(path: &Path) -> Result<Acceptances, ZooError> {
        match fs::read(path) {
            Ok(b) => Acceptances::parse(&b).map_err(|e| match e {
                ZooError::Acceptances(m) => ZooError::Acceptances(format!(
                    "{}: {m}; fix the file by hand, or remove it to accept again from nothing",
                    path.display()
                )),
                other => other,
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Acceptances::default()),
            Err(e) => Err(ZooError::Io { path: path.to_path_buf(), message: e.to_string() }),
        }
    }

    /// Write the canonical form to `path`, through a temporary name.
    pub fn save(&self, path: &Path) -> Result<(), ZooError> {
        write_atomic(path, self.to_canonical_json().as_bytes())
    }

    /// The canonical form: serde_json's pretty printer and a final newline,
    /// at version 1 unless an acceptance holds a version 2 field.
    pub fn to_canonical_json(&self) -> String {
        let version = if self.accepted.iter().any(Acceptance::needs_version_2) { 2 } else { 1 };
        let file = AcceptancesFile { acceptances_version: version, accepted: self.accepted.clone() };
        let mut s = serde_json::to_string_pretty(&file).expect("an acceptance file always serializes");
        s.push('\n');
        s
    }

    /// Record that `user` accepted `licence`, whose text is its
    /// `legal_url`, at `accepted_at` (seconds since the Unix epoch). Call it
    /// only on the user's explicit action, after showing them that text.
    /// Accepting again is a no-op that keeps the first time. A permissive
    /// or refused licence, or `LicenseRef-RestrictedPublic` (whose terms
    /// are each host's own: [`Acceptances::accept_terms`]), is
    /// `ZooError::Acceptances`.
    pub fn accept(&mut self, user: &UserId, licence: Licence, accepted_at: u64) -> Result<(), ZooError> {
        let Some(text) = licence.legal_url() else {
            return Err(ZooError::Acceptances(format!(
                "{licence} names no single text; accept the terms address an artifact gives for it"
            )));
        };
        self.accept_terms(user, &Requirement { licence, text: text.to_owned() }, accepted_at)
    }

    /// Record that `user` accepted the terms `r`, as
    /// [`crate::Artifact::licences_to_accept`] or
    /// [`crate::ZooError::NotAccepted`] gives them. The same rules as
    /// [`Acceptances::accept`]; for `LicenseRef-RestrictedPublic` the text is
    /// the host's terms page, so each host's terms are accepted on their
    /// own.
    pub fn accept_terms(&mut self, user: &UserId, r: &Requirement, accepted_at: u64) -> Result<(), ZooError> {
        self.accept_shown(user, r, &Shown::default(), accepted_at)
    }

    /// [`Acceptances::accept_terms`] from a page that showed the terms:
    /// `shown` names the artifact it was for and the SHA-256 of the text
    /// as it displayed it, recorded for audit. A digest that is not the
    /// SHA-256 of `r.text` is `ZooError::Acceptances`. Accepting again
    /// keeps the first acceptance as it was.
    pub fn accept_shown(
        &mut self,
        user: &UserId,
        r: &Requirement,
        shown: &Shown,
        accepted_at: u64,
    ) -> Result<(), ZooError> {
        check(r).and_then(|()| check_shown(r, shown)).map_err(ZooError::Acceptances)?;
        if !self.has(user, r) {
            self.accepted.push(Acceptance {
                user: user.clone(),
                license: r.licence,
                license_text: r.text.clone(),
                accepted_at,
                artifact: shown.artifact.clone(),
                license_text_sha256: shown.text_sha256,
            });
        }
        Ok(())
    }

    /// Withdraw `user`'s acceptance of `r`. Returns whether there was one.
    /// Files already fetched stay on disk, but the zoo hands none of them
    /// out again until the terms are accepted again.
    pub fn revoke(&mut self, user: &UserId, r: &Requirement) -> bool {
        let before = self.accepted.len();
        self.accepted.retain(|a| !(a.user == *user && a.requirement() == *r));
        before != self.accepted.len()
    }

    /// Whether `user` accepted the terms `r`.
    pub fn has(&self, user: &UserId, r: &Requirement) -> bool {
        self.find(user, r).is_some()
    }

    /// `user`'s acceptance of the terms `r`, if there is one.
    pub fn find(&self, user: &UserId, r: &Requirement) -> Option<&Acceptance> {
        self.accepted.iter().find(|a| a.user == *user && a.license == r.licence && a.license_text == r.text)
    }

    /// `user`'s acceptances, in the order they were made.
    pub fn of_user<'a>(&'a self, user: &'a UserId) -> impl Iterator<Item = &'a Acceptance> {
        self.accepted.iter().filter(move |a| a.user == *user)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Acceptance> {
        self.accepted.iter()
    }
}

#[cfg(test)]
mod tests;
