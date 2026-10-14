//! The acceptance record: which user accepted which licence, and when.
//!
//! The record is a strict JSON file in one canonical form, as the bundle
//! manifest is (docs/zoo.md, "Acceptance file"). The zoo reads it before
//! every fetch of an artifact whose class needs acceptance and never
//! writes it on its own: only [`Acceptances::accept`], called on the
//! user's explicit action, adds to it.

use std::fmt;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::ZooError;
use crate::fetch::write_atomic;
use crate::index::Url;
use crate::licence::{Licence, LicenceClass, Requirement};

/// The only `acceptances_version` this build reads.
pub const ACCEPTANCES_VERSION: u32 = 1;

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
}

impl Acceptance {
    fn requirement(&self) -> Requirement {
        Requirement { licence: self.license, text: self.license_text.clone() }
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
    /// other than 1, terms that cannot be accepted (a permissive or refused
    /// licence, a `license_text` that is not the licence's own address),
    /// or the same user and terms twice is `ZooError::Acceptances`.
    pub fn parse(bytes: &[u8]) -> Result<Acceptances, ZooError> {
        let file: AcceptancesFile = serde_json::from_slice(bytes).map_err(|e| ZooError::Acceptances(e.to_string()))?;
        if file.acceptances_version != ACCEPTANCES_VERSION {
            return Err(ZooError::Acceptances(format!(
                "acceptances_version {} is not {ACCEPTANCES_VERSION}",
                file.acceptances_version
            )));
        }
        let mut out = Acceptances::default();
        for a in file.accepted {
            let r = a.requirement();
            check(&r).map_err(|e| ZooError::Acceptances(format!("{}: {e}", a.user)))?;
            if out.has(&a.user, &r) {
                return Err(ZooError::Acceptances(format!("{} accepted {r} twice", a.user)));
            }
            out.accepted.push(a);
        }
        Ok(out)
    }

    /// The file at `path`, or no acceptances when there is no file.
    pub fn load(path: &Path) -> Result<Acceptances, ZooError> {
        match fs::read(path) {
            Ok(b) => Acceptances::parse(&b),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Acceptances::default()),
            Err(e) => Err(ZooError::Io { path: path.to_path_buf(), message: e.to_string() }),
        }
    }

    /// Write the canonical form to `path`, through a temporary name.
    pub fn save(&self, path: &Path) -> Result<(), ZooError> {
        write_atomic(path, self.to_canonical_json().as_bytes())
    }

    /// The canonical form: serde_json's pretty printer and a final newline.
    pub fn to_canonical_json(&self) -> String {
        let file = AcceptancesFile { acceptances_version: ACCEPTANCES_VERSION, accepted: self.accepted.clone() };
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
        check(r).map_err(ZooError::Acceptances)?;
        if !self.has(user, r) {
            self.accepted.push(Acceptance {
                user: user.clone(),
                license: r.licence,
                license_text: r.text.clone(),
                accepted_at,
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
        self.accepted.iter().any(|a| a.user == *user && a.license == r.licence && a.license_text == r.text)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Acceptance> {
        self.accepted.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(s: &str) -> UserId {
        s.parse().unwrap()
    }

    fn req(l: Licence) -> Requirement {
        Requirement::of(l).unwrap()
    }

    fn restricted(text: &str) -> Requirement {
        Requirement { licence: Licence::RestrictedPublic, text: text.to_owned() }
    }

    #[test]
    fn restricted_terms_are_accepted_per_host() {
        let imdb = restricted("https://ai.stanford.edu/~amaas/data/sentiment/");
        let other = restricted("https://example.org/terms");
        let mut a = Acceptances::default();
        // There is no one text to accept for the licence as a whole.
        assert!(a.accept(&user("alice"), Licence::RestrictedPublic, 0).is_err());
        a.accept_terms(&user("alice"), &imdb, 7).unwrap();
        assert!(a.has(&user("alice"), &imdb));
        assert!(!a.has(&user("alice"), &other));
        assert!(!a.has(&user("bob"), &imdb));
        let back = Acceptances::parse(a.to_canonical_json().as_bytes()).unwrap();
        assert_eq!(back, a);
        assert!(a.revoke(&user("alice"), &imdb));
        assert!(!a.has(&user("alice"), &imdb));
        // The text must be an https address.
        assert!(a.accept_terms(&user("alice"), &restricted("not a url"), 0).is_err());
        assert!(a.accept_terms(&user("alice"), &restricted("http://example.org/terms"), 0).is_err());
        // A licence with its own text accepts that text only.
        let wrong = Requirement { licence: Licence::CcBySa40, text: "https://example.org/terms".into() };
        assert!(a.accept_terms(&user("alice"), &wrong, 0).is_err());
    }

    #[test]
    fn accept_round_trips_through_the_file() {
        let mut a = Acceptances::default();
        a.accept(&user("alice"), Licence::CcBySa40, 1_792_000_000).unwrap();
        a.accept(&user("alice"), Licence::Lgpl30Only, 1_792_000_001).unwrap();
        a.accept(&user("bob"), Licence::CcBySa40, 1_792_000_002).unwrap();
        // A second acceptance keeps the first time.
        a.accept(&user("alice"), Licence::CcBySa40, 1_800_000_000).unwrap();
        let text = a.to_canonical_json();
        let back = Acceptances::parse(text.as_bytes()).unwrap();
        assert_eq!(back, a);
        assert_eq!(back.to_canonical_json(), text);
        assert_eq!(back.iter().next().unwrap().accepted_at, 1_792_000_000);
        assert!(back.has(&user("alice"), &req(Licence::CcBySa40)));
        assert!(!back.has(&user("carol"), &req(Licence::CcBySa40)));
        assert!(!back.has(&user("bob"), &req(Licence::Lgpl30Only)));
    }

    #[test]
    fn the_canonical_form_is_documented() {
        let mut a = Acceptances::default();
        a.accept(&user("alice"), Licence::CcBySa40, 1_792_000_000).unwrap();
        assert_eq!(
            a.to_canonical_json(),
            r#"{
  "acceptances_version": 1,
  "accepted": [
    {
      "user": "alice",
      "license": "CC-BY-SA-4.0",
      "license_text": "https://creativecommons.org/licenses/by-sa/4.0/legalcode",
      "accepted_at": 1792000000
    }
  ]
}
"#
        );
    }

    #[test]
    fn permissive_and_refused_licences_cannot_be_accepted() {
        let mut a = Acceptances::default();
        assert!(a.accept(&user("alice"), Licence::Apache2, 0).is_err());
        assert!(a.accept(&user("alice"), Licence::CcByNcSa40, 0).is_err());
        assert!(a.accept(&user("alice"), Licence::Unconfirmed, 0).is_err());
        assert_eq!(a, Acceptances::default());
    }

    #[test]
    fn revoke_removes_one_users_acceptance() {
        let mut a = Acceptances::default();
        a.accept(&user("alice"), Licence::Gpl30Only, 0).unwrap();
        a.accept(&user("bob"), Licence::Gpl30Only, 0).unwrap();
        assert!(a.revoke(&user("alice"), &req(Licence::Gpl30Only)));
        assert!(!a.revoke(&user("alice"), &req(Licence::Gpl30Only)));
        assert!(!a.has(&user("alice"), &req(Licence::Gpl30Only)));
        assert!(a.has(&user("bob"), &req(Licence::Gpl30Only)));
    }

    #[test]
    fn parse_is_strict() {
        let ok = r#"{"acceptances_version":1,"accepted":[{"user":"a","license":"CC-BY-SA-4.0","license_text":"https://creativecommons.org/licenses/by-sa/4.0/legalcode","accepted_at":1}]}"#;
        assert!(Acceptances::parse(ok.as_bytes()).is_ok());
        for bad in [
            ok.replace("\"acceptances_version\":1", "\"acceptances_version\":2"),
            ok.replace("\"accepted_at\":1", "\"accepted_at\":1,\"note\":\"x\""),
            ok.replace("\"accepted_at\":1", "\"accepted_at\":\"1\""),
            ok.replace(",\"accepted_at\":1", ""),
            ok.replace("CC-BY-SA-4.0", "cc-by-sa-4.0"),
            ok.replace("CC-BY-SA-4.0\",\"license_text\":\"https://creativecommons.org/licenses/by-sa/4.0/legalcode", "MIT\",\"license_text\":\"https://opensource.org/license/mit"),
            ok.replace("by-sa/4.0/legalcode", "by-sa/3.0/legalcode"),
            ok.replace("\"user\":\"a\"", "\"user\":\"\""),
            ok.replace("\"user\":\"a\"", "\"user\":\" a\""),
            ok.replace("}]}", "},{\"user\":\"a\",\"license\":\"CC-BY-SA-4.0\",\"license_text\":\"https://creativecommons.org/licenses/by-sa/4.0/legalcode\",\"accepted_at\":2}]}"),
        ] {
            assert!(Acceptances::parse(bad.as_bytes()).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_missing_file_is_no_acceptances() {
        let dir = std::env::temp_dir().join(format!("turbo-zoo-accept-{}", std::process::id()));
        let path = dir.join("absent").join("acceptances.json");
        assert_eq!(Acceptances::load(&path).unwrap(), Acceptances::default());
        let mut a = Acceptances::default();
        a.accept(&user("alice"), Licence::CcBySa30, 5).unwrap();
        a.save(&path).unwrap();
        assert_eq!(Acceptances::load(&path).unwrap(), a);
        fs::remove_dir_all(&dir).unwrap();
    }
}
