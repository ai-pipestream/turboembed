//! The licence class of the model a recipe names, by the model zoo's own
//! rules (analysis/zoo/src/licence): the tool refuses a licence the zoo
//! refuses, fetches a class that needs acceptance only once the caller
//! accepts the licence, and records the class in `model.license_class`.
//! The acceptance is the zoo's own: the `acceptances.json` the `turbo-zoo`
//! command and the server's catalog keep (docs/zoo.md), so a licence
//! accepted with any of them counts for the others.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use turbo::manifest::LicenseClass;
use turbo_zoo::{Acceptances, Home, Licence, LicenceClass, Policy, Requirement, UserId};

use crate::Result;

/// The licence `manifest.model.license` names and the zoo's class for it.
pub fn of(manifest: &Value) -> Result<(Licence, LicenceClass)> {
    let id = manifest["model"]["license"].as_str().ok_or("manifest.model.license: missing")?;
    let licence: Licence = id.parse().map_err(|e| format!("manifest.model.license: {e}"))?;
    Ok((licence, licence.class()))
}

/// The class a manifest records for `manifest.model.license`. A refused
/// licence is an error: no bundle is made under it.
pub fn manifest_class(manifest: &Value) -> Result<LicenseClass> {
    let (licence, class) = of(manifest)?;
    to_manifest(licence, class)
}

/// Check the class a sealed manifest records against its licence. A
/// bundle sealed before the tool recorded classes has none, and passes.
pub fn check_recorded(m: &turbo::manifest::Manifest) -> Result<()> {
    let licence: Licence = m.model.license.parse().map_err(|e| format!("manifest.model.license: {e}"))?;
    let want = to_manifest(licence, licence.class())?;
    match m.model.license_class {
        Some(c) if c == want => Ok(()),
        Some(c) => Err(format!("manifest.model.license_class: {c:?}, and {licence} is {want:?}")),
        None => Ok(()),
    }
}

fn to_manifest(licence: Licence, class: LicenceClass) -> Result<LicenseClass> {
    Ok(match class {
        LicenceClass::Permissive => LicenseClass::Permissive,
        LicenceClass::Lgpl => LicenseClass::Lgpl,
        LicenceClass::ShareAlike => LicenseClass::ShareAlike,
        LicenceClass::Gpl => LicenseClass::Gpl,
        LicenceClass::LocalOnly => LicenseClass::LocalOnly,
        LicenceClass::Refused => {
            let why = licence.refusal().map_or_else(String::new, |r| format!(": {r}"));
            return Err(format!("manifest.model.license: {licence} is refused{why}"));
        }
    })
}

/// How the caller accepts a licence: `--accept-terms` on this run, and
/// the zoo's acceptance file for a user, which holds what was accepted
/// before, with this tool, `turbo-zoo accept` or the demo's page.
#[derive(Debug, Clone, Default)]
pub struct Consent {
    /// `--accept-terms`: the caller read the terms and accepts them now.
    pub flag: bool,
    /// The acceptance file and its user, when the environment names both.
    pub store: Option<(PathBuf, UserId)>,
}

impl Consent {
    /// The flag alone, with no acceptance file.
    pub fn flag(flag: bool) -> Consent {
        Consent { flag, store: None }
    }

    /// The flag and, as the `turbo-zoo` command finds them, the zoo's
    /// acceptance file (`$TURBO_ZOO_HOME`, docs/zoo.md) and the user
    /// (`$USER`). Without a home or a user name there is no file.
    pub fn from_env(flag: bool) -> Consent {
        let store = Home::from_env().zip(UserId::from_env().ok()).map(|(h, u)| (h.acceptances(), u));
        Consent { flag, store }
    }
}

fn store_error(path: &Path, e: impl std::fmt::Display) -> String {
    format!("{}: {e}", path.display())
}

/// Check or record `r` in the zoo's acceptance file when `consent.store`
/// is present. Returns `Ok(true)` when `r` was already accepted or was
/// just recorded under `consent.flag`.
fn accept_or_check(r: &Requirement, consent: &Consent) -> Result<bool> {
    let Some((path, user)) = &consent.store else {
        return Ok(false);
    };
    let mut acceptances = Acceptances::load(path).map_err(|e| e.to_string())?;
    if acceptances.has(user, r) {
        println!("{user} accepted {r} ({}) before", path.display());
        return Ok(true);
    }
    if !consent.flag {
        return Ok(false);
    }
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    acceptances.accept_terms(user, r, now).map_err(|e| e.to_string())?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| store_error(dir, e))?;
    }
    acceptances.save(path).map_err(|e| e.to_string())?;
    println!("{user} accepts {r}, recorded in {}", path.display());
    Ok(true)
}

/// Whether the model may be fetched: never under a refused licence, and
/// under a class that needs acceptance (CC BY-NC, CC BY-SA, LGPL, GPL)
/// only when `accept_terms` says the caller accepted the licence.
pub fn check_fetch(manifest: &Value, accept_terms: bool) -> Result<LicenseClass> {
    check_fetch_with(manifest, &Consent::flag(accept_terms))
}

/// [`check_fetch`] with the zoo's acceptance file: a licence the user
/// accepted before passes without the flag, and the flag records the
/// acceptance there, as `turbo-zoo accept --yes` does, so the server and
/// the zoo see it too.
pub fn check_fetch_with(manifest: &Value, consent: &Consent) -> Result<LicenseClass> {
    let class = manifest_class(manifest)?;
    let (licence, zoo_class) = of(manifest)?;
    if !zoo_class.needs_acceptance(Policy::default()) {
        return Ok(class);
    }
    let requirement = Requirement::of(licence);
    if let Some(r) = &requirement
        && accept_or_check(r, consent)?
    {
        return Ok(class);
    }
    if consent.flag {
        return Ok(class);
    }
    let text = requirement.map_or_else(|| "its terms".to_owned(), |r| r.text);
    Err(format!(
        "manifest.model.license: {licence} is {zoo_class}, which must be accepted before a fetch: \
         read {text}, then fetch with --accept-terms (or accept it once with turbo-zoo accept)"
    ))
}

/// The zoo [`Requirement`] for a recipe's `notice_terms` address, under
/// [`Licence::RestrictedPublic`]. `notice_terms` without `notice`, or an
/// address the zoo's URL rule rejects, is an error.
pub fn notice_requirement(recipe: &crate::recipe::Recipe) -> Result<Option<Requirement>> {
    let Some(raw) = &recipe.notice_terms else {
        return Ok(None);
    };
    if recipe.notice.is_none() {
        return Err("notice_terms requires notice".into());
    }
    let url: turbo_zoo::Url = raw.parse().map_err(|e| format!("notice_terms: {e}"))?;
    Ok(Some(Requirement { licence: Licence::RestrictedPublic, text: url.as_str().to_owned() }))
}

/// Check a recipe's `notice` before a fetch. When `notice_terms` gives
/// the notice's address, it is checked against and recorded in the
/// zoo's acceptance file as a [`Licence::RestrictedPublic`]
/// [`Requirement`], so an acceptance made on the page, with
/// `turbo-zoo accept` or with `--accept-terms` satisfies later fetches
/// without the flag. A notice with no `notice_terms` requires the flag
/// on each fetch.
pub fn check_notice_with(recipe: &crate::recipe::Recipe, consent: &Consent) -> Result<()> {
    let requirement = notice_requirement(recipe)?;
    let Some(n) = &recipe.notice else {
        return Ok(());
    };
    println!("notice: {n}");
    if let Some(r) = &requirement {
        if accept_or_check(r, consent)? || consent.flag {
            return Ok(());
        }
        return Err(format!(
            "this model carries terms beyond its licence (the notice above): \
             read {}, then fetch with --accept-terms (or accept it once with turbo-zoo accept)",
            r.text
        ));
    }
    if consent.flag {
        return Ok(());
    }
    Err("this model carries terms beyond its licence (the notice above): \
         fetch it with --accept-terms once you have read them"
        .into())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn with(license: &str) -> Value {
        json!({ "model": { "license": license } })
    }

    #[test]
    fn classes_follow_the_zoo() {
        assert_eq!(manifest_class(&with("MIT")).unwrap(), LicenseClass::Permissive);
        assert_eq!(manifest_class(&with("CC-BY-SA-4.0")).unwrap(), LicenseClass::ShareAlike);
        assert_eq!(manifest_class(&with("CC-BY-NC-4.0")).unwrap(), LicenseClass::LocalOnly);
        assert_eq!(manifest_class(&with("GPL-3.0-only")).unwrap(), LicenseClass::Gpl);
        let e = manifest_class(&with("CC-BY-ND-4.0")).unwrap_err();
        assert!(e.contains("refused"), "{e}");
        let e = manifest_class(&with("WTFPL")).unwrap_err();
        assert!(e.contains("not a licence the zoo knows"), "{e}");
    }

    #[test]
    fn a_non_commercial_model_needs_acceptance() {
        let m = with("CC-BY-NC-4.0");
        let e = check_fetch(&m, false).unwrap_err();
        assert!(e.contains("--accept-terms") && e.contains("creativecommons.org/licenses/by-nc/4.0"), "{e}");
        assert_eq!(check_fetch(&m, true).unwrap(), LicenseClass::LocalOnly);
        assert_eq!(check_fetch(&with("Apache-2.0"), false).unwrap(), LicenseClass::Permissive);
        assert!(check_fetch(&with("LicenseRef-Proprietary"), true).is_err());
    }

    fn store(name: &str) -> (PathBuf, UserId) {
        let dir = std::env::temp_dir().join(format!("turbo-bundle-consent-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (dir.join("home").join("acceptances.json"), "alice".parse().unwrap())
    }

    /// The flag records the acceptance in the zoo's file; a later fetch
    /// passes without it, and another user's acceptance does not count.
    #[test]
    fn the_flag_records_in_the_zoo_file_and_a_later_fetch_needs_none() {
        let (path, alice) = store("record");
        let m = with("CC-BY-SA-4.0");
        let alone = Consent { flag: false, store: Some((path.clone(), alice.clone())) };
        assert!(check_fetch_with(&m, &alone).unwrap_err().contains("turbo-zoo accept"));
        assert!(!path.exists());
        let flagged = Consent { flag: true, ..alone.clone() };
        assert_eq!(check_fetch_with(&m, &flagged).unwrap(), LicenseClass::ShareAlike);
        let file = Acceptances::load(&path).unwrap();
        assert!(file.has(&alice, &Requirement::of(Licence::CcBySa40).unwrap()));
        assert_eq!(check_fetch_with(&m, &alone).unwrap(), LicenseClass::ShareAlike);
        let bob = Consent { flag: false, store: Some((path.clone(), "bob".parse().unwrap())) };
        assert!(check_fetch_with(&m, &bob).is_err());
        // Another licence is its own acceptance.
        assert!(check_fetch_with(&with("CC-BY-NC-4.0"), &alone).is_err());
        // A permissive licence never touches the file.
        let before = std::fs::read(&path).unwrap();
        assert_eq!(check_fetch_with(&with("MIT"), &flagged).unwrap(), LicenseClass::Permissive);
        assert_eq!(std::fs::read(&path).unwrap(), before);
        std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap()).unwrap();
    }

    /// An acceptance the zoo command made counts here.
    #[test]
    fn an_acceptance_made_elsewhere_counts() {
        let (path, alice) = store("elsewhere");
        let mut file = Acceptances::default();
        file.accept(&alice, Licence::CcByNc40, 1_792_000_000).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        file.save(&path).unwrap();
        let consent = Consent { flag: false, store: Some((path.clone(), alice)) };
        assert_eq!(check_fetch_with(&with("CC-BY-NC-4.0"), &consent).unwrap(), LicenseClass::LocalOnly);
        // A file that does not parse is an error, not a pass or a rewrite.
        std::fs::write(&path, b"{}").unwrap();
        assert!(check_fetch_with(&with("CC-BY-NC-4.0"), &Consent { flag: true, ..consent }).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"{}");
        std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap()).unwrap();
    }

    #[test]
    fn notice_terms_validate_and_share_the_zoo_acceptance_file() {
        let mut r = crate::recipe::Recipe::load(Path::new("cross-encoder/ms-marco-MiniLM-L6-v2")).unwrap();
        let req = notice_requirement(&r).unwrap().unwrap();
        assert_eq!(
            req,
            Requirement { licence: Licence::RestrictedPublic, text: "https://microsoft.github.io/msmarco/".into() }
        );
        let (path, alice) = store("notice");
        let alone = Consent { flag: false, store: Some((path.clone(), alice.clone())) };
        let e = check_notice_with(&r, &alone).unwrap_err();
        assert!(e.contains("https://microsoft.github.io/msmarco/") && e.contains("--accept-terms"), "{e}");
        let flagged = Consent { flag: true, ..alone.clone() };
        check_notice_with(&r, &flagged).unwrap();
        assert!(Acceptances::load(&path).unwrap().has(&alice, &req));
        check_notice_with(&r, &alone).unwrap();
        r.notice_terms = Some("http://microsoft.github.io/msmarco/".into());
        assert!(notice_requirement(&r).unwrap_err().contains("notice_terms:"));
        r.notice = None;
        assert_eq!(notice_requirement(&r).unwrap_err(), "notice_terms requires notice");
        std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap()).unwrap();
    }
}
