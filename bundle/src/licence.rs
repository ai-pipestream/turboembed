//! The licence class of the model a recipe names, by the model zoo's own
//! rules (analysis/zoo/src/licence): the tool refuses a licence the zoo
//! refuses, fetches a class that needs acceptance only once the caller
//! accepts the licence, and records the class in `model.license_class`.

use serde_json::Value;
use turbo::manifest::LicenseClass;
use turbo_zoo::{Licence, LicenceClass, Policy, Requirement};

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

/// Whether the model may be fetched: never under a refused licence, and
/// under a class that needs acceptance (CC BY-NC, CC BY-SA, LGPL, GPL)
/// only when `accept_terms` says the caller accepted the licence.
pub fn check_fetch(manifest: &Value, accept_terms: bool) -> Result<LicenseClass> {
    let class = manifest_class(manifest)?;
    let (licence, zoo_class) = of(manifest)?;
    if zoo_class.needs_acceptance(Policy::default()) && !accept_terms {
        let text = Requirement::of(licence).map_or_else(|| "its terms".to_owned(), |r| r.text);
        return Err(format!(
            "manifest.model.license: {licence} is {zoo_class}, which must be accepted before a fetch: \
             read {text}, then fetch with --accept-terms"
        ));
    }
    Ok(class)
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
}
