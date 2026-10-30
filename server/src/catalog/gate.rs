//! LoadBundle's licence check: a bundle whose licence needs acceptance
//! (docs/zoo.md, licence classes) loads only for a user who accepted it
//! in the catalog's acceptance file, the one the turbo-zoo command, the
//! turbo-bundle tool and RecordLicenceAcceptance keep.

use std::path::Path;

use turbo::manifest::Manifest;
use turbo_zoo::{Licence, Policy, Requirement, UserId};

use super::Catalog;
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE};

/// Refuse loading the bundle at `bundle` for `user` when its manifest
/// names a licence the zoo fetches only after acceptance and `user` has
/// not accepted it. A bundle whose manifest cannot be read, or whose
/// licence the zoo does not know, is left to the loader, which refuses
/// or loads it as before. `field` is the request's user field.
pub(crate) fn check_load(catalog: &Catalog, bundle: &str, user: &str, field: u32) -> Result<(), Failure> {
    let Ok(bytes) = std::fs::read(Path::new(bundle).join("manifest.json")) else { return Ok(()) };
    let Ok(manifest) = Manifest::parse(&bytes) else { return Ok(()) };
    let Ok(licence) = manifest.model.license.parse::<Licence>() else { return Ok(()) };
    let class = licence.class();
    if !class.needs_acceptance(Policy::default()) {
        return Ok(());
    }
    let terms = Requirement::of(licence).map_or_else(|| "its host's terms".to_owned(), |r| r.text);
    if user.is_empty() {
        return Err(Failure::new(
            TURBO_E_INVALID_STATE,
            field,
            format!("{bundle}: {licence} is {class}; name the user who accepted {terms}"),
        ));
    }
    let who: UserId = user.parse().map_err(|e: String| Failure::new(TURBO_E_INVALID_ARGUMENT, field, e))?;
    let acceptances = catalog.acceptances().map_err(|e| Failure::new(TURBO_E_INVALID_STATE, 0, e.to_string()))?;
    let accepted = match Requirement::of(licence) {
        Some(r) => acceptances.has(&who, &r),
        // Restricted terms are each host's own and the manifest names no
        // host, so any of the user's acceptances under the licence counts.
        None => acceptances.of_user(&who).any(|a| a.license == licence),
    };
    if accepted {
        return Ok(());
    }
    Err(Failure::new(
        TURBO_E_INVALID_STATE,
        0,
        format!("{bundle}: {licence} is {class}, and {who} has not accepted it: read {terms}, then accept it"),
    ))
}
