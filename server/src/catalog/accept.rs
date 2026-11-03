//! Licence acceptances: record, revoke and list, on the same
//! acceptances.json the turbo-zoo command keeps (docs/zoo.md).

use std::time::{SystemTime, UNIX_EPOCH};

use tonic::Status;
use turbo_zoo::{Acceptance, Acceptances, Policy, Requirement, Sha256, Shown, UserId, ZooError};

use super::{Catalog, invalid, terms, zoo_status};
use crate::proto::training::{
    LicenceAcceptance, ListLicenceAcceptancesResponse, RecordLicenceAcceptanceResponse, RevokeLicenceAcceptanceResponse,
};

/// What a page says it showed: an artifact the index lists that needs
/// `r`, and the SHA-256 of `r`'s text. Field numbers are the request's.
pub(super) fn shown(catalog: &Catalog, r: &Requirement, artifact: &str, sha256: &str) -> Result<Shown, Status> {
    let artifact = match artifact {
        "" => None,
        id => {
            let a = catalog.index.get(id).map_err(|e| invalid(4, e.to_string()))?;
            if !a.licences_to_accept(Policy::default()).contains(r) {
                return Err(invalid(4, format!("{id} does not need {r}")));
            }
            Some(a.id.clone())
        }
    };
    let text_sha256 = match sha256 {
        "" => None,
        h => {
            let h: Sha256 = h.parse().map_err(|e: String| invalid(5, format!("text_sha256: {e}")))?;
            if h != Sha256::of(r.text.as_bytes()) {
                return Err(invalid(5, format!("text_sha256 {h} is not the SHA-256 of the served text {:?}", r.text)));
            }
            Some(h)
        }
    };
    Ok(Shown { artifact, text_sha256 })
}

fn save(catalog: &Catalog, acceptances: &Acceptances) -> Result<(), Status> {
    std::fs::create_dir_all(&catalog.home)
        .map_err(|e| zoo_status(ZooError::Io { path: catalog.home.clone(), message: e.to_string() }))?;
    acceptances.save(&catalog.acceptances_path()).map_err(zoo_status)
}

/// Record `who`'s acceptance of `r`. Accepting again keeps the first.
pub(super) fn record(
    catalog: &Catalog,
    who: &UserId,
    r: &Requirement,
    shown: &Shown,
) -> Result<RecordLicenceAcceptanceResponse, Status> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let mut acceptances = catalog.acceptances().map_err(zoo_status)?;
    acceptances.accept_shown(who, r, shown, now).map_err(zoo_status)?;
    save(catalog, &acceptances)?;
    Ok(RecordLicenceAcceptanceResponse {
        user: who.to_string(),
        accepted: Some(terms(r)),
        accepted_at: acceptances.find(who, r).map_or(now, |a| a.accepted_at),
    })
}

/// Withdraw `who`'s acceptance of `r`; the file is written only when
/// there was one.
pub(super) fn revoke(
    catalog: &Catalog,
    who: &UserId,
    r: &Requirement,
) -> Result<RevokeLicenceAcceptanceResponse, Status> {
    let mut acceptances = catalog.acceptances().map_err(zoo_status)?;
    let was_accepted = acceptances.revoke(who, r);
    if was_accepted {
        save(catalog, &acceptances)?;
    }
    Ok(RevokeLicenceAcceptanceResponse { user: who.to_string(), revoked: Some(terms(r)), was_accepted })
}

/// `who`'s acceptances in the order made; nothing without a user.
pub(super) fn list(
    catalog: &Catalog,
    who: Option<&UserId>,
    admin: bool,
) -> Result<ListLicenceAcceptancesResponse, Status> {
    let Some(who) = who else {
        return Ok(ListLicenceAcceptancesResponse { user: String::new(), accepted: Vec::new(), admin });
    };
    let acceptances = catalog.acceptances().map_err(zoo_status)?;
    let accepted = acceptances.of_user(who).map(one).collect();
    Ok(ListLicenceAcceptancesResponse { user: who.to_string(), accepted, admin })
}

fn one(a: &Acceptance) -> LicenceAcceptance {
    let r = Requirement { licence: a.license, text: a.license_text.clone() };
    LicenceAcceptance {
        terms: Some(terms(&r)),
        accepted_at: a.accepted_at,
        artifact_id: a.artifact.as_ref().map(|x| x.to_string()).unwrap_or_default(),
        text_sha256: a.license_text_sha256.map(|h| h.to_string()).unwrap_or_default(),
    }
}
