//! The library reading models from a turbo-zoo container
//! (scripts/test-zoo-image.sh, #1693): `TURBO_ZOO_SOURCE` names the
//! running zoo (`grpc://HOST:PORT`) and TURBO_ZOO_SMOKE_INDEX the index it
//! serves. A local zoo with its own cache and acceptances loads a
//! permissive model through it, holds a share-alike one back until its
//! terms are accepted both here and on the zoo, and then serves both from
//! its cache with no source at all. Without the variables there is no
//! container to read and it passes.

use std::path::PathBuf;

use turbo_kserve::proto::training::RecordLicenceAcceptanceRequest;
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_zoo::{Acceptances, Artifact, Index, LicenceClass, Offline, Policy, Requirement, UserId, Zoo, ZooError};
use turbo_zoo_remote::{ENV_SOURCE, sources};

struct Dir(PathBuf);

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Loads `a` into `cache` through the sources `spec` names, as `user`,
/// and checks every file it returns against the index.
fn load(
    spec: &str,
    index: &Index,
    cache: &Dir,
    a: &Artifact,
    user: &UserId,
    acc: &Acceptances,
) -> Result<(), ZooError> {
    let source = sources(spec, user, Offline).unwrap();
    let zoo = Zoo::new(index.clone(), cache.0.clone(), Policy::default(), &source);
    let got = zoo.fetch(a.id.as_str(), user, acc)?;
    assert_eq!(got.files.len(), a.files.len(), "{}", a.id);
    for (path, f) in got.files.iter().zip(&a.files) {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len() as u64, f.size, "{}/{}", a.id, f.path.as_str());
        assert_eq!(turbo_zoo::Sha256::of(&bytes), f.sha256, "{}/{}", a.id, f.path.as_str());
        assert_eq!(turbo_zoo::Sha512::of(&bytes), f.sha512, "{}/{}", a.id, f.path.as_str());
    }
    Ok(())
}

/// Records `user`'s acceptance of `a`'s licence on the zoo itself.
fn accept_on_zoo(spec: &str, user: &UserId, a: &Artifact) {
    let addr = spec.replace("grpc://", "http://");
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let mut c = TurboCatalogServiceClient::connect(addr).await.unwrap();
        let req = RecordLicenceAcceptanceRequest {
            user: user.to_string(),
            licence: a.license.to_string(),
            artifact_id: a.id.to_string(),
            ..Default::default()
        };
        c.record_licence_acceptance(req).await.unwrap();
    });
}

#[test]
fn the_library_loads_models_from_a_running_zoo_image() {
    let (Ok(spec), Ok(index)) = (std::env::var(ENV_SOURCE), std::env::var("TURBO_ZOO_SMOKE_INDEX")) else {
        eprintln!("{ENV_SOURCE} and TURBO_ZOO_SMOKE_INDEX are not set; no zoo image to read");
        return;
    };
    assert!(spec.starts_with("grpc://"), "{ENV_SOURCE}={spec}: the smoke run names the zoo with grpc://");
    let index = Index::parse(&std::fs::read(&index).unwrap()).unwrap();
    let cache = Dir(std::env::temp_dir().join(format!("turbo-zoo-image-library-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&cache.0);
    // A user of its own, so the server smoke test's acceptances do not count.
    let user: UserId = "library".parse().unwrap();
    let mut acc = Acceptances::default();
    for a in index.list() {
        if a.class == LicenceClass::Permissive {
            load(&spec, &index, &cache, a, &user, &acc).unwrap();
            continue;
        }
        // Not accepted here: refused before the zoo is asked.
        let e = load(&spec, &index, &cache, a, &user, &acc).unwrap_err();
        assert!(matches!(e, ZooError::NotAccepted { .. }), "{e:?}");
        // Accepted here but not on the zoo: the zoo sends nothing.
        acc.accept_terms(&user, &Requirement::of(a.license).unwrap(), 1).unwrap();
        let e = load(&spec, &index, &cache, a, &user, &acc).unwrap_err();
        assert!(!matches!(e, ZooError::NotAccepted { .. }), "{e:?}");
        assert!(e.to_string().contains(&a.license.to_string()), "{e}");
        accept_on_zoo(&spec, &user, a);
        load(&spec, &index, &cache, a, &user, &acc).unwrap();
    }
    // The local cache now serves every model with no source at all.
    let zoo = Zoo::new(index.clone(), cache.0.clone(), Policy::default(), &Offline);
    for a in index.list() {
        zoo.fetch(a.id.as_str(), &user, &acc).unwrap();
    }
}
