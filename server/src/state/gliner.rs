use std::path::Path;
use std::sync::Arc;

use turbo::deberta::package::PackageManifest;
use turbo_gliner::package::Manifest;
use turbo_relation::Gliner2Relations;

use super::{LoadError, Served, State};
use crate::api::{Failure, TURBO_E_BUNDLE_INVALID};

fn failure(served: &Served, call: &'static str, message: String) -> LoadError {
    LoadError {
        model: served.name.clone(),
        call,
        failure: Failure::new(TURBO_E_BUNDLE_INVALID, 0, message),
        fields: &[],
    }
}

pub(super) fn is_loaded(served: &Served) -> bool {
    served.loaded.get().is_some() || served.gliner.get().is_some() || served.relations.get().is_some()
}

pub(super) fn load(state: &State, served: &Arc<Served>) -> Result<bool, LoadError> {
    let path = Path::new(&served.config.bundle);
    if path.join("gliner.json").is_file() {
        let model = Manifest::load(path).map_err(|message| failure(served, "gliner_load", message))?;
        let _ = served.gliner.set(Arc::new(model));
        state.notify_health();
        return Ok(true);
    }
    if path.join("added_tokens.json").is_file() && path.join("encoder/deberta.json").is_file() {
        let (identity, encoder) = PackageManifest::load(&path.join("encoder"))
            .map_err(|error| failure(served, "gliner2_encoder_load", error.message))?;
        let model =
            Gliner2Relations::load_with(encoder.into(), path, &identity.source_repository, &identity.source_revision)
                .map_err(|message| failure(served, "gliner2_load", message))?;
        let _ = served.relations.set(Arc::new(model));
        state.notify_health();
        return Ok(true);
    }
    Ok(false)
}
