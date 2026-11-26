use std::path::Path;
use std::sync::Arc;

use turbo::deberta::package::PackageManifest;
use turbo_gliner::package::Manifest;
use turbo_relation::Gliner2Relations;

use super::{LoadError, Served, State};
use crate::api::{Failure, Runtime, TURBO_E_BUNDLE_INVALID, TURBO_E_DEVICE_NOT_FOUND, TURBO_E_UNSUPPORTED_OPTION};
use crate::config::Device;

fn failure(served: &Served, call: &'static str, message: String) -> LoadError {
    LoadError {
        model: served.name.clone(),
        call,
        failure: Failure::new(TURBO_E_BUNDLE_INVALID, 0, message),
        fields: &[],
    }
}

/// These package loaders run the CPU encoder, so anything the config
/// asks for that would not run there is refused before the package is
/// read: a device index the runtime does not list, one that is not the
/// CPU, or a precision other than the model's. `select` leaves the
/// choice to the server, and the CPU encoder is the choice it has.
fn require_cpu(state: &State, served: &Served, call: &'static str) -> Result<(), LoadError> {
    let c = &served.config;
    if c.precision != turbo::TURBO_PRECISION_MODEL {
        return Err(LoadError {
            model: served.name.clone(),
            call,
            failure: Failure::new(
                TURBO_E_UNSUPPORTED_OPTION,
                0,
                format!(
                    "precision {}: GLiNER packages run at PRECISION_MODEL on the cpu",
                    crate::api::names::precision(c.precision)
                ),
            ),
            fields: &[],
        });
    }
    let Device::Index(index) = c.device else { return Ok(()) };
    let rt = match state.runtime.get() {
        Some(rt) => rt,
        None => {
            let rt = Runtime::create().map_err(|failure| LoadError {
                model: served.name.clone(),
                call: "turbo_runtime_create",
                failure,
                fields: &[],
            })?;
            state.runtime.get_or_init(|| rt)
        }
    };
    let info = rt.device_info(index).map_err(|failure| LoadError {
        model: served.name.clone(),
        call: "turbo_runtime_device_info",
        failure,
        fields: &[],
    })?;
    if info.kind == turbo::TURBO_DEVICE_CPU {
        return Ok(());
    }
    Err(LoadError {
        model: served.name.clone(),
        call,
        failure: Failure::new(
            TURBO_E_DEVICE_NOT_FOUND,
            0,
            format!(
                "device {index} ({} {}): GLiNER packages run only on cpu",
                crate::api::field(&info.backend),
                crate::api::field(&info.name)
            ),
        ),
        fields: &[],
    })
}

pub(crate) fn is_loaded(served: &Served) -> bool {
    served.loaded.get().is_some() || served.gliner.get().is_some() || served.relations.get().is_some()
}

pub(super) fn load(state: &State, served: &Arc<Served>) -> Result<bool, LoadError> {
    let path = Path::new(&served.config.bundle);
    if path.join("gliner.json").is_file() {
        require_cpu(state, served, "gliner_load")?;
        let model = Manifest::load(path).map_err(|message| failure(served, "gliner_load", message))?;
        let _ = served.gliner.set(Arc::new(model));
        state.notify_health();
        return Ok(true);
    }
    if path.join("added_tokens.json").is_file() && path.join("encoder/deberta.json").is_file() {
        require_cpu(state, served, "gliner2_load")?;
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
