//! Zoo mode's read-through: a read of a permitted artifact that is not in
//! the cache fetches the whole artifact first, with the licence and free
//! disk checks and the verified fetch InstallModel makes. One fetch of an
//! artifact runs at a time: a reader that arrives while one runs waits for
//! it, then finds the files cached and fetches nothing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tonic::Status;
use turbo_zoo::{Policy, UserId, Zoo};

use super::{Catalog, install, zoo_status};

/// One gate per artifact id, made on the first read-through of that id.
/// The index bounds how many there are.
pub(crate) type Fetching = Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>;

/// Fetches the artifact `id` into the cache for `user`, holding its gate.
pub(super) fn fetch(catalog: &Catalog, user: &UserId, id: &str, gates: &Fetching) -> Result<(), Status> {
    let gate = gates.lock().unwrap_or_else(|p| p.into_inner()).entry(id.to_owned()).or_default().clone();
    let _held = gate.lock().unwrap_or_else(|p| p.into_inner());
    install::admit(catalog, user, id)?;
    let zoo = Zoo::new((*catalog.index).clone(), catalog.cache(), Policy::default(), &*catalog.transport);
    zoo.fetch_with(id, user, || catalog.acceptances()).map_err(zoo_status)?;
    Ok(())
}
