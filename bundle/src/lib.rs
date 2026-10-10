//! The bundle tool: makes a bundle directory from a recipe, as
//! docs/bundle.md describes it, and checks one with the core's own loader.
//!
//! A recipe is a manifest without `files`, plus the upstream files to fetch
//! at the manifest's `model.source.commit`. The tool fetches them, copies
//! the ones the bundle carries, runs the reference pipeline in its pinned
//! container and makes the converted artifacts there, fills in `files`
//! and each `produced_by` from what actually ran, writes the manifest, and
//! loads the result through the core. Nothing it writes is typed by hand.

pub mod api;
pub mod catalogue;
pub mod convert;
pub mod distill;
pub mod fetch;
pub mod recipe;
pub mod reference;
pub mod seal;
pub mod static_reference;

use std::path::Path;

pub type Result<T> = std::result::Result<T, String>;

/// A bundle or upstream path: relative, `/`-separated, no `..`.
pub fn check_rel(path: &str) -> Result<()> {
    if path.is_empty() || path.starts_with('/') || path.split('/').any(|p| p.is_empty() || p == "." || p == "..") {
        return Err(format!("{path:?} is not a relative path inside the directory"));
    }
    Ok(())
}

/// Make a bundle from upstream files already fetched: stage, reference,
/// convert, seal and verify.
pub fn make(r: &recipe::Recipe, upstream: &Path, bundle: &Path) -> Result<()> {
    if bundle.join("manifest.json").exists() {
        return Err(format!("{} already holds a bundle; make it into an empty directory", bundle.display()));
    }
    seal::stage(r, upstream, bundle)?;
    if r.manifest.get("static_embedding").is_some() {
        static_reference::make(r, upstream, bundle)?;
        println!("{}: verified", bundle.display());
        return Ok(());
    }
    let produced_by = reference::run(r, upstream, bundle)?;
    let converted = convert::run(r, upstream, bundle)?;
    seal::seal(r, bundle, produced_by, converted)?;
    println!("{}: verified", bundle.display());
    Ok(())
}
