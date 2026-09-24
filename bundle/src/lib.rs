//! The bundle tool: makes a bundle directory from a recipe, as
//! docs/bundle.md describes it, and checks one with the core's own loader.
//!
//! A recipe is a manifest without `files`, plus the upstream files to fetch
//! at the manifest's `model.source.commit`. The tool fetches them, copies
//! the ones the bundle carries, runs the reference pipeline in its pinned
//! container, fills in `files` and the reference's `produced_by` from what
//! actually ran, writes the manifest, and loads the result through the
//! core. Nothing it writes is typed by hand.

pub mod fetch;
pub mod recipe;
pub mod reference;
pub mod seal;

pub type Result<T> = std::result::Result<T, String>;

/// A bundle or upstream path: relative, `/`-separated, no `..`.
pub fn check_rel(path: &str) -> Result<()> {
    if path.is_empty() || path.starts_with('/') || path.split('/').any(|p| p.is_empty() || p == "." || p == "..") {
        return Err(format!("{path:?} is not a relative path inside the directory"));
    }
    Ok(())
}
