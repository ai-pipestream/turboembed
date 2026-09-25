//! turbo-bundle: make and check model bundles.

use std::path::Path;
use std::process::ExitCode;

use turbo_bundle::recipe::Recipe;
use turbo_bundle::{Result, convert, fetch, reference, seal};

const USAGE: &str = "\
usage:
  turbo-bundle make <recipe.json> <upstream-dir> <bundle-dir>
      fetch, stage, reference, convert, seal and verify, in that order
  turbo-bundle fetch <recipe.json> <upstream-dir>
      fetch the upstream files at the recipe's commit
  turbo-bundle reference <recipe.json> <upstream-dir> <bundle-dir>
      copy the files the bundle carries, run the reference container
      and the conversions, then seal and verify
  turbo-bundle verify <bundle-dir>
      load a bundle the way a machine does and check every file";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("turbo-bundle: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[&str]) -> Result<()> {
    match args {
        ["make", recipe, upstream, bundle] => {
            let r = Recipe::load(Path::new(recipe))?;
            fetch::fetch(&r, Path::new(upstream))?;
            make_from_upstream(&r, Path::new(upstream), Path::new(bundle))
        }
        ["fetch", recipe, upstream] => fetch::fetch(&Recipe::load(Path::new(recipe))?, Path::new(upstream)),
        ["reference", recipe, upstream, bundle] => {
            make_from_upstream(&Recipe::load(Path::new(recipe))?, Path::new(upstream), Path::new(bundle))
        }
        ["verify", bundle] => seal::verify(Path::new(bundle)),
        _ => Err(USAGE.into()),
    }
}

fn make_from_upstream(r: &Recipe, upstream: &Path, bundle: &Path) -> Result<()> {
    if bundle.join("manifest.json").exists() {
        return Err(format!("{} already holds a bundle; make it into an empty directory", bundle.display()));
    }
    seal::stage(r, upstream, bundle)?;
    let produced_by = reference::run(r, upstream, bundle)?;
    let converted = convert::run(r, bundle)?;
    seal::seal(r, bundle, produced_by, converted)?;
    println!("{}: verified", bundle.display());
    Ok(())
}
