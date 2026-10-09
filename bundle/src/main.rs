//! turbo-bundle: make and check model bundles.

use std::path::Path;
use std::process::ExitCode;

use turbo_bundle::recipe::Recipe;
use turbo_bundle::{Result, catalogue, distill, fetch, seal};

const USAGE: &str = "\
usage:
  turbo-bundle make [--accept-terms] <recipe.json | model-id> <upstream-dir> <bundle-dir>
      fetch, stage, reference, convert, seal and verify, in that order
  turbo-bundle fetch [--accept-terms] <recipe.json | model-id> <upstream-dir>
      fetch the upstream files at the recipe's commit. A model with terms
      beyond its licence is fetched only with --accept-terms
  turbo-bundle catalogue
      the models a model-id names, with their licences (docs/static.md)
  turbo-bundle reference <recipe.json> <upstream-dir> <bundle-dir>
      copy the files the bundle carries, write the reference (in the
      reference container, or for a static model by the tool itself)
      and run the conversions, then seal and verify
  turbo-bundle seal <recipe.json> <upstream-dir> <bundle-dir>
      stage, then seal from files already in the bundle. Does not run
      docker. The reference file and its report must already be there,
      and so must an OpenVINO IR the recipe converts (docs/npu.md).
      A conversion whose files are absent is left out of the manifest
  turbo-bundle verify <bundle-dir>
      load a bundle the way a machine does and check every file
  turbo-bundle distill <recipe.json> <base-bundle-dir> <bundle-dir>
      distil a static model from the base bundle, write its reference,
      then seal and verify (docs/static.md)";

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
    let accept = args.contains(&"--accept-terms");
    let args: Vec<&str> = args.iter().copied().filter(|a| *a != "--accept-terms").collect();
    match args.as_slice() {
        ["make", recipe, upstream, bundle] => {
            let r = Recipe::load(Path::new(recipe))?;
            fetch::fetch(&r, Path::new(upstream), accept)?;
            turbo_bundle::make(&r, Path::new(upstream), Path::new(bundle))
        }
        ["fetch", recipe, upstream] => fetch::fetch(&Recipe::load(Path::new(recipe))?, Path::new(upstream), accept),
        ["reference", recipe, upstream, bundle] => {
            turbo_bundle::make(&Recipe::load(Path::new(recipe))?, Path::new(upstream), Path::new(bundle))
        }
        ["seal", recipe, upstream, bundle] => {
            let mut r = Recipe::load(Path::new(recipe))?;
            seal::seal_staged(&mut r, Path::new(upstream), Path::new(bundle))
        }
        ["verify", bundle] => seal::verify(Path::new(bundle)),
        ["catalogue"] => {
            for (id, _) in catalogue::MODELS {
                let r = Recipe::load(Path::new(id))?;
                let license = r.str_at("/model/license")?;
                let (_, commit) = r.source()?;
                println!("{id}  {license}  {commit}");
                if let Some(n) = &r.notice {
                    println!("    {n}");
                }
            }
            Ok(())
        }
        ["distill", recipe, base, bundle] => {
            let mut r = Recipe::load(Path::new(recipe))?;
            distill::stage(&r, Path::new(base), Path::new(bundle))?;
            distill::seal(&mut r, Path::new(bundle))
        }
        _ => Err(USAGE.into()),
    }
}
