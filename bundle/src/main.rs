//! turbo-bundle: make and check model bundles.

use std::path::Path;
use std::process::ExitCode;

use turbo_bundle::recipe::Recipe;
use turbo_bundle::{Result, catalogue, distill, fetch, seal};

const USAGE: &str = "\
usage:
  turbo-bundle deberta-convert <config.json> <checkpoint.safetensors> <prefix-or-dash> <repo> <revision> <license> <new-dir>
      create a native DeBERTa-v3 encoder package for the Rust API
  turbo-bundle deberta-verify <directory>
      check encoder configuration, tensor shapes and weight digest
  turbo-bundle glirel-convert <pytorch_model.bin> <base-config.json> <spm.model> <repo> <revision> <license> <new-dir>
      write a GLiREL checkpoint as the native scorer's package: the encoder,
      the heads and glirel.json
  turbo-bundle torch-convert <checkpoint.bin> <new.safetensors>
      write a torch.save checkpoint's F32 tensors as safetensors, reading
      its pickle as data without running it
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
        ["deberta-convert", config, weights, prefix, repo, revision, license, out] => {
            turbo_bundle::deberta::convert_files(
                Path::new(config),
                Path::new(weights),
                prefix,
                (repo, revision, license),
                Path::new(out),
            )
        }
        ["glirel-convert", checkpoint, config, spm, repo, revision, license, out] => {
            let read = |p: &str| std::fs::read(p).map_err(|e| format!("{p}: {e}"));
            turbo_bundle::glirel::write_package(
                &read(checkpoint)?,
                &read(config)?,
                &read(spm)?,
                (repo, revision, license),
                Path::new(out),
            )
        }
        ["torch-convert", checkpoint, out] => {
            let bytes = std::fs::read(checkpoint).map_err(|e| format!("{checkpoint}: {e}"))?;
            let tensors = turbo_bundle::torch::read(&bytes)?;
            let out_bytes = turbo_bundle::torch::to_safetensors(&tensors, |k| Some(k.to_owned()));
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out)
                .and_then(|mut f| std::io::Write::write_all(&mut f, &out_bytes))
                .map_err(|e| format!("{out}: {e}"))
        }
        ["deberta-verify", dir] => {
            turbo::deberta::package::PackageManifest::load(Path::new(dir)).map(|_| ()).map_err(|e| e.message)
        }
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
