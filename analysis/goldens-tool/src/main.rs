//! `turbo-goldens`: makes and checks the golden layer files in
//! `analysis/goldens/data` (see this crate's README).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use turbo_goldens_tool::{docker, models, output, paths, recipe};

const USAGE: &str = "\
usage: turbo-goldens <command>

  image                          build the generator's container image and print its id
  fetch-models <dir>             download the models into <dir> and check their SHA-512
  generate [--out <dir>] [--models <dir>]
                                 run the generator in the container; write into analysis/goldens/data or <dir>
  verify [--models <dir>]        generate into a scratch directory and compare with the checked in goldens

--models defaults to target/turbo-goldens/models, fetched as by fetch-models.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("turbo-goldens: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let Some((command, rest)) = args.split_first() else {
        return usage();
    };
    match command.as_str() {
        "image" if rest.is_empty() => {
            println!("{}", docker::build_image()?);
            Ok(ExitCode::SUCCESS)
        }
        "fetch-models" if rest.len() == 1 => {
            models::fetch(Path::new(&rest[0]))?;
            Ok(ExitCode::SUCCESS)
        }
        "generate" => {
            let opts = Options::parse(rest, true)?;
            let run = generate(&opts.models)?;
            let target = opts.out.unwrap_or_else(paths::data_dir);
            let n = output::install(run.path(), &target)?;
            println!("installed {n} goldens in {}", target.display());
            Ok(ExitCode::SUCCESS)
        }
        "verify" => {
            let opts = Options::parse(rest, false)?;
            let run = generate(&opts.models)?;
            let d = output::compare(&paths::data_dir(), run.path())?;
            if d.is_empty() {
                println!("every golden matches");
                Ok(ExitCode::SUCCESS)
            } else {
                print!("{}", d.report());
                Ok(ExitCode::FAILURE)
            }
        }
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        _ => usage(),
    }
}

fn usage() -> Result<ExitCode, String> {
    eprintln!("{USAGE}");
    Ok(ExitCode::from(2))
}

/// The options of `generate` and `verify`.
struct Options {
    out: Option<PathBuf>,
    models: PathBuf,
}

impl Options {
    fn parse(args: &[String], out_allowed: bool) -> Result<Options, String> {
        let mut opts = Options { out: None, models: paths::scratch_dir().join("models") };
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            let value = || format!("{flag} needs a directory");
            match flag.as_str() {
                "--out" if out_allowed => opts.out = Some(PathBuf::from(it.next().ok_or_else(value)?)),
                "--models" => opts.models = PathBuf::from(it.next().ok_or_else(value)?),
                _ => return Err(format!("unknown argument {flag}\n{USAGE}")),
            }
        }
        Ok(opts)
    }
}

/// Fetches the models, checks the inputs and runs the container into a
/// fresh scratch directory, which the caller installs or compares.
fn generate(models_dir: &Path) -> Result<output::RunDir, String> {
    models::fetch(models_dir)?;
    check_input_sets(&paths::inputs_dir())?;
    let scratch = paths::scratch_dir();
    fs::create_dir_all(&scratch).map_err(|e| format!("{}: {e}", scratch.display()))?;
    let run = output::RunDir::new(&scratch)?;
    docker::generate(&paths::inputs_dir(), models_dir, run.path())?;
    run.check_layout()?;
    Ok(run)
}

/// Fails unless the inputs directory holds exactly the recipe's input sets.
fn check_input_sets(inputs: &Path) -> Result<(), String> {
    let mut found: Vec<String> = fs::read_dir(inputs)
        .map_err(|e| format!("{}: {e}", inputs.display()))?
        .map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{}: {e}", inputs.display()))?;
    found.sort();
    let mut expected: Vec<String> = recipe::INPUT_SETS.iter().map(|s| s.to_string()).collect();
    expected.sort();
    if found != expected {
        return Err(format!("{} holds {found:?}; the recipe's input sets are {expected:?}", inputs.display()));
    }
    Ok(())
}
