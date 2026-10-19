//! `turbo-goldens`: makes and checks the golden layer files in
//! `analysis/goldens/data` (see this crate's README).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use turbo_goldens_tool::{docker, models, output, paths, recipe};

const USAGE: &str = "\
usage: turbo-goldens <command>

  image                          build the generator's container image and print its id
  fetch-models <dir> [--local]   download the models into <dir> and check their SHA-512; --local also
                                 downloads the OpenNLP 1.5 models, which CI does not fetch
  generate [--out <dir>] [--models <dir>]
                                 run the generator in the container; write the goldens and their decision
                                 point dumps into analysis/goldens/data or <dir>
  verify [--models <dir>]        generate into a scratch directory and compare with the checked in files;
                                 check that every checked in golden has a dump whose header parses
  fuzz [--seed <n>] [--count <n>] [--sets <a,b>] [--out <dir>] [--models <dir>] [--image <tag>]
       [--reduce <n>] [--no-reduce] [--reduce-documented] [--rounds <n>] [--reuse]
                                 generate <count> texts from <seed>, run the generator over them in the
                                 container, compare every golden with the Rust annotators and check the
                                 offset invariants; reduce up to <reduce> diverging texts (10) to minimal
                                 inputs under <out>/candidates; --reuse compares an earlier run's output
                                 again without the container (see this crate's README); the command
                                 is behind the crate's fuzz feature (--features fuzz)

--models defaults to target/turbo-goldens/models, fetched as by fetch-models --local.";

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
            models::fetch(Path::new(&rest[0]), models::Models::Published)?;
            Ok(ExitCode::SUCCESS)
        }
        "fetch-models" if rest.len() == 2 && rest[1] == "--local" => {
            models::fetch(Path::new(&rest[0]), models::Models::WithLocal)?;
            Ok(ExitCode::SUCCESS)
        }
        "generate" => {
            let opts = Options::parse(rest, true)?;
            let run = generate(&opts.models)?;
            let target = opts.out.unwrap_or_else(paths::data_dir);
            let n = output::install(run.path(), &target)?;
            println!("installed {n} goldens and dumps in {}", target.display());
            Ok(ExitCode::SUCCESS)
        }
        "verify" => {
            let opts = Options::parse(rest, false)?;
            let run = generate(&opts.models)?;
            let dumps = output::check_dumps(&paths::data_dir())?;
            println!("{dumps} checked in dumps have a golden and a header that parses");
            let d = output::compare(&paths::data_dir(), run.path())?;
            if d.is_empty() {
                println!("every golden and dump matches");
                Ok(ExitCode::SUCCESS)
            } else {
                print!("{}", d.report());
                Ok(ExitCode::FAILURE)
            }
        }
        "fuzz" => fuzz_command(rest),
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

/// Fetches the models, checks the fixtures and the inputs and runs the
/// container into a fresh scratch directory, which the caller installs or
/// compares.
fn generate(models_dir: &Path) -> Result<output::RunDir, String> {
    models::fetch(models_dir, models::Models::WithLocal)?;
    models::check_fixtures(&paths::fixtures_dir())?;
    check_input_sets(&paths::inputs_dir())?;
    let scratch = paths::scratch_dir();
    fs::create_dir_all(&scratch).map_err(|e| format!("{}: {e}", scratch.display()))?;
    let run = output::RunDir::new(&scratch)?;
    docker::generate(&paths::inputs_dir(), &paths::rates_dir(), models_dir, &paths::fixtures_dir(), run.path())?;
    run.check_layout()?;
    output::check_dumps(run.path())?;
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

/// The `fuzz` command, with the feature that brings its crates in.
#[cfg(feature = "fuzz")]
fn fuzz_command(rest: &[String]) -> Result<ExitCode, String> {
    use turbo_goldens_tool::fuzz;
    let opts = fuzz::Options::parse(rest).map_err(|e| format!("{e}\n{USAGE}"))?;
    let summary = fuzz::run(&opts)?;
    println!(
        "fuzz: {} inputs, {} goldens compared, {} problems, {} candidates written under {}",
        summary.inputs,
        summary.goldens,
        summary.problems,
        summary.candidates.len(),
        summary.out.display()
    );
    Ok(if summary.problems == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE })
}

#[cfg(not(feature = "fuzz"))]
fn fuzz_command(_rest: &[String]) -> Result<ExitCode, String> {
    Err("the fuzz command is behind the fuzz feature: cargo run -p turbo-goldens-tool --features fuzz -- fuzz ..."
        .into())
}
