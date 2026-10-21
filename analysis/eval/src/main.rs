//! turbo-eval: run a retrieval pipeline over evaluation sets and write a
//! record per set, or compare two directories of records with the gate
//! (docs/analysis.md, "Evaluation").

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use turbo_analysis_embed::{Device, Precision};
use turbo_chunker::terms::Bm25;
use turbo_chunker::{Budget, Cohesion, Costs, Drift, WeighBoundaries};
use turbo_eval::Result;
use turbo_eval::analysis::{Analysis, Sentencer};
use turbo_eval::chunkers::{Chunker, FixedTokens, Semantic, SentenceWindows, Signals, TurboChunker, WholeDocument};
use turbo_eval::embedder::{Embedder, Role, TurboEmbedder};
use turbo_eval::fetch::{cache_root, fetch};
use turbo_eval::lexical::{Lexical, TermRule};
use turbo_eval::pipeline::{Pipeline, Titles};
use turbo_eval::run::{gate_dirs, run_sets};
use turbo_eval::set::LoadedSet;
use turbo_eval::sets;
use turbo_structure::Syntax;

const USAGE: &str = "\
usage:
  turbo-eval fetch --sets <name,...> [--cache <dir>]
      download each named set into the cache and check its SHA-256s
  turbo-eval run --bundle <dir> --sets <name,...> [--cache <dir>] --out <dir>
                 --chunker <name> [options]
      run one pipeline over each named set and write <out>/<set>.json;
      an existing record is never replaced
  turbo-eval gate <candidate-dir> <baseline-dir>
      compare two directories of records, one per set; exit status 1
      when the gate fails, 2 on an error

The cache root is --cache, or $TURBO_EVAL_CACHE when it is not given.
Sets: scifact, nfcorpus, arguana, legalbench-rag-mini, conteb-covid-qa,
conteb-squad, gutenqa, hicbench, qasper (docs/analysis.md, \"The sets\").

run options:
  --bundle <dir>          the embedder's bundle; the reference embedder is
                          all-MiniLM-L6-v2, as turbo-bench measures it
  --device <d>            cpu, select, a device index or a backend name
                          (default cpu)
  --precision <p>         model, fastest or exact (default model)
  --sentences <s>         newline, or an OpenNLP sentence model file
                          (default: the English UD EWT model in the
                          directory $TURBO_GOLDENS_MODELS names, or
                          target/turbo-goldens/models, where
                          turbo-goldens-tool fetch-models puts it)
  --structure <s>         plain, markdown or none (default plain)
  --titles <t>            ignore, or prefix each chunk with the document's
                          title (default ignore)
  --retrieval <r>         dense (the embedder's vectors, by cosine) or
                          bm25 (the chunks' terms; nothing is embedded)
                          (default dense)
  --k1 <f>, --b <f>       BM25's parameters (default 1.2 and 0.75, Lucene's)
  --terms <t>             BM25's terms: folded (lowercased word tokens) or
                          english (also Lucene's English stop words,
                          possessive and Porter stemmer) (default folded)
  --dump-terms <dir>      with bm25, also write each set's chunk and query
                          terms there, for a reference BM25
  --repo <dir>            the clean, pushed working tree this tool was
                          built from (default: the one it was built in)

chunkers:
  --chunker document
      the whole document, its first sentence to its last, as one chunk
  --chunker fixed-tokens --tokens <n> [--overlap <n>]
      windows of n model tokens (overlap default 0)
  --chunker sentence-windows --window <n>
      windows of n sentences
  --chunker opennlp-sandbox-semantic [--similarity <f>] [--percentile <p>]
                  [--min-sentences <n>] [--max-sentences <n>]
      the sandbox's semantic chunker; unset values take its defaults
  --chunker turbo-chunker --max <n> --target <n> [--size-weight <n>]
                  [--drift] [--cohesion <weight>]
      turbo-chunker's optimizer on the structure's costs (size weight
      default 1000), with drift weighed in and a cohesion term when given";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("turbo-eval: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode> {
    match args.first().map(String::as_str) {
        Some("fetch") => fetch_cmd(&args[1..]).map(|()| ExitCode::SUCCESS),
        Some("run") => run_cmd(&args[1..]).map(|()| ExitCode::SUCCESS),
        Some("gate") if args.len() == 3 => {
            let report = gate_dirs(Path::new(&args[1]), Path::new(&args[2]))?;
            println!("{report}");
            Ok(if report.pass { ExitCode::SUCCESS } else { ExitCode::FAILURE })
        }
        _ => Err(USAGE.into()),
    }
}

/// Reads the named set from under the cache root, after its licence and
/// recorded digests are checked.
fn load_set(name: &str, cache: &Path) -> Result<LoadedSet> {
    sets::find(name)?.load(cache)
}

/// The comma separated `--sets` list.
fn set_names(o: &mut Opts) -> Result<Vec<String>> {
    Ok(o.need("--sets")?.split(',').filter(|s| !s.is_empty()).map(str::to_owned).collect())
}

fn fetch_cmd(args: &[String]) -> Result<()> {
    let mut o = Opts::parse(args)?;
    let names = set_names(&mut o)?;
    let cache = cache_root(o.take("--cache").as_deref().map(Path::new))?;
    for name in &names {
        fetch(sets::find(name)?, &cache)?;
        println!("{name}: fetched and checked in {}", cache.join(name).display());
    }
    Ok(())
}

/// `--name value` pairs and bare `--flag`s.
struct Opts(BTreeMap<String, Option<String>>);

const FLAGS: [&str; 1] = ["--drift"];

impl Opts {
    fn parse(args: &[String]) -> Result<Opts> {
        let mut m = BTreeMap::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if !a.starts_with("--") {
                return Err(format!("{a}: not an option\n{USAGE}"));
            }
            let v = if FLAGS.contains(&a.as_str()) {
                None
            } else {
                Some(it.next().ok_or_else(|| format!("{a} needs a value"))?.clone())
            };
            if m.insert(a.clone(), v).is_some() {
                return Err(format!("{a} is given twice"));
            }
        }
        Ok(Opts(m))
    }

    fn take(&mut self, name: &str) -> Option<String> {
        self.0.remove(name).flatten()
    }

    fn need(&mut self, name: &str) -> Result<String> {
        self.take(name).ok_or_else(|| format!("{name} is required\n{USAGE}"))
    }

    fn flag(&mut self, name: &str) -> bool {
        self.0.remove(name).is_some()
    }

    fn parsed<T: std::str::FromStr>(&mut self, name: &str) -> Result<Option<T>> {
        self.take(name).map(|v| v.parse().map_err(|_| format!("{name} {v}: not a valid value"))).transpose()
    }

    fn required<T: std::str::FromStr>(&mut self, name: &str) -> Result<T> {
        self.parsed(name)?.ok_or_else(|| format!("{name} is required\n{USAGE}"))
    }
}

fn device(s: &str) -> Device {
    match s {
        "cpu" => Device::Cpu,
        "select" => Device::Select,
        _ => s.parse().map_or_else(|_| Device::Backend(s.into()), Device::Index),
    }
}

/// The chunker the options name, its overhead taken from the embedder.
fn chunker(o: &mut Opts, embedder: &dyn Embedder) -> Result<Box<dyn Chunker>> {
    let name = o.need("--chunker")?;
    Ok(match name.as_str() {
        "fixed-tokens" => {
            let c = FixedTokens::new(o.required("--tokens")?, o.parsed("--overlap")?.unwrap_or(0))?;
            let row = u64::from(c.tokens()) + u64::from(embedder.overhead(Role::Document));
            if row > u64::from(embedder.max_tokens()) {
                return Err(format!(
                    "--tokens {}: with the model's overhead a window is {row} tokens, over its {}",
                    c.tokens(),
                    embedder.max_tokens()
                ));
            }
            Box::new(c)
        }
        "document" => Box::new(WholeDocument),
        "sentence-windows" => Box::new(SentenceWindows::new(o.required("--window")?)?),
        "opennlp-sandbox-semantic" => Box::new(Semantic::from_config(
            o.parsed("--similarity")?.unwrap_or(0.0),
            o.parsed("--percentile")?.unwrap_or(0),
            o.parsed("--min-sentences")?.unwrap_or(0),
            o.parsed("--max-sentences")?.unwrap_or(0),
        )?),
        "turbo-chunker" => {
            let budget = Budget {
                max: o.required("--max")?,
                target: o.required("--target")?,
                size_weight: o.parsed("--size-weight")?.unwrap_or(1000),
                overhead: embedder.overhead(Role::Document),
            };
            if budget.max > embedder.max_tokens() {
                return Err(format!("--max {}: over the model's {} tokens per row", budget.max, embedder.max_tokens()));
            }
            let signals = if o.flag("--drift") {
                {
                    let w = WeighBoundaries::default();
                    Signals::StructureDrift { drift: Drift::default(), structure: w.structure, weight: w.drift }
                }
            } else {
                Signals::Structure
            };
            let cohesion = o.parsed("--cohesion")?.map(|weight| Cohesion { weight });
            Box::new(TurboChunker { budget, costs: Costs::default(), signals, cohesion })
        }
        other => return Err(format!("--chunker {other}: not a chunker\n{USAGE}")),
    })
}

fn run_cmd(args: &[String]) -> Result<()> {
    let mut o = Opts::parse(args)?;
    let bundle = PathBuf::from(o.need("--bundle")?);
    let sets = set_names(&mut o)?;
    let cache = cache_root(o.take("--cache").as_deref().map(Path::new))?;
    let out = PathBuf::from(o.need("--out")?);
    let device = device(&o.take("--device").unwrap_or_else(|| "cpu".into()));
    let precision = match o.take("--precision").as_deref().unwrap_or("model") {
        "model" => Precision::Model,
        "fastest" => Precision::Fastest,
        "exact" => Precision::Exact,
        p => return Err(format!("--precision {p}: model, fastest or exact")),
    };
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sentences = match o.take("--sentences") {
        Some(s) if s == "newline" => Sentencer::Newline,
        Some(path) => Sentencer::maxent(Path::new(&path))?,
        None => {
            let dir = match std::env::var_os("TURBO_GOLDENS_MODELS") {
                Some(d) => PathBuf::from(d),
                None => workspace.join("target/turbo-goldens/models"),
            };
            let path = dir.join("opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin");
            if !path.is_file() {
                return Err(format!(
                    "{}: not found; download it with `cargo run -p turbo-goldens-tool -- fetch-models \
                     target/turbo-goldens/models` (or into the directory TURBO_GOLDENS_MODELS names), \
                     or give --sentences",
                    path.display()
                ));
            }
            Sentencer::maxent(&path)?
        }
    };
    let structure = match o.take("--structure").as_deref().unwrap_or("plain") {
        "plain" => Some(Syntax::Plain),
        "markdown" => Some(Syntax::Markdown),
        "none" => None,
        s => return Err(format!("--structure {s}: plain, markdown or none")),
    };
    let titles = match o.take("--titles").as_deref().unwrap_or("ignore") {
        "ignore" => Titles::Ignore,
        "prefix" => Titles::Prefix,
        t => return Err(format!("--titles {t}: ignore or prefix")),
    };
    let lexical = match o.take("--retrieval").as_deref().unwrap_or("dense") {
        "dense" => None,
        "bm25" => {
            let terms = match o.take("--terms").as_deref().unwrap_or("folded") {
                "folded" => TermRule::Folded,
                "english" => TermRule::English,
                t => return Err(format!("--terms {t}: folded or english")),
            };
            let k1: f64 = o.parsed("--k1")?.unwrap_or(Bm25::LUCENE.k1);
            let b: f64 = o.parsed("--b")?.unwrap_or(Bm25::LUCENE.b);
            if !(k1.is_finite() && k1 >= 0.0 && (0.0..=1.0).contains(&b)) {
                return Err(format!("--k1 {k1} --b {b}: k1 at least 0 and b from 0 to 1"));
            }
            Some(Lexical { model: Bm25 { k1, b }, terms, dump: o.take("--dump-terms").map(PathBuf::from) })
        }
        r => return Err(format!("--retrieval {r}: dense or bm25")),
    };
    let repo = o.take("--repo").map_or(workspace, PathBuf::from);
    if sets.is_empty() {
        return Err("--sets names no set".into());
    }

    // Refused before the model loads, and checked again before writing.
    let source = turbo_eval::source::source(&repo)?;
    let embedder = TurboEmbedder::open(&bundle, device, precision)?;
    let chunker = chunker(&mut o, &embedder)?;
    if let Some(unknown) = o.0.keys().next() {
        return Err(format!("{unknown}: not an option here\n{USAGE}"));
    }
    let mut pipeline = Pipeline {
        analysis: Analysis { sentences, structure },
        chunker,
        embedder: Box::new(embedder),
        titles,
        lexical,
    };
    let written = run_sets(&mut pipeline, &sets, &cache, load_set, &out, &source)?;
    if turbo_eval::source::source(&repo)? != source {
        return Err("the working tree moved during the run; the records written name the commit it started at".into());
    }
    for p in written {
        println!("{}", p.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_usage_names_every_set() {
        for s in sets::ALL {
            assert!(USAGE.contains(s.name), "{} is missing from the usage", s.name);
        }
    }

    #[test]
    fn an_unknown_set_is_refused_by_name() {
        let e = load_set("no-such-set", Path::new("/nonexistent")).unwrap_err();
        assert!(e.contains("no-such-set"), "{e}");
    }
}
