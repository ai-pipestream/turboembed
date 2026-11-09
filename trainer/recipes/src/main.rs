//! `turbo-train-head`: checks a corpus, scores its majority baseline, or
//! trains a head on a frozen encoder, writing a receipt each time.
//! ```text
//! turbo-train-head check    --corpus <id> --dir <corpus dir> [--unrecorded-corpus]
//! turbo-train-head baseline --corpus <id> --dir <corpus dir> --out <dir> [--unrecorded-corpus]
//! turbo-train-head train    --corpus <id> --dir <corpus dir> --bundle <encoder bundle> --out <dir>
//!     [--device cpu|cuda|levelzero|metal] [--precision model|exact] [--pooling first|mean] [--crf]
//!     [--bilstm <hidden>x<layers>] [--hash-embed <dim>] [--hash-rows <norm>,<prefix>,<suffix>,<shape>]
//!     [--seed N] [--epochs N] [--batch N] [--lr X] [--patience N] [--tolerance X]
//!     [--threads N] [--max-train N] [--commit <sha>] [--unrecorded-corpus] [--against <run dir>]
//!     Multiple UD treebanks: --corpus <bank,bank,...> [--column upos|xpos|feats] [--sample 0..1]
//! turbo-train-head cache    --corpus <id> --dir <corpus dir> --bundle <encoder bundle> --out <dir> --layers N
//!     [--device ..] [--precision ..] [--pooling ..] [--max-batch N] [--splits train,dev,test]
//!     [--commit <sha>] [--unrecorded-corpus]
//! turbo-train-head fit-cache --corpus <id> --dir <corpus dir> --cache <cache dir> --out <dir>
//!     (--layer N | --mix [--from-layer N]) [head and training settings as train] [--commit <sha>]
//! turbo-train-head train-lemma --corpus <treebank> ...   (as train, no --crf or --hash-embed;
//!     [--min-tree-freq N] [--seen-lookup K]; multiple banks: --corpus <bank,bank,...> [--sample 0..1])
//! turbo-train-head export   --corpus <id> --dir <corpus dir> --out <test.conllu>
//! turbo-train-head peer     --corpus <id> --dir <corpus dir> --peer <name> --predictions <peer.conllu>
//!     --out <dir> [--column upos|xpos|feats|lemma|deps|bio] [--map FROM=TO,...] [--commit <sha>]
//! turbo-train-head train-parse --corpus <treebank> ...   (as train, no --crf or --hash-embed;
//!     [--arc N] [--rel N] [--tune-encoder [--lora N] [--bf16] [--max-test N]])
//!     Multiple banks: --corpus <bank,bank,...> [--sample 0..1]
//! turbo-train-head train-split --corpus <treebank> ...   (as train, no --crf or --hash-embed;
//!     [--min-tree-freq N] [--seen-lookup K]; multiple banks: --corpus <bank,bank,...> [--sample 0..1])
//! turbo-train-head train-relation --corpus re-docred ...   (as train, no --crf or --hash-embed;
//!     [--mlp N] [--type-dim N])
//! turbo-train-head train-open-span --corpus wnut-17 ...  (open-label span typing, #1735;
//!     [--balance] to repeat smaller training corpora to the largest split)
//! turbo-train-head eval-gliner --corpus wnut-17 ... --package <converted GLiNER> --hold-out <types>
//!     [--encoder-bundle <sealed encoder> --device cpu|cuda|levelzero|metal] [--threshold X] [--threads N]
//! turbo-train-head train-coref --corpus litbank ...      (as train, no --crf or --hash-embed;
//!     [--mlp N] [--max-antecedents N] [--max-words N] [--width-dim N] [--against <run dir>])
//! turbo-train-head train-sentences --corpus <masc-pos|treebank> ...   (as train)
//! turbo-train-head distill-head --corpus <id> --dir <corpus dir> --bundle <encoder bundle> --out <dir>
//!     [--student-bundle <student bundle>] [--teacher <teacher dir>] [--temperature T] [--alpha A]
//!     [--max-accuracy-drop X] [--min-speedup X] [--allow-gate-failure] [head and training settings]
//! ```
//! `cache` writes every split's word features at the encoder's last N
//! layers, for heads on one layer or a mix of them (`cache` module).
//! Exit status 2 on any error. `--out` must not exist yet. A run claims
//! it before anything else, by making it, which fails when it exists, so
//! of two runs given one `--out` exactly one goes on; a run that fails
//! removes it, so a refused run leaves none.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use turbo::manifest::{HashEmbed, WordPooling};
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_tagger::WordEncoder;
use turbo_trainer::Config;
use turbo_trainer_recipes::cache;
use turbo_trainer_recipes::cli::{Args, HeadArgs, OutDir, commit, encoder_args, head_args};
use turbo_trainer_recipes::corpus::{Sentence, Task};
use turbo_trainer_recipes::eval::{Majority, Seen, bootstrap, evaluate};
use turbo_trainer_recipes::fit_cache::CacheTask;
use turbo_trainer_recipes::prior;
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, HeadOptions, named, replace, spec, write};
use turbo_trainer_recipes::ud_heads::UdHead;
use turbo_trainer_recipes::{Error, Result, coarse, read_corpus, safetensors};

mod baseline_cmd;
mod eval_gliner;
mod peer_cmd;
mod summarize;
mod train_coref;
mod train_open_span;
mod train_relation;
mod train_sentences;
mod train_split;
mod train_tuned;
mod train_ud;

const RESAMPLES: usize = 1000;

fn main() -> ExitCode {
    match run(std::env::args().collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("turbo-train-head: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(argv: Vec<String>) -> Result<()> {
    let command = argv.get(1).cloned().unwrap_or_default();
    let mut a = Args { rest: argv[2.min(argv.len())..].to_vec() };
    match command.as_str() {
        "assemble" => return assemble_cmd(a),
        "summarize" => return summarize::run(&argv[2..]),
        _ => {}
    }
    let corpus_id = a.required("--corpus")?;
    let dir = PathBuf::from(a.required("--dir")?);
    let unrecorded = a.flag("--unrecorded-corpus");
    match command.as_str() {
        "check" if corpus_id == turbo_trainer_recipes::redocred::ID => {
            a.done()?;
            let c = train_relation::read(&corpus_id, &dir, unrecorded)?;
            println!("{}", serde_json::to_string_pretty(&c.record).expect("the record serializes"));
            println!("labels: {}", c.labels().len());
            Ok(())
        }
        "check" => peer_cmd::check(&corpus_id, &dir, unrecorded, a),
        "baseline" => baseline_cmd::baseline_cmd(&argv, &corpus_id, &dir, unrecorded, a),
        "train" => train(&argv, &corpus_id, &dir, unrecorded, a),
        "cache" => cache::command(&argv, &corpus_id, &dir, unrecorded, a),
        "fit-cache" | "train-cached" => {
            turbo_trainer_recipes::fit_cache::command(&argv, &corpus_id, &dir, unrecorded, a)
        }
        "sweep" => turbo_trainer_recipes::sweep::command(&argv, &corpus_id, &dir, unrecorded, a),
        "distill-head" => turbo_trainer_recipes::distill::command(&argv, &corpus_id, &dir, unrecorded, a),
        "predict-ner" => turbo_trainer_recipes::ner_cmd::predict(&corpus_id, &dir, unrecorded, a),
        "compare-ner" => turbo_trainer_recipes::ner_cmd::compare(&corpus_id, &dir, unrecorded, a),
        "export" => peer_cmd::export(&corpus_id, &dir, unrecorded, a),
        "peer" => peer_cmd::peer(&argv, &corpus_id, &dir, unrecorded, a),
        "train-lemma" | "train-parse" => {
            let head = if command == "train-lemma" { UdHead::Lemma } else { UdHead::Deps };
            train_ud::train(&argv, head, &corpus_id, &dir, unrecorded, a)
        }
        "train-split" => train_split::train(&argv, &corpus_id, &dir, unrecorded, a),
        "train-relation" => train_relation::train(&argv, &corpus_id, &dir, unrecorded, a),
        "train-open-span" => train_open_span::train(&argv, &corpus_id, &dir, unrecorded, a),
        "eval-gliner" => eval_gliner::evaluate_command(&argv, &corpus_id, &dir, unrecorded, a),
        "train-coref" => train_coref::train(&argv, &corpus_id, &dir, unrecorded, a),
        "train-sentences" => train_sentences::train(&argv, &corpus_id, &dir, unrecorded, a),
        _ => Err(Error(format!(
            "unknown command {command:?}: check, baseline, train, train-lemma, train-parse, train-split, train-relation, train-open-span, eval-gliner, train-coref, train-sentences, distill-head, assemble, summarize, cache, fit-cache, train-cached, sweep, export, peer, predict-ner or compare-ner"
        ))),
    }
}

fn assemble_cmd(mut a: Args) -> Result<()> {
    let bundle = PathBuf::from(a.required("--bundle")?);
    let runs: Vec<PathBuf> = a.values("--run")?.into_iter().map(PathBuf::from).collect();
    if runs.is_empty() {
        return Err(Error("--run is required at least once".into()));
    }
    let out_arg = a.required("--out")?;
    let tuned_encoder = a.value("--tuned-encoder")?.map(PathBuf::from);
    let zoo_id = a.value("--zoo-id")?;
    let component = match a.value("--component")? {
        Some(c) => Some(c.parse().map_err(Error)?),
        None => None,
    };
    let language = a.value("--language")?;
    let version = a.value("--version")?.unwrap_or_else(|| "1".into());
    let license_terms = a.value("--license-terms")?;
    let url_prefix = a.value("--url-prefix")?;
    let origin = a.value("--origin")?;
    let zoo_out = a.value("--zoo-out")?.map(PathBuf::from);
    a.done()?;
    let zoo_req = match zoo_id {
        Some(id) => {
            let terms =
                license_terms.ok_or_else(|| Error("--license-terms is required when --zoo-id is given".into()))?;
            let prefix = url_prefix.ok_or_else(|| Error("--url-prefix is required when --zoo-id is given".into()))?;
            Some((id, terms, prefix))
        }
        None => None,
    };
    let mut out = OutDir::claim(&out_arg)?;

    let run_refs: Vec<&Path> = runs.iter().map(PathBuf::as_path).collect();
    let assembled = match tuned_encoder {
        Some(tuned) => turbo_trainer_recipes::assemble::assemble_tuned_bundle(&bundle, &run_refs, &tuned, &out.path)?,
        None => turbo_trainer_recipes::assemble::assemble_bundle(&bundle, &run_refs, &out.path)?,
    };
    out.keep = true;
    println!(
        "{}: assembled {} head(s), tier {:?}, manifest {}",
        out.path.display(),
        assembled.runs.len(),
        assembled.license_tier,
        assembled.manifest_sha256
    );
    if let Some((id, license_terms, url_prefix)) = zoo_req {
        let kind = if assembled.runs.len() == 1 { turbo_zoo::Kind::Head } else { turbo_zoo::Kind::Model };
        let entry = turbo_trainer_recipes::assemble_zoo::zoo_artifact(
            &assembled,
            &turbo_trainer_recipes::assemble_zoo::ZooEntryOptions {
                id,
                kind,
                component,
                language,
                version,
                license_terms,
                url_prefix,
                origin,
            },
        )?;
        let json = serde_json::to_string_pretty(&entry).expect("artifact serializes");
        if let Some(p) = zoo_out {
            std::fs::write(&p, format!("{json}\n")).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        } else {
            println!("{json}");
        }
    }
    Ok(())
}

/// The settings `train` and the UD commands share.
struct Opts {
    bundle: PathBuf,
    out: OutDir,
    device: Device,
    precision: Precision,
    pooling: WordPooling,
    crf: bool,
    bilstm: Option<(u32, u32)>,
    hash: Option<HashEmbed>,
    c: Config,
    max_train: usize,
    commit: String,
}

fn options(a: &mut Args) -> Result<Opts> {
    let bundle = PathBuf::from(a.required("--bundle")?);
    let out = OutDir::claim(&a.required("--out")?)?;
    let (device, precision, pooling) = encoder_args(a)?;
    let HeadArgs { crf, bilstm, hash, config: c } = head_args(a)?;
    let max_train: usize = a.parsed("--max-train", 0)?;
    let commit = commit(a.value("--commit")?);
    Ok(Opts { bundle, out, device, precision, pooling, crf, bilstm, hash, c, max_train, commit })
}

fn train(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    if a.rest.iter().any(|x| x == "--cache") {
        return turbo_trainer_recipes::fit_cache::command_with_task(
            argv,
            corpus_id,
            dir,
            unrecorded,
            Some(CacheTask::Tag),
            a,
        );
    }
    let column = if corpus_id.contains(',') {
        let name = a.value("--column")?.unwrap_or_else(|| "upos".into());
        Some(turbo_trainer_recipes::ud_multi::column(&name)?)
    } else {
        None
    };
    let sample = if column.is_some() { a.parsed("--sample", 1.0)? } else { 1.0 };
    let against = a.value("--against")?.map(PathBuf::from);
    let Opts { bundle, out, device, precision, pooling, crf, bilstm, hash, c, max_train, commit } = options(&mut a)?;
    a.done()?;

    let t = Instant::now();
    let (corpus, languages) = match column {
        Some(column) => {
            let multi = turbo_trainer_recipes::ud_multi::read(
                corpus_id,
                column,
                dir.to_str().ok_or_else(|| Error("corpus directory is not UTF-8".into()))?,
                unrecorded,
                sample,
            )?;
            (multi.corpus, Some((multi.languages, multi.train_languages, multi.dev_languages, multi.test_languages)))
        }
        None => (read_corpus(corpus_id, dir, unrecorded)?, None),
    };
    let read = t.elapsed().as_secs_f64();
    let mut head = spec(&corpus, &HeadOptions { pooling, crf, bilstm, hash })?;
    if let Some((names, _, _, _)) = &languages {
        head.language = Some(turbo::manifest::LanguageInput { languages: names.clone(), dim: 16 });
        head.validate("head").map_err(|e| Error(e.message))?;
    }
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    // The head to score against, read before training so a bad one fails
    // first.
    let prior = against.as_deref().map(|dir| prior::tagger(dir, &head, &encoder)).transpose()?;

    let t = Instant::now();
    let capped = max_train.min(corpus.splits.train.len());
    let train_set = if max_train == 0 { &corpus.splits.train[..] } else { &corpus.splits.train[..capped] };
    let train_languages = languages.as_ref().map(|(_, rows, _, _)| &rows[..train_set.len()]);
    let dev_languages = languages.as_ref().map(|(_, _, rows, _)| &rows[..]);
    let (train_ex, _) = turbo_trainer_recipes::run::examples_with_languages(
        &mut enc,
        train_set,
        &head.labels,
        pooling,
        train_languages,
    )?;
    let (dev_ex, dev_left_out) = turbo_trainer_recipes::run::examples_with_languages(
        &mut enc,
        &corpus.splits.dev,
        &head.labels,
        pooling,
        dev_languages,
    )?;
    let features = t.elapsed().as_secs_f64();
    let feature_words = train_ex.iter().chain(&dev_ex).map(|e| e.labels.len() as u64).sum();

    // The best epoch's head is on disk whenever it improves, the epoch and
    // its dev score in the same file's metadata, so a run stopped midway
    // keeps a head and the epoch it is from, replaced together by one
    // rename; from the first one on, the directory stays if the run fails.
    let t = Instant::now();
    let spec_json = serde_json::to_vec_pretty(&head).expect("a head spec serializes");
    let hidden = enc.hidden();
    let mut out = out;
    let (model, report) = {
        let dir = out.path.clone();
        let mut saved = false;
        let mut save = |m: &turbo_trainer::Tagger<f32>, epoch: usize, dev: f64| -> turbo_trainer_recipes::Result<()> {
            if !saved {
                replace(&dir, "head.json", &spec_json)?;
                saved = true;
            }
            let at = [(CHECKPOINT_EPOCH, (epoch + 1).to_string()), (CHECKPOINT_DEV, dev.to_string())];
            replace(&dir, "head.safetensors", &safetensors::write_with(&named(&head, hidden, m.tensors()), &at))?;
            Ok(())
        };
        let r = turbo_trainer::fit_with(&head, hidden, &train_ex, &dev_ex, &c, &mut |m, e, d| {
            save(m, e, d).map_err(|x| turbo::status::invalid(x.0))
        });
        out.keep = out.keep || saved_any(&dir);
        r.map_err(|e| Error(e.message))?
    };
    drop((train_ex, dev_ex));
    let trained = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let tagger = model.to_token_tagger().map_err(|e| Error(e.message))?;
    let test = &corpus.splits.test;
    let test_languages = languages.as_ref().map(|(_, _, _, rows)| &rows[..]);
    let predicted = turbo_trainer_recipes::run::predict_with_languages(&mut enc, &tagger, test, test_languages)?;
    let base = baseline(train_set, test);
    let seen = Seen::of(train_set);
    let task: Task = corpus.record.task;
    let against = match prior {
        Some((name, prior, files)) => {
            let theirs = turbo_trainer_recipes::run::predict_with_languages(&mut enc, &prior, test, test_languages)?;
            Some(BaselineEval {
                name,
                baseline: evaluate(task, test, &theirs, &seen, coarse(corpus_id)),
                comparison: bootstrap(task, test, &predicted, &theirs, RESAMPLES, c.seed),
                files,
            })
        }
        None => None,
    };
    let by_language = match &languages {
        Some((names, train_rows, _, test_rows)) => turbo_trainer_recipes::eval::language_evals(
            train_set,
            test,
            &predicted,
            &train_rows[..train_set.len()],
            test_rows,
            names,
            turbo_trainer_recipes::eval::LanguageEvalOptions {
                task,
                baseline: |train_rows: &[usize], test_rows: &[usize]| {
                    let train_part: Vec<Sentence> = train_rows.iter().map(|&i| train_set[i].clone()).collect();
                    let test_part: Vec<Sentence> = test_rows.iter().map(|&i| test[i].clone()).collect();
                    baseline(&train_part, &test_part)
                },
                resamples: RESAMPLES,
                seed: c.seed,
                attachment: None,
            },
        )?,
        None => Vec::new(),
    };
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(task, test, &predicted, &seen, coarse(corpus_id))),
        baseline: evaluate(task, test, &base, &seen, coarse(corpus_id)),
        dictionary: None,
        open_span: None,
        comparison: Some(bootstrap(task, test, &predicted, &base, RESAMPLES, c.seed)),
        uas: None,
        rules: None,
        against,
        by_language,
    };
    let eval_s = t.elapsed().as_secs_f64();

    // The final head is the best checkpoint's, published with the same
    // epoch and dev score, so no moment of finishing leaves weights
    // without the record of where they are from.
    let best = report.best_epoch;
    let at = [(CHECKPOINT_EPOCH, (best + 1).to_string()), (CHECKPOINT_DEV, report.dev[best].to_string())];
    let weights = safetensors::write_with(&named(&head, enc.hidden(), model.tensors()), &at);
    let outputs = vec![replace(&out.path, "head.safetensors", &weights)?, replace(&out.path, "head.json", &spec_json)?];
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        unix_time: unix_time(),
        corpus: corpus.record.clone(),
        encoder: Some(encoder),
        head: Some(head),
        peer: None,
        config: Some(ConfigRecord {
            seed: c.seed,
            epochs: c.epochs,
            batch: c.batch,
            lr: c.lr,
            weight_decay: c.weight_decay,
            clip: c.clip,
            warmup: c.warmup,
            patience: c.patience,
            tolerance: c.tolerance,
            threads: c.threads,
            max_train,
            dev_left_out,
            sample: column.map(|_| sample),
        }),
        training: Some(TrainingRecord {
            train_loss: report.train_loss,
            dev: report.dev,
            best_epoch: best + 1,
            epoch_seconds: report.epoch_seconds,
        }),
        eval,
        timing_s: Timing { read, features, train: trained, eval: eval_s, feature_words },
        outputs,
    };
    finish(out, receipt)
}

/// The labels for `test` of the majority baseline learned from `train`,
/// the sentences the model is given, so the two compare on equal data.
fn baseline(train: &[Sentence], test: &[Sentence]) -> Vec<Vec<String>> {
    let m = Majority::train(train);
    test.iter().map(|s| m.predict(&s.words)).collect()
}

fn encoder_record(bundle: &Path, enc: &WordEncoder) -> Result<EncoderRecord> {
    let manifest =
        std::fs::read(bundle.join("manifest.json")).map_err(|e| Error(format!("{}: {e}", bundle.display())))?;
    let m: turbo::manifest::Manifest =
        serde_json::from_slice(&manifest).map_err(|e| Error(format!("{}: {e}", bundle.display())))?;
    let x = enc.execution();
    Ok(EncoderRecord {
        model: m.model.id,
        revision: m.model.revision,
        manifest_sha256: turbo::bundle::sha256_hex(&manifest),
        hidden: enc.hidden(),
        layers: m.architecture.as_ref().map(|a| a.layers),
        width: enc.width(),
        backend: x.backend.clone(),
        device: x.device.clone(),
        precision: x.precision.clone(),
    })
}
/// Whether a checkpoint was written into `dir`.
fn saved_any(dir: &Path) -> bool {
    dir.join("head.safetensors").is_file()
}

fn finish(mut out: OutDir, receipt: Receipt) -> Result<()> {
    let text = serde_json::to_string_pretty(&receipt).expect("the receipt serializes");
    write(&out.path, "receipt.json", text.as_bytes())?;
    out.keep = true;
    if let Some(w) = receipt.timing_s.words_per_s() {
        println!(
            "features: {} train and dev words in {:.1} s, {w:.0} words/s",
            receipt.timing_s.feature_words, receipt.timing_s.features
        );
    }
    let model = receipt.eval.model.as_ref().map_or("-".into(), |m| format!("{:.4}", m.score()));
    println!(
        "{}: {} model {model} baseline {:.4}; {}",
        receipt.corpus.id,
        receipt.eval.split,
        receipt.eval.baseline.score(),
        out.path.join("receipt.json").display()
    );
    Ok(())
}
