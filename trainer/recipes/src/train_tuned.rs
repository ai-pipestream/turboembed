//! `train-parse --tune-encoder`: R2 parser training with the encoder
//! weights and head saved together at every best dev checkpoint.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use turbo::bundle::Bundle;
use turbo::head::tagger::NO_WORD;
use turbo::manifest::{Biaffine, Format, GraphInput, TensorRole};
use turbo::tokenizer::Tokenizer;
use turbo_analysis_embed::{Device, EncoderOptions};
use turbo_tagger::WordEncoder;
use turbo_trainer::encoder::{Encoder, Lora, Work, pool};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Parser, TunedParseExample, fit_tuned_parser_with};
use turbo_trainer_recipes::corpus::Task;
use turbo_trainer_recipes::eval::{Seen, bootstrap, evaluate};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, replace};
use turbo_trainer_recipes::ud::{TREEBANKS, UdSentence, read_treebank};
use turbo_trainer_recipes::ud_heads::{
    self, UdHead, UdOptions, deps_baseline, predict_parses, record, relations, spec, uas,
};
use turbo_trainer_recipes::{Error, Result, safetensors};

use super::{Args, Opts, RESAMPLES, encoder_record, finish, options, saved_any};

pub(super) fn train(
    argv: &[String],
    bank: &str,
    dir: &Path,
    unrecorded: bool,
    lora: Option<usize>,
    bf16: bool,
    mut a: Args,
) -> Result<()> {
    let bank = *TREEBANKS.iter().find(|b| b.id == bank).ok_or_else(|| Error(format!("unknown treebank {bank:?}")))?;
    let biaffine = Biaffine { arc: a.parsed("--arc", 256)?, rel: a.parsed("--rel", 64)? };
    let Opts { bundle, mut out, device, precision: _, pooling, crf, bilstm, hash, mut c, max_train, commit } =
        options(&mut a)?;
    a.done()?;
    if crf || hash.is_some() {
        return Err(Error("a tuned parser takes neither --crf nor --hash-embed".into()));
    }
    let cuda = matches!(&device, Device::Backend(d) if d == "cuda");
    if !cuda && device != Device::Cpu {
        return Err(Error("a tuned parser runs on cpu or cuda".into()));
    }
    if bf16 {
        c.loss_scale = Some(Default::default());
    }

    let t = Instant::now();
    let (mut splits, files, recorded) = read_treebank(&bank, dir, unrecorded)?;
    if max_train > 0 {
        splits.train.truncate(max_train);
    }
    let record = record(&bank, UdHead::Deps, &splits, files, recorded);
    let read = t.elapsed().as_secs_f64();
    let labels = relations(&splits.train);
    let spec = spec(UdHead::Deps, labels, &record, &UdOptions { pooling, bilstm, biaffine })?;
    let mut words = WordEncoder::open(&bundle, &EncoderOptions::default()).map_err(|e| Error(e.message))?;
    let mut encoder_record = encoder_record(&bundle, &words)?;
    encoder_record.backend = if cuda { "trainer-cuda" } else { "trainer-cpu" }.into();
    encoder_record.device = if cuda { "0" } else { "cpu" }.into();
    encoder_record.precision = if bf16 { "bf16" } else { "f32" }.into();
    let hidden = words.hidden();

    let t = Instant::now();
    let train = examples(&mut words, &bundle, &splits.train, &spec.labels, true)?;
    let (dev, dev_left_out) = {
        let all = examples(&mut words, &bundle, &splits.dev, &spec.labels, false)?;
        let mut kept = Vec::new();
        let mut left = 0;
        for (s, e) in splits.dev.iter().zip(all) {
            if s.rels.iter().all(|r| spec.labels.contains(r)) {
                kept.push(e);
            } else {
                left += 1;
            }
        }
        (kept, left)
    };
    let features = t.elapsed().as_secs_f64();
    let feature_words = splits.train.iter().chain(&splits.dev).map(|s| s.words.len() as u64).sum();
    let names = tensor_names(&bundle)?;
    let spec_json = serde_json::to_vec_pretty(&spec).expect("a head spec serializes");

    let t = Instant::now();
    let base = Encoder::load(&bundle).map_err(|e| Error(e.message))?;
    let base = match lora {
        Some(rank) => base
            .with_lora(Lora { rank, alpha: 2.0 * rank as f64 }, &mut Rng::new(c.seed))
            .map_err(|e| Error(e.message))?,
        None => base,
    };
    let base = if bf16 { base.with_bf16() } else { base };
    let dir = out.path.clone();
    let (encoder, head, report) = if cuda {
        fit_cuda(base, &spec, &train, &dev, &c, &dir, &spec_json, &names)?
    } else {
        let (model, report) = fit_tuned_parser_with(base, &spec, &train, &dev, &c, &mut |m, epoch, dev| {
            let encoder = m.encoder().named(&names)?;
            checkpoint(&dir, &spec, &spec_json, hidden, m.head(), &encoder, epoch, dev)
        })
        .map_err(|e| Error(e.message))?;
        let (encoder, head) = model.into_parts();
        (encoder, head, report)
    };
    out.keep = out.keep || saved_any(&dir);
    let trained = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let test = examples(&mut words, &bundle, &splits.test, &spec.labels, false)?;
    let test_f = self::features(&encoder, &test, pooling)?;
    let parser = head.to_core().map_err(|e| Error(e.message))?;
    let predicted = predict_parses(&parser, &splits.test, &test_f)?;
    let base = deps_baseline(&splits.train, &splits.test);
    let gold = UdHead::Deps.sentences(&splits.test);
    let seen = Seen::of(&UdHead::Deps.sentences(&splits.train));
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(Task::Tags, &gold, &predicted, &seen, None)),
        baseline: evaluate(Task::Tags, &gold, &base, &seen, None),
        dictionary: None,
        open_span: None,
        comparison: Some(bootstrap(Task::Tags, &gold, &predicted, &base, RESAMPLES, c.seed)),
        uas: Some(Uas { model: uas(&gold, &predicted), baseline: uas(&gold, &base) }),
        rules: None,
        against: None,
    };
    let eval_s = t.elapsed().as_secs_f64();
    let best = report.best_epoch;
    let at = [(CHECKPOINT_EPOCH, (best + 1).to_string()), (CHECKPOINT_DEV, report.dev[best].to_string())];
    let encoder_weights = safetensors::write_with(&encoder.named(&names).map_err(|e| Error(e.message))?, &at);
    let head_weights = safetensors::write_with(&ud_heads::named(UdHead::Deps, &spec, hidden, head.tensors()), &at);
    let outputs = vec![
        replace(&out.path, "head.safetensors", &head_weights)?,
        replace(&out.path, "encoder.safetensors", &encoder_weights)?,
        replace(&out.path, "head.json", &spec_json)?,
    ];
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        unix_time: unix_time(),
        corpus: record,
        encoder: Some(encoder_record),
        head: Some(spec),
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
            sample: None,
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

fn examples(
    words: &mut WordEncoder,
    bundle: &Path,
    part: &[UdSentence],
    rels: &[String],
    filter: bool,
) -> Result<Vec<TunedParseExample>> {
    let index: BTreeMap<&str, u32> = rels.iter().enumerate().map(|(i, r)| (r.as_str(), i as u32)).collect();
    let bundle = Bundle::open(bundle).map_err(|e| Error(e.message))?;
    let tokenizer = Tokenizer::load(&bundle).map_err(|e| Error(e.message))?;
    let (cls, sep) = tokenizer.template_ends();
    let mut out = Vec::new();
    for s in part {
        let gold: Option<Vec<u32>> = s.rels.iter().map(|r| index.get(r.as_str()).copied()).collect();
        if filter && gold.is_none() {
            continue;
        }
        let mut ids: Vec<u32> = cls.iter().map(|&id| id as u32).collect();
        let mut map = vec![NO_WORD; ids.len()];
        for (i, word) in s.words.iter().enumerate() {
            let pieces = words.pieces(word).map_err(|e| Error(e.message))?;
            let n = pieces.len();
            ids.extend(pieces.into_iter().map(|id| id as u32));
            map.extend(std::iter::repeat_n(i as u32, n));
        }
        ids.extend(sep.iter().map(|&id| id as u32));
        map.extend(std::iter::repeat_n(NO_WORD, ids.len() - map.len()));
        out.push(TunedParseExample {
            ids,
            words: map,
            heads: s.heads.clone(),
            rels: gold.unwrap_or_else(|| vec![0; s.rels.len()]),
        });
    }
    Ok(out)
}

fn features(
    encoder: &Encoder<f32>,
    examples: &[TunedParseExample],
    pooling: turbo::manifest::WordPooling,
) -> Result<Vec<Vec<f32>>> {
    let mut work = Work::default();
    let mut out = Vec::with_capacity(examples.len());
    for e in examples {
        encoder.forward(&e.ids, &mut work).map_err(|e| Error(e.message))?;
        let (mut features, mut count) = (Vec::new(), Vec::new());
        pool::forward(&work.out, &e.words, encoder.geometry().hidden, pooling, &mut features, &mut count)
            .map_err(|e| Error(e.message))?;
        out.push(features);
    }
    Ok(out)
}

fn tensor_names(bundle: &Path) -> Result<BTreeMap<TensorRole, String>> {
    let bundle = Bundle::open(bundle).map_err(|e| Error(e.message))?;
    bundle
        .manifest
        .artifacts
        .iter()
        .find(|a| a.format == Format::Safetensors && a.graph_input == GraphInput::TokenIds)
        .map(|a| a.tensor_names.clone())
        .ok_or_else(|| Error("the bundle has no raw encoder weights".into()))
}

#[allow(clippy::too_many_arguments)]
fn checkpoint(
    dir: &Path,
    spec: &turbo::manifest::HeadSpec,
    spec_json: &[u8],
    hidden: usize,
    head: &Parser<f32>,
    encoder: &[turbo_trainer::encoder::Named],
    epoch: usize,
    dev: f64,
) -> turbo::status::Result<()> {
    let err = |e: Error| turbo::status::invalid(e.0);
    let at = [(CHECKPOINT_EPOCH, (epoch + 1).to_string()), (CHECKPOINT_DEV, dev.to_string())];
    replace(dir, "head.json", spec_json).map_err(err)?;
    let head = ud_heads::named(UdHead::Deps, spec, hidden, head.tensors());
    replace(dir, "head.safetensors", &safetensors::write_with(&head, &at)).map_err(err)?;
    replace(dir, "encoder.safetensors", &safetensors::write_with(encoder, &at)).map_err(err)?;
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "cuda-shim"))]
#[allow(clippy::too_many_arguments)]
fn fit_cuda(
    base: Encoder<f32>,
    spec: &turbo::manifest::HeadSpec,
    train: &[TunedParseExample],
    dev: &[TunedParseExample],
    c: &turbo_trainer::Config,
    dir: &Path,
    spec_json: &[u8],
    names: &BTreeMap<TensorRole, String>,
) -> Result<(Encoder<f32>, Parser<f32>, turbo_trainer::Report)> {
    use turbo_trainer::encoder::cuda::CudaEncoder;
    use turbo_trainer::fit_tuned_cuda_parser_with;

    let hidden = base.geometry().hidden;
    let cuda = CudaEncoder::new(&base, 0).map_err(|e| Error(e.message))?;
    let (model, report) = fit_tuned_cuda_parser_with(cuda, spec, train, dev, c, &mut |m, epoch, dev| {
        let encoder = m.encoder()?.to_cpu()?.named(names)?;
        checkpoint(dir, spec, spec_json, hidden, m.head(), &encoder, epoch, dev)
    })
    .map_err(|e| Error(e.message))?;
    let (encoder, head) = model.into_parts().map_err(|e| Error(e.message))?;
    Ok((encoder.to_cpu().map_err(|e| Error(e.message))?, head, report))
}

#[cfg(not(any(feature = "cuda", feature = "cuda-shim")))]
#[allow(clippy::too_many_arguments)]
fn fit_cuda(
    _: Encoder<f32>,
    _: &turbo::manifest::HeadSpec,
    _: &[TunedParseExample],
    _: &[TunedParseExample],
    _: &turbo_trainer::Config,
    _: &Path,
    _: &[u8],
    _: &BTreeMap<TensorRole, String>,
) -> Result<(Encoder<f32>, Parser<f32>, turbo_trainer::Report)> {
    Err(Error("this binary was built without the cuda feature".into()))
}
