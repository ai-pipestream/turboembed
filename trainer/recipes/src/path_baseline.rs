//! `baseline --corpus re-docred --dependency-path`: the dependency-path
//! rule row (#1825, the row the #1734 receipt lists as missing). The
//! documents' own sentences are parsed by our dependency head
//! (`--parser`, a `train-parse` run directory, over `--bundle`), the
//! rule of [`redocred_path`] mined from the training split and applied
//! to the test split, and the receipt's `baseline` is the rule's score.
//! The receipt's `encoder` and `head` name what produced the parses;
//! `outputs` holds the rule's predictions, digested.

use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo::head::BiaffineParser;
use turbo::manifest::HeadSpec;
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::cli::{Args, OutDir, commit, encoder_args};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::replace;
use turbo_trainer_recipes::{Error, Result, redocred, redocred_eval, redocred_path};

use super::{encoder_record, finish};

pub(super) fn dependency_path(
    argv: &[String],
    corpus_id: &str,
    dir: &Path,
    unrecorded: bool,
    mut a: Args,
) -> Result<()> {
    if corpus_id != redocred::ID {
        return Err(Error(format!("{corpus_id}: --dependency-path is re-docred's baseline")));
    }
    let parser_dir = PathBuf::from(a.required("--parser")?);
    let bundle = PathBuf::from(a.required("--bundle")?);
    let (device, precision, _) = encoder_args(&mut a)?;
    let out = OutDir::claim(&a.required("--out")?)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;

    let t = Instant::now();
    let c = redocred::read(dir, unrecorded)?;
    let read = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let spec_bytes =
        std::fs::read(parser_dir.join("head.json")).map_err(|e| Error(format!("{}: {e}", parser_dir.display())))?;
    let spec: HeadSpec = serde_json::from_slice(&spec_bytes).map_err(|e| Error(format!("head.json: {e}")))?;
    if spec.kind != turbo::manifest::HeadKind::BiaffineParser {
        return Err(Error(format!("{}: {:?} is not a parser head", parser_dir.display(), spec.kind)));
    }
    let weights = std::fs::read(parser_dir.join("head.safetensors"))
        .map_err(|e| Error(format!("{}: {e}", parser_dir.display())))?;
    let tensors = turbo::safetensors::File::parse("head.safetensors", &weights).map_err(|e| Error(e.message))?;
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let get = |s: &str| tensors.tensor(&format!("{}.{s}", spec.tensor_prefix)).map(|t| t.f32s());
    let parser = BiaffineParser::from_tensors(&spec, enc.hidden(), &get).map_err(|e| Error(e.message))?;

    let train_parsed = parse_documents(&mut enc, &parser, &c.splits.train)?;
    let test_parsed = parse_documents(&mut enc, &parser, &c.splits.test)?;
    let features_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let table = redocred_path::mine(&c.splits.train, &train_parsed);
    let predicted = redocred_path::predict(&c.splits.test, &test_parsed, &table);
    let baseline = redocred_eval::evaluate(&c.splits.test, &predicted);
    let eval_s = t.elapsed().as_secs_f64();

    let predictions = serde_json::to_vec_pretty(
        &c.splits
            .test
            .iter()
            .zip(&predicted)
            .map(|(d, rs)| {
                serde_json::json!({
                    "title": d.title,
                    "relations": rs.iter().map(|(h, t, r)| serde_json::json!([h, t, r])).collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>(),
    )
    .expect("predictions serialize");
    let outputs = vec![replace(&out.path, "predictions.json", &predictions)?];

    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        host: host(),
        unix_time: unix_time(),
        corpus: c.record.clone(),
        encoder: Some(encoder),
        head: Some(spec),
        peer: None,
        config: None,
        training: None,
        eval: EvalRecord {
            split: "test".into(),
            model: None,
            baseline,
            dictionary: None,
            open_span: None,
            comparison: None,
            uas: None,
            rules: None,
        },
        timing_s: Timing { read, features: features_s, eval: eval_s, ..Timing::default() },
        outputs,
    };
    println!(
        "re-docred: dependency-path rule, {} paths mined, test micro F1 {:.4}; {}/receipt.json",
        table.len(),
        receipt.eval.baseline.score(),
        out.path.display(),
    );
    finish(out, receipt)
}

/// Each document's sentences parsed by `parser` over the encoder's
/// features, the head's own pooling.
fn parse_documents(
    enc: &mut WordEncoder,
    parser: &BiaffineParser,
    docs: &[redocred::Document],
) -> Result<Vec<redocred_path::Parsed>> {
    let pooling = parser.spec().word_pooling;
    let (mut arcs, mut heads, mut scores, mut rels, mut w) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Default::default());
    let mut out = Vec::with_capacity(docs.len());
    for d in docs {
        let mut at = 0;
        let mut sentences: Vec<Vec<&str>> = Vec::with_capacity(d.sentences.len());
        for &n in &d.sentences {
            sentences.push(d.words[at..at + n].iter().map(String::as_str).collect());
            at += n;
        }
        let features = enc.features(&sentences, pooling).map_err(|e| Error(e.message))?;
        let mut parsed = Vec::with_capacity(sentences.len());
        for (f, words) in features.iter().zip(&sentences) {
            let n = words.len();
            let ids: Vec<u32> = (0..n as u32).collect();
            parser.arcs(f, &ids, &mut arcs, &mut w).map_err(|e| Error(e.message))?;
            parser.decode(&arcs, n, &mut heads).map_err(|e| Error(e.message))?;
            parser.relations(&heads, &mut scores, &w).map_err(|e| Error(e.message))?;
            parser.relation_labels(&scores, n, &mut rels).map_err(|e| Error(e.message))?;
            parsed.push(heads.iter().zip(&rels).map(|(&h, &r)| (h, parser.labels()[r as usize].clone())).collect());
        }
        out.push(parsed);
    }
    Ok(out)
}
