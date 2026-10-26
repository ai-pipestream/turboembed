//! Trained GoEmotions head and evaluation for `EmotionAnnotator` (#926).
//!
//! Embeds the GoEmotions `train.tsv`, `dev.tsv`, and `test.tsv` splits with
//! one encoder bundle, fits an `EmotionVectorHead` to the train split's 28
//! labels, selects both a uniform threshold (by micro F1) and per-label
//! thresholds (by per-label F1) on the dev split for the lexicon, learned,
//! and hybrid stages, scores the test split under each threshold scheme,
//! writes the bound `.tevh` head file, and records the full metrics (macro
//! F1, micro F1, and per-label precision/recall/F1) to a JSON file.
//!
//! ```text
//! cargo run --release -p turbo-analysis-embed --example goemotions_eval -- \
//!     --bundle <bundle dir> --device cuda --goemotions <dir> \
//!     --cache <dir> --head-out <head.tevh> --out <file.json>
//! ```

#[path = "../incivility_bias/embed.rs"]
mod embed;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde_json::{Value, json};
use turbo_analysis::{Ix, IxRange, Span, Spans, ViewId};
use turbo_analysis_embed::{Device, EncoderOptions};
use turbo_analysis_layers::{
    EMOTION_DIMENSIONS, EmotionDimension, EmotionItem, EmotionLayer, EmotionScores, EmotionTarget, Sentences, Tokens,
};
use turbo_doccat::sentiment::token_key;
use turbo_doccat::{
    EmotionAnnotator, EmotionFitOptions, EmotionMetrics, EmotionStage, EmotionVectorHead, EmotionVectorScorer,
    GoldEmotionUnit, evaluate_emotion_corpus, evaluate_emotion_corpus_with_thresholds,
};
use turbo_zoo::{Sha256, Sha512};

/// Candidate thresholds swept on the validation split.
const THRESHOLDS: [f64; 19] =
    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

const STAGES: [(&str, EmotionStage); 3] =
    [("lexicon", EmotionStage::Lexicon), ("learned", EmotionStage::Learned), ("hybrid", EmotionStage::Hybrid)];

struct Args {
    bundle: PathBuf,
    device: Device,
    goemotions: PathBuf,
    cache: PathBuf,
    head_out: Option<PathBuf>,
    out: PathBuf,
    fit: EmotionFitOptions,
}

fn args() -> Result<Args, String> {
    let mut a = std::env::args().skip(1);
    let (mut bundle, mut device, mut goemotions, mut cache, mut head_out, mut out) =
        (None, Device::Cpu, None, None, None, None);
    let mut fit = EmotionFitOptions { epochs: 500, learning_rate: 16.0, l2: 1e-4 };
    while let Some(flag) = a.next() {
        let mut value = || a.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(value()?)),
            "--device" => {
                device = match value()?.as_str() {
                    "cpu" => Device::Cpu,
                    d => Device::Backend(d.into()),
                }
            }
            "--goemotions" => goemotions = Some(PathBuf::from(value()?)),
            "--cache" => cache = Some(PathBuf::from(value()?)),
            "--head-out" => head_out = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--epochs" => fit.epochs = value()?.parse().map_err(|e| format!("--epochs: {e}"))?,
            "--learning-rate" => fit.learning_rate = value()?.parse().map_err(|e| format!("--learning-rate: {e}"))?,
            "--l2" => fit.l2 = value()?.parse().map_err(|e| format!("--l2: {e}"))?,
            f => return Err(format!("unknown flag {f}")),
        }
    }
    let need = |v: Option<PathBuf>, f: &str| v.ok_or_else(|| format!("{f} is required"));
    Ok(Args {
        bundle: need(bundle, "--bundle")?,
        device,
        goemotions: need(goemotions, "--goemotions")?,
        cache: need(cache, "--cache")?,
        head_out,
        out: need(out, "--out")?,
        fit,
    })
}

struct Row {
    text: String,
    labels: Vec<EmotionDimension>,
    targets: EmotionScores,
}

fn read_split(path: &Path) -> Result<Vec<Row>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 {
            return Err(format!("{}:{}: expected tab-separated text and labels", path.display(), line_idx + 1));
        }
        let text = parts[0].to_owned();
        let mut labels = Vec::new();
        let mut targets = EmotionScores::default();
        for tok in parts[1].split(',') {
            let idx: usize = tok
                .trim()
                .parse()
                .map_err(|e| format!("{}:{}: bad label index {tok:?}: {e}", path.display(), line_idx + 1))?;
            let dim = *EmotionDimension::ALL
                .get(idx)
                .ok_or_else(|| format!("{}:{}: label index {idx} out of 0..28", path.display(), line_idx + 1))?;
            if !labels.contains(&dim) {
                labels.push(dim);
            }
            targets.set(dim, 1.0);
        }
        rows.push(Row { text, labels, targets });
    }
    Ok(rows)
}

fn verify_emotions_txt(dir: &Path) -> Result<(), String> {
    let path = dir.join("emotions.txt");
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let names: Vec<&str> = text.lines().map(str::trim).filter(|s| !s.is_empty()).collect();
    if names.len() != EMOTION_DIMENSIONS {
        return Err(format!("emotions.txt has {} lines, expected {EMOTION_DIMENSIONS}", names.len()));
    }
    for (i, (&got, want)) in names.iter().zip(EmotionDimension::ALL).enumerate() {
        if got != want.name() {
            return Err(format!("emotions.txt line {i} is {got:?}, expected {:?}", want.name()));
        }
    }
    Ok(())
}

struct Prepared {
    keys: Vec<Vec<String>>,
    gold: Vec<GoldEmotionUnit>,
    tokens: Tokens,
    sentences: Sentences,
}

fn prepare(rows: &[Row]) -> Prepared {
    let mut all_keys = Vec::with_capacity(rows.len());
    let mut gold = Vec::with_capacity(rows.len());
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    let mut tok_sent = Vec::new();
    let mut byte_cursor = 0u32;
    for (i, r) in rows.iter().enumerate() {
        let mut raw_spans = Vec::new();
        turbo_tokenize::simple_spans(&r.text, turbo_tokenize::TokenizerOptions::default(), &mut raw_spans);
        let keys: Vec<String> =
            raw_spans.iter().map(|s| token_key(&r.text[s.start as usize..s.end as usize])).collect();
        let sent_start = byte_cursor;
        if keys.is_empty() {
            // Ensure at least one synthetic token span so evidence ranges `0..1` are valid.
            tok_spans.push(Span::new(byte_cursor, byte_cursor + 1));
            tok_sent.push(Ix::new(i as u32));
            byte_cursor += 1;
        } else {
            for _ in 0..keys.len() {
                tok_spans.push(Span::new(byte_cursor, byte_cursor + 1));
                tok_sent.push(Ix::new(i as u32));
                byte_cursor += 1;
            }
        }
        sent_spans.push(Span::new(sent_start, byte_cursor));
        all_keys.push(keys);
        gold.push(GoldEmotionUnit { labels: r.labels.clone() });
    }
    Prepared {
        keys: all_keys,
        gold,
        tokens: Tokens { spans: tok_spans, sentence: Some(tok_sent) },
        sentences: Sentences { spans: sent_spans },
    }
}

fn score_split(
    annotator: &EmotionAnnotator,
    p: &Prepared,
    vectors: &[f32],
    dim: usize,
) -> Result<EmotionLayer, String> {
    let mut layer = EmotionLayer::new();
    let mut tok_offset = 0usize;
    for (i, keys) in p.keys.iter().enumerate() {
        let v = &vectors[i * dim..(i + 1) * dim];
        let (scores, local_ev) = annotator.score_unit(keys, 0, keys.len(), Some(v))?;
        let step = keys.len().max(1);
        let shifted: Vec<_> = local_ev
            .into_iter()
            .map(|(d, r)| {
                (d, IxRange::new(r.start.get() as u32 + tok_offset as u32, r.end.get() as u32 + tok_offset as u32))
            })
            .collect();
        layer.try_push(
            &p.tokens,
            Some(&p.sentences),
            None,
            EmotionItem { target: EmotionTarget::Sentence(Ix::new(i as u32)), scores, evidence: &shifted },
        )?;
        tok_offset += step;
    }
    Ok(layer)
}

fn tune_thresholds(
    annotator: &EmotionAnnotator,
    p: &Prepared,
    vectors: &[f32],
    dim: usize,
) -> Result<(f64, f64, [f64; EMOTION_DIMENSIONS], f64, f64), String> {
    let base_layer = score_split(annotator, p, vectors, dim)?;
    let mut best_uniform = (0.5, f64::NEG_INFINITY);
    let mut best_per_label = [0.5f64; EMOTION_DIMENSIONS];
    let mut best_label_f1 = [f64::NEG_INFINITY; EMOTION_DIMENSIONS];

    for t in THRESHOLDS {
        let m = evaluate_emotion_corpus(&p.gold, &base_layer, t)?;
        if m.micro_f1 > best_uniform.1 {
            best_uniform = (t, m.micro_f1);
        }
        for (idx, &(_, lm)) in m.per_label.iter().enumerate() {
            if lm.f1 > best_label_f1[idx] {
                best_label_f1[idx] = lm.f1;
                best_per_label[idx] = t;
            }
        }
    }
    let tuned_m = evaluate_emotion_corpus_with_thresholds(&p.gold, &base_layer, &best_per_label)?;
    Ok((best_uniform.0, best_uniform.1, best_per_label, tuned_m.macro_f1, tuned_m.micro_f1))
}

fn metrics_json(m: &EmotionMetrics) -> Value {
    let labels: Vec<Value> = m
        .per_label
        .iter()
        .map(|(d, x)| {
            json!({
                "label": d.name(),
                "tp": x.tp,
                "fp": x.fp,
                "fn": x.fn_count,
                "tn": x.tn,
                "precision": x.precision,
                "recall": x.recall,
                "f1": x.f1,
            })
        })
        .collect();
    json!({
        "per_label": labels,
        "macro": {
            "precision": m.macro_precision,
            "recall": m.macro_recall,
            "f1": m.macro_f1,
        },
        "micro": {
            "precision": m.micro_precision,
            "recall": m.micro_recall,
            "f1": m.micro_f1,
        },
        "active": m.active,
        "active_with_evidence": m.active_with_evidence,
    })
}

fn run() -> Result<(), String> {
    let a = args()?;
    verify_emotions_txt(&a.goemotions)?;
    eprintln!("reading GoEmotions splits");
    let train = read_split(&a.goemotions.join("train.tsv"))?;
    let dev = read_split(&a.goemotions.join("dev.tsv"))?;
    let test = read_split(&a.goemotions.join("test.tsv"))?;
    eprintln!("{} train, {} dev, {} test comments", train.len(), dev.len(), test.len());

    let mut embedder =
        embed::Embedder::open(&a.bundle, &EncoderOptions { device: a.device.clone(), ..Default::default() })?;
    let dim = embedder.dim;
    let mut vectors = |rows: &[Row], name: &str| {
        eprintln!("embedding {name}");
        let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
        embedder.embed(&texts, &a.cache)
    };
    let (train_v, dev_v, test_v) = (vectors(&train, "train")?, vectors(&dev, "dev")?, vectors(&test, "test")?);

    eprintln!("fitting EmotionVectorHead on {} comments, {:?}", train.len(), a.fit);
    let started = Instant::now();
    let vec_rows: Vec<&[f32]> = train_v.chunks_exact(dim).collect();
    let targets: Vec<EmotionScores> = train.iter().map(|r| r.targets).collect();
    let head = EmotionVectorHead::fit(&vec_rows, &targets, &a.fit)
        .map(|h| h.with_model(embedder.model.clone()))
        .map_err(|e| e.to_string())?;
    let fit_seconds = started.elapsed().as_secs_f64();
    let head_bytes = head.to_bytes();
    let head_sha256 = Sha256::of(&head_bytes).to_string();
    let head_sha512 = Sha512::of(&head_bytes).to_string();
    if let Some(path) = &a.head_out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        fs::write(path, &head_bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    let scorer: Arc<dyn EmotionVectorScorer> = Arc::new(head);

    let dev_p = prepare(&dev);
    let test_p = prepare(&test);
    let mut tuned_json = serde_json::Map::new();
    let mut test_json = serde_json::Map::new();

    for (stage_name, stage) in STAGES {
        let base = EmotionAnnotator::new().with_vector_scorer(Arc::clone(&scorer)).with_stage(stage);
        eprintln!("tuning thresholds on dev for {stage_name}");
        let (uni_t, dev_uni_micro, per_label_t, dev_pl_macro, dev_pl_micro) =
            tune_thresholds(&base, &dev_p, &dev_v, dim)?;
        let per_label_map: serde_json::Map<String, Value> = EmotionDimension::ALL
            .into_iter()
            .enumerate()
            .map(|(i, d)| (d.name().to_owned(), json!(per_label_t[i])))
            .collect();
        tuned_json.insert(
            stage_name.into(),
            json!({
                "uniform_threshold": uni_t,
                "dev_uniform_micro_f1": dev_uni_micro,
                "per_label_thresholds": per_label_map,
                "dev_per_label_macro_f1": dev_pl_macro,
                "dev_per_label_micro_f1": dev_pl_micro,
            }),
        );

        eprintln!("scoring test split for {stage_name}");
        let layer_default = score_split(&base, &test_p, &test_v, dim)?;
        let m_default = evaluate_emotion_corpus(&test_p.gold, &layer_default, 0.5)?;

        let layer_uni = score_split(&base.clone().with_threshold(uni_t), &test_p, &test_v, dim)?;
        let m_uni = evaluate_emotion_corpus(&test_p.gold, &layer_uni, uni_t)?;

        let layer_pl = score_split(&base.with_thresholds(per_label_t), &test_p, &test_v, dim)?;
        let m_pl = evaluate_emotion_corpus_with_thresholds(&test_p.gold, &layer_pl, &per_label_t)?;

        test_json.insert(
            stage_name.into(),
            json!({
                "default_threshold_0_5": metrics_json(&m_default),
                "tuned_uniform_threshold": {
                    "threshold": uni_t,
                    "metrics": metrics_json(&m_uni),
                },
                "tuned_per_label_thresholds": {
                    "thresholds": per_label_map,
                    "metrics": metrics_json(&m_pl),
                },
            }),
        );
    }

    let record = json!({
        "encoder": {
            "id": embedder.model.id,
            "revision": embedder.model.revision,
            "manifest_sha256": embedder.model.sha256,
            "dim": dim,
        },
        "head": {
            "train_comments": train.len(),
            "epochs": a.fit.epochs,
            "learning_rate": a.fit.learning_rate,
            "l2": a.fit.l2,
            "fit_seconds": fit_seconds,
            "size_bytes": head_bytes.len(),
            "sha256": head_sha256,
            "sha512": head_sha512,
        },
        "tuned_on_dev": {
            "dev_comments": dev.len(),
            "stages": tuned_json,
        },
        "test": {
            "test_comments": test.len(),
            "stages": test_json,
        },
    });
    if let Some(parent) = a.out.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?;
    fs::write(&a.out, text + "\n").map_err(|e| format!("{}: {e}", a.out.display()))
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("goemotions_eval: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
