use super::{Model, Result, write_new};
use std::{fs, path::Path, time::Instant};
use turbo_analysis::ViewId;
use turbo_analysis_layers::Words;
use turbo_depparse::{
    Activation, DependencyModel, DependencySample, FeatureLayout, FeedforwardDependencyTrainer, FeedforwardSettings,
};
use turbo_formats::conllu::{ConlluDependencySample, Tagset, words::ConlluWordReader};
use turbo_train::TrainingParameters;
use turbo_unicode::WhitespaceMode;

fn checked_keys(params: &TrainingParameters, allowed: &[&str]) -> Result<()> {
    for (key, _) in params.iter() {
        if !allowed.iter().any(|name| name.eq_ignore_ascii_case(key)) {
            return Err(format!("unsupported training parameter: {key}").into());
        }
    }
    Ok(())
}
fn required_count(params: &TrainingParameters, key: &str) -> Result<usize> {
    let value = params.get_int(key)?.ok_or_else(|| format!("explicit {key} required"))?;
    if value <= 0 {
        return Err(format!("{key} must be positive").into());
    }
    Ok(value as usize)
}
fn settings(params: &TrainingParameters) -> Result<FeedforwardSettings> {
    checked_keys(
        params,
        &[
            "Epochs",
            "EmbeddingSize",
            "HiddenSize",
            "BatchSize",
            "LearningRate",
            "L2",
            "Dropout",
            "WordCutoff",
            "Seed",
            "Layout",
            "Activation",
        ],
    )?;
    let mut settings = FeedforwardSettings { epochs: required_count(params, "Epochs")?, ..Default::default() };
    for (key, value) in [
        ("EmbeddingSize", &mut settings.embedding_size),
        ("HiddenSize", &mut settings.hidden_size),
        ("BatchSize", &mut settings.batch_size),
        ("WordCutoff", &mut settings.word_cutoff),
    ] {
        if let Some(parsed) = params.get_int(key)? {
            *value = usize::try_from(parsed).map_err(|_| format!("negative {key}"))?;
        }
    }
    settings.learning_rate = params.get_double("LearningRate")?.unwrap_or(settings.learning_rate);
    settings.l2 = params.get_double("L2")?.unwrap_or(settings.l2);
    settings.dropout = params.get_double("Dropout")?.unwrap_or(settings.dropout);
    if let Some(seed) = params.get_str("Seed")? {
        settings.seed = seed.parse()?;
    }
    settings.layout = match params.get_str("Layout")?.unwrap_or("ChenManning18") {
        "ChenManning18" => FeatureLayout::ChenManning18,
        "OpenNlp14" => FeatureLayout::OpenNlp14,
        _ => return Err("Layout must be ChenManning18 or OpenNlp14".into()),
    };
    settings.activation = match params.get_str("Activation")?.unwrap_or("cube") {
        "cube" => Activation::Cube,
        "relu" => Activation::Relu,
        _ => return Err("Activation must be cube or relu".into()),
    };
    settings.check()?;
    Ok(settings)
}

pub(super) fn run(args: &[String]) -> Result<()> {
    if Path::new(&args[5]).exists() {
        return Err("output already exists".into());
    }
    if args[2].is_empty() || args[2].contains(char::is_whitespace) {
        return Err("a nonempty language identifier without whitespace is required".into());
    }
    let params_bytes = fs::read(&args[4])?;
    let params = TrainingParameters::load(&params_bytes)?;
    let neural = match args[1].as_str() {
        "feedforward" => Some(settings(&params)?),
        "linear" => {
            checked_keys(&params, &["Algorithm", "Iterations", "Cutoff", "Threads"])?;
            if params.get_str("Algorithm")? != Some("MAXENT") {
                return Err("linear CLI training requires Algorithm=MAXENT".into());
            }
            required_count(&params, "Iterations")?;
            required_count(&params, "Threads")?;
            if params.get_int("Cutoff")?.is_none_or(|n| n < 0) {
                return Err("explicit nonnegative Cutoff required".into());
            }
            None
        }
        _ => return Err("training kind must be linear or feedforward".into()),
    };
    let input = fs::read_to_string(&args[3])?;
    let mut samples = Vec::new();
    let mode = args.get(6).map(String::as_str).unwrap_or("--pseudo-projective");
    let projective_only = mode == "--projective-only";
    let mut crossing = 0;
    let (mut words, mut skipped, mut skipped_words) = (0, 0, 0);
    for (index, sentence) in ConlluWordReader::new(&input)?.enumerate() {
        let sentence = sentence?;
        if sentence.words.iter().any(|word| word.upos.as_str() == "_") {
            return Err(format!("sentence {} requires UPOS", index + 1).into());
        }
        let span = sentence.source;
        let sample = ConlluDependencySample::read_sentence(
            &input[span.start as usize..span.end as usize],
            Tagset::U,
            WhitespaceMode::Unicode,
            ViewId::SOURCE,
        )
        .map_err(|error| format!("sentence {}: {error}", index + 1))?;
        let sample = DependencySample::<Words>::from_conllu(sample)?;
        words += sample.tokens().len();
        if !sample.graph().is_projective() {
            crossing += 1;
            if projective_only {
                skipped += 1;
                skipped_words += sample.tokens().len();
                continue;
            }
        }
        samples.push(sample);
    }
    eprintln!(
        "sentences={} words={words} excluded_nonprojective_sentences={skipped} excluded_nonprojective_words={skipped_words}",
        samples.len() + skipped
    );
    eprintln!("trained_sentences={} trained_words={}", samples.len(), words - skipped_words);
    eprintln!("included_nonprojective_sentences={} training_mode={}", crossing - skipped, &mode[2..]);
    if samples.is_empty() {
        return Err("no projective training sentences".into());
    }
    eprintln!("language={} kind={} neural_settings={neural:?}", args[2], args[1]);
    let start = Instant::now();
    let model = match neural {
        Some(settings) => Model::Feedforward(match mode {
            "--projective-only" => FeedforwardDependencyTrainer::train_projective(&samples, &settings)?,
            "--swap" => FeedforwardDependencyTrainer::train_swap(&samples, &settings)?,
            _ => FeedforwardDependencyTrainer::train(&samples, &settings)?,
        }),
        None => Model::Linear(match mode {
            "--projective-only" => DependencyModel::train_projective(&args[2], &samples, &params)?,
            "--swap" => DependencyModel::train_swap(&args[2], &samples, &params)?,
            _ => DependencyModel::train(&args[2], &samples, &params)?,
        }),
    };
    let bytes = model.bytes()?;
    write_new(&args[5], &bytes)?;
    println!(
        "training_seconds={:.6} corpus_sha256={} params_sha256={} output_sha256={}",
        start.elapsed().as_secs_f64(),
        turbo_maxent::sha256(input.as_bytes()).to_hex(),
        turbo_maxent::sha256(&params_bytes).to_hex(),
        turbo_maxent::sha256(&bytes).to_hex()
    );
    Ok(())
}
