use std::path::Path;

use turbo::bundle::Bundle;
use turbo::status::Result;
use turbo_analysis_embed::{EncoderOptions, Precision};
use turbo_tagger::WordEncoder;
use turbo_trainer::{Config, Example, Report, Tagger};

use super::{Prepared, Sentence, digest, field, head, invalid};

/// Hashes of data, membership, complete training settings and encoder
/// manifest. The manifest identity includes its pinned tokenizer/weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunIdentity {
    pub corpus_sha256: String,
    pub membership_sha256: String,
    pub config_sha256: String,
    pub encoder_sha256: String,
}

/// Preparation-only in-memory result. The MASC entity-identity blocker
/// documented on the module must be fixed before real-data training.
/// No claim of released accuracy or production packaging.
pub struct Run {
    pub model: Tagger<f32>,
    pub report: Report,
    pub identity: RunIdentity,
}

/// Train on CPU with one encoder thread and Exact precision. Only train
/// and development examples are encoded: the test set never selects weights.
/// This preparation entry point is not a production or release command;
/// see the module's MASC reader blocker before using real annotations.
pub fn fit_cpu(data: &Prepared, bundle: &Path, config: &Config) -> Result<Run> {
    let spec = head();
    let manifest = Bundle::open(bundle)?.manifest_sha256;
    let options = EncoderOptions { precision: Precision::Exact, cpu_threads: 1, ..EncoderOptions::default() };
    let mut encoder = WordEncoder::open(bundle, &options)?;
    let mut settings = b"native-masc-ner-v1:cpu:exact:threads1:dev=train[8mod10]".to_vec();
    field(&mut settings, &serde_json::to_vec(&spec).map_err(invalid)?);
    // Exhaustive on purpose: adding a trainer setting must also update
    // this identity, so new optimizer or stopping controls cannot vanish.
    let Config { seed, epochs, batch, lr, weight_decay, clip, warmup, patience } = *config;
    for n in [seed, epochs as u64, batch as u64, patience as u64] {
        settings.extend_from_slice(&n.to_le_bytes());
    }
    for n in [lr, weight_decay, clip, warmup] {
        settings.extend_from_slice(&n.to_bits().to_le_bytes());
    }
    let train = examples(&data.train, &mut encoder)?;
    let dev = examples(&data.dev, &mut encoder)?;
    let (model, report) = turbo_trainer::fit(&spec, encoder.hidden(), &train, &dev, config)?;
    if report.train_loss.iter().chain(&report.dev).any(|x| !x.is_finite())
        || model.params().iter().flatten().any(|x| !x.is_finite())
    {
        return Err(invalid("training produced a nonfinite result"));
    }
    if Bundle::open(bundle)?.manifest_sha256 != manifest {
        return Err(invalid("encoder manifest changed during training"));
    }
    Ok(Run {
        model,
        report,
        identity: RunIdentity {
            corpus_sha256: data.corpus_sha256().into(),
            membership_sha256: data.membership_sha256().into(),
            config_sha256: digest(&settings),
            encoder_sha256: manifest,
        },
    })
}

fn examples(sentences: &[Sentence], encoder: &mut WordEncoder) -> Result<Vec<Example>> {
    sentences
        .iter()
        .map(|sentence| {
            let sequence = vec![(0..sentence.labels().len()).collect()];
            let mut rows = encoder
                .token_features(sentence.view(), sentence.tokens(), &sequence, None, head().word_pooling)
                .map_err(invalid)?;
            let features = rows.pop().ok_or_else(|| invalid("encoder returned no sequence"))?;
            if features.iter().any(|x| !x.is_finite()) {
                return Err(invalid("encoder features are not finite"));
            }
            Ok(Example { features, labels: sentence.labels().to_vec() })
        })
        .collect()
}
