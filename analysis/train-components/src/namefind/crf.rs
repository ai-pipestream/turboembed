//! CRF sequence training alongside the existing event-model training API.
use super::TokenNameFinderFactory;
use crate::package::Stamp;
use turbo_formats::NameSample;
use turbo_sequence::codec::Span;
use turbo_sequence::crf::CrfSequenceModel;
use turbo_sequence::featuregen::{AdditionalContext, Window};
use turbo_sequence::namefind::crf::{MODEL_ENTRY, MODEL_TYPE};
use turbo_train::ModelPackage;
use turbo_train::crf::{CrfTrainer, CrfTrainingSequence};

pub struct TrainedCrfNameModel {
    pub model: CrfSequenceModel,
    pub package: ModelPackage,
}
/// Extract observation-only features and constrained gold sequences.
/// Empty, overlapping and out-of-range annotated names are rejected.
pub fn crf_sequences(
    factory: &TokenNameFinderFactory,
    samples: &[NameSample<'_>],
) -> Result<Vec<CrfTrainingSequence>, String> {
    let mut context = factory.context_generator().map_err(|e| e.to_string())?;
    context.add_feature_generator(Box::new(Window::new(Box::new(AdditionalContext::new()), 8, 8)));
    samples
        .iter()
        .map(|sample| {
            let tokens: Vec<_> = sample.tokens().iter().map(|f| f.as_str()).collect();
            let mut spans = Vec::new();
            let mut end = 0;
            for name in sample.names() {
                let (a, b) = (name.start as usize, name.end as usize);
                if a >= b || b > tokens.len() || a < end {
                    return Err("invalid or overlapping gold entity".into());
                }
                end = b;
                spans.push(Span::new(a, b, name.kind));
            }
            Ok(CrfTrainingSequence {
                features: context.sequence_features(&tokens, &[]),
                outcomes: factory.codec().encode(&spans, tokens.len()),
            })
        })
        .collect()
}
/// Train a separate CRF model artifact. The package uses the existing zip
/// container and feature descriptor, with an explicit Turbo CRF format ID;
/// it does not claim compatibility with OpenNLP's event-model serializer.
pub fn train_crf_name_model(
    factory: &TokenNameFinderFactory,
    samples: &[NameSample<'_>],
    trainer: &CrfTrainer,
    stamp: Stamp<'_>,
) -> Result<TrainedCrfNameModel, String> {
    if factory.codec() != trainer.codec {
        return Err("factory/trainer codec mismatch".into());
    }
    let sequences = crf_sequences(factory, samples)?;
    let model = trainer.train(&sequences)?;
    let mut manifest = stamp.manifest(super::COMPONENT_NAME, factory.language());
    manifest.set("Model-Type", MODEL_TYPE);
    manifest.set("Algorithm", "CRF");
    for (k, v) in factory.manifest_entries() {
        manifest.set(&k, &v);
    }
    let mut package = ModelPackage::new(manifest);
    package.add_entry(MODEL_ENTRY, model.to_bytes());
    package.add_entry(super::DESCRIPTOR_ENTRY, factory.descriptor().to_vec());
    Ok(TrainedCrfNameModel { model, package })
}
