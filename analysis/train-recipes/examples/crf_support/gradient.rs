use std::collections::BTreeMap;
use turbo_formats::NameSample;
use turbo_sequence::crf::CrfConstraints;
use turbo_sequence::namefind::NameCodec;
use turbo_train::crf::{CrfObjective, IndexedSequence};
use turbo_train_components::namefind::{TokenNameFinderFactory, crf_sequences};

pub fn check(samples: &[NameSample<'_>]) -> Result<(), Box<dyn std::error::Error>> {
    let (index, sample) = samples
        .iter()
        .enumerate()
        .find(|(_, s)| !s.names().is_empty() && s.tokens().len() <= 12)
        .ok_or("fixture has no short named-entity sentence")?;
    for codec in [NameCodec::Bio, NameCodec::Bilou] {
        let factory = TokenNameFinderFactory::new("en", None, codec)?;
        let sequence = crf_sequences(&factory, std::slice::from_ref(sample))?.remove(0);
        let mut labels = vec!["other".to_owned()];
        for name in sample.names() {
            for suffix in
                if codec == NameCodec::Bio { &["start", "cont"][..] } else { &["start", "cont", "last", "unit"][..] }
            {
                labels.push(format!("{}-{suffix}", name.kind.unwrap_or("default")));
            }
        }
        labels.sort();
        labels.dedup();
        let mut features = BTreeMap::new();
        for (f, _) in sequence.features.iter().flatten() {
            let n = features.len();
            features.entry(f.clone()).or_insert(n);
        }
        let rows = sequence.features.iter().map(|r| r.iter().map(|(f, v)| (features[f], *v)).collect()).collect();
        let gold = sequence.outcomes.iter().map(|s| labels.iter().position(|l| l == s).unwrap()).collect();
        let objective = CrfObjective::new(
            features.len(),
            CrfConstraints::new(&labels, codec)?,
            vec![IndexedSequence { features: rows, outcomes: gold }],
            0.17,
        )?;
        let mut x: Vec<_> = (0..objective.dimension()).map(|i| (i as f64 * 0.13).sin() * 0.05).collect();
        let mut g = vec![0.0; x.len()];
        objective.value_gradient(&x, &mut g)?;
        let mut scratch = vec![0.0; x.len()];
        let mut error = 0.0f64;
        let epsilon = 1e-5;
        for i in 0..x.len() {
            let old = x[i];
            x[i] = old + epsilon;
            let high = objective.value_gradient(&x, &mut scratch)?;
            x[i] = old - epsilon;
            let low = objective.value_gradient(&x, &mut scratch)?;
            x[i] = old;
            error = error.max(((high - low) / (2.0 * epsilon) - g[i]).abs());
        }
        println!(
            "MASC gradient codec={codec:?} sample={index} tokens={} parameters={} max_absolute_error={error:.12e}",
            sample.tokens().len(),
            x.len()
        );
        if error > 1e-5 {
            return Err("real-sequence finite-difference gradient failed".into());
        }
    }
    Ok(())
}
