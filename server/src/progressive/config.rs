use turbo_gliner::{Decode, ExtractionSchema};
use turbo_relation::{RelationLabel, ZeroShotRelationConfig};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::{OpenRelationSchema, OpenSpanSchema};

pub(super) fn open_span(schema: Option<&OpenSpanSchema>) -> Result<Option<ExtractionSchema>, Failure> {
    schema
        .map(|schema| {
            ExtractionSchema::with_prompts(
                schema.labels.clone(),
                if schema.prompts.is_empty() { schema.labels.clone() } else { schema.prompts.clone() },
                schema.threshold.unwrap_or(0.5),
                if schema.nested { Decode::Nested } else { Decode::Flat },
            )
            .map_err(|error| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("open_span: {error}")))
        })
        .transpose()
}

pub(super) fn relations(schema: Option<&OpenRelationSchema>) -> Option<ZeroShotRelationConfig> {
    schema.map(|schema| {
        let labels = schema
            .labels
            .iter()
            .map(|label| RelationLabel {
                name: label.name.clone(),
                threshold: label.threshold.map(f64::from),
                symmetric: label.symmetric,
                domain: (!label.domain.is_empty()).then(|| label.domain.clone()),
                range: (!label.range.is_empty()).then(|| label.range.clone()),
            })
            .collect();
        ZeroShotRelationConfig {
            threshold: schema.threshold.unwrap_or(0.5).into(),
            within_sentence: schema.within_sentence,
            ..ZeroShotRelationConfig::new(labels)
        }
    })
}
