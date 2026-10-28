//! Block 5 opt-in progressive steps: SaT sentences, PII scanning and
//! masking, and the numeric annotators.
//!
//! These steps are never part of the default profile. A request names
//! them explicitly beside the existing steps. SaT writes the same
//! `Sentences` layer as the rule sentence detector, so one pipeline picks
//! one sentence tier. PII masking writes the named `masked` view and
//! requires the PII layer from the scanning step.

use std::path::Path;

use turbo_analysis::analyzer::Builder;
use turbo_numeric::money::MoneyExtractor;
use turbo_numeric::quantity::QuantityExtractor;
use turbo_numeric::temporal::TemporalExtractor;
use turbo_numeric::{DocumentDateSelector, NumberScanner};
use turbo_pii::rewrite::MaskedView;
use turbo_pii::{Pack, PiiDetector};
use turbo_sat::{SatAnnotator, SatOptions};

use super::EmbedContext;
use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT};

/// Whether `name` selects the SaT sentence tier.
pub(super) fn is_sat_step(name: &str) -> bool {
    matches!(name, "turbo-sat:sat" | "turbo:sat" | "sat")
}

/// Adds one block 5 step to `builder` when `name` names one.
///
/// Returns the builder and whether the step was handled here.
pub(super) fn add_step(
    builder: Builder,
    name: &str,
    embed_cx: Option<&EmbedContext>,
) -> Result<(Builder, bool), Failure> {
    let builder = match name {
        _ if is_sat_step(name) => {
            let cx = embed_cx
                .ok_or_else(|| Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, "turbo:sat needs a loaded SaT bundle"))?;
            let annotator = SatAnnotator::open(
                Path::new(&cx.loaded.bundle),
                &SatOptions { encoder: cx.encoder_options(), ..SatOptions::default() },
            )
            .map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("turbo:sat: {e}")))?;
            builder.add(annotator)
        }
        "turbo-pii:pii" | "turbo:pii" | "pii" => builder.add(PiiDetector::pack(Pack::AllStructured)),
        "turbo-pii:masked" | "turbo:pii-masked" | "pii-masked" => {
            let masked = MaskedView::by_kind('*')
                .map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("turbo-pii:masked: {e}")))?;
            builder.step(masked.bound())
        }
        "turbo-numeric:numbers" | "turbo:numbers" | "numbers" => builder.add(NumberScanner::default()),
        "turbo-numeric:money" | "turbo:money" | "money" => builder.add(MoneyExtractor::default()),
        "turbo-numeric:quantities" | "turbo:quantities" | "quantities" => builder.add(QuantityExtractor::default()),
        "turbo-numeric:temporal" | "turbo:temporal" | "temporal" => builder.add(TemporalExtractor::new()),
        "turbo-numeric:document-date" | "turbo:document-date" | "document-date" => builder.add(DocumentDateSelector),
        _ => return Ok((builder, false)),
    };
    Ok((builder, true))
}
