//! The temporal runs: the extractor with the reference date taken from
//! each text's dateline (`dateline`) or fixed at the date the goldens were
//! made with (`fixed`), and the document date selected from its mentions.

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_layers::{Date, DocumentDate, Temporals};
use turbo_goldens::{Golden, Tolerance, matches_golden};
use turbo_numeric::{DocumentDateSelector, TemporalExtractor};

use super::{Outcome, Run, RustError, analyze, config, outcome};
use crate::resources::Unavailable;

/// The reference date of the `fixed` goldens (their `config reference`).
pub(crate) const REFERENCE: &str = "2026-03-15";

/// The `config reference` value of the `dateline` goldens.
const DATELINE: &str = "dateline";

pub(crate) struct Temporal {
    analyzer: Analyzer,
    reference: &'static str,
}

impl Temporal {
    pub(crate) fn open(run: &str) -> Result<Box<dyn Run>, Unavailable> {
        let broken = |what: String| Unavailable::Broken(what);
        let (extractor, reference) = match run {
            "dateline" => (TemporalExtractor::new().with_opennlp_parity(true), DATELINE),
            "fixed" => {
                let date = Date::parse(REFERENCE).ok_or_else(|| broken("the reference date".into()))?;
                (TemporalExtractor::with_reference(date).with_opennlp_parity(true), REFERENCE)
            }
            other => return Err(Unavailable::NotCovered(format!("the temporal run {other}"))),
        };
        let analyzer =
            Analyzer::builder().add(extractor).add(DocumentDateSelector).build().map_err(|e| broken(e.to_string()))?;
        Ok(Box::new(Temporal { analyzer, reference }))
    }
}

impl Run for Temporal {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        if config(golden, "reference") != Some(self.reference) {
            return Outcome::Skipped(format!("the golden was made with reference {:?}", config(golden, "reference")));
        }
        match matches_golden::<Temporals>(doc, golden, Tolerance::exact()) {
            Ok(()) => outcome(matches_golden::<DocumentDate>(doc, golden, Tolerance::exact())),
            Err(m) => outcome(Err(m)),
        }
    }
}
