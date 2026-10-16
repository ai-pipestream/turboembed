//! The PII runs: one per `PiiPacks` factory, named as the pack
//! (`payment`, `all-structured`).

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_layers::Pii;
use turbo_goldens::{Golden, Tolerance, matches_golden};
use turbo_pii::{Pack, PiiDetector};

use super::{Outcome, Run, RustError, analyze, config, outcome};
use crate::resources::Unavailable;

pub(crate) struct PiiRun {
    analyzer: Analyzer,
    pack: Pack,
}

impl PiiRun {
    pub(crate) fn open(run: &str) -> Result<Box<dyn Run>, Unavailable> {
        let pack = Pack::parse(run).ok_or_else(|| Unavailable::NotCovered(format!("the PII pack {run}")))?;
        let analyzer =
            Analyzer::builder().add(PiiDetector::pack(pack)).build().map_err(|e| Unavailable::Broken(e.to_string()))?;
        Ok(Box::new(PiiRun { analyzer, pack }))
    }
}

impl Run for PiiRun {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        if config(golden, "pack") != Some(self.pack.name()) {
            return Outcome::Skipped(format!("the golden was made with pack {:?}", config(golden, "pack")));
        }
        outcome(matches_golden::<Pii>(doc, golden, Tolerance::exact()))
    }
}
