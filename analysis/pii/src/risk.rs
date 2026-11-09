//! The per-document risk step (#1889): reads the view and its
//! `turbo:pii` layer and writes `turbo:pii-risk`, one score and level for
//! the whole document, with no value.

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::{Pii, PiiRisk, PiiSeverityTable};

/// Scores a document's `turbo:pii` layer under a severity table: the
/// built-in one unless [`PiiRiskScorer::with_table`] sets overrides.
#[derive(Clone, Debug, Default)]
pub struct PiiRiskScorer {
    table: PiiSeverityTable,
}

impl PiiRiskScorer {
    pub fn new() -> PiiRiskScorer {
        PiiRiskScorer::default()
    }

    pub fn with_table(table: PiiSeverityTable) -> PiiRiskScorer {
        PiiRiskScorer { table }
    }

    pub fn table(&self) -> &PiiSeverityTable {
        &self.table
    }
}

impl Annotator for PiiRiskScorer {
    const ID: &'static str = "turbo-pii:risk";
    const VERSION: &'static str = "1";
    type Reads = (View, Pii);
    type Writes = (PiiRisk,);

    fn annotate(&self, (view, pii): Refs<'_, Self>) -> Result<(PiiRisk,), Error> {
        Ok((PiiRisk::compute(pii, view.text(), &self.table),))
    }

    /// The severity table's digest: overrides change it, the built-in
    /// table always gives the same one.
    fn config_hash(&self) -> u64 {
        self.table.digest()
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Analyzer, Document, Key};
    use turbo_analysis_layers::{PiiKind, PiiRiskLevel, PiiSeverity};

    use super::*;
    use crate::{Pack, PiiDetector};

    fn run(text: &str, scorer: PiiRiskScorer) -> PiiRisk {
        let analyzer = Analyzer::builder().add(PiiDetector::pack(Pack::AllStructured)).add(scorer).build().unwrap();
        let mut docs = vec![Document::new(text)];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        docs[0].get_key(&Key::<PiiRisk>::DEFAULT).unwrap().clone()
    }

    #[test]
    fn a_document_with_a_card_is_critical_and_plain_text_is_none() {
        let r = run("Pay with 4111 1111 1111 1111 today.", PiiRiskScorer::new());
        assert_eq!(r.level, PiiRiskLevel::Critical);
        assert_eq!(r.kinds[0].kind, PiiKind::Card);
        let plain = run("Nothing to see here.", PiiRiskScorer::new());
        assert_eq!((plain.level, plain.mentions), (PiiRiskLevel::None, 0));
    }

    #[test]
    fn an_override_changes_the_level_and_the_config_hash() {
        let text = "Write to bob@example.com please.";
        let base = run(text, PiiRiskScorer::new());
        assert_eq!(base.level, PiiRiskLevel::Low);
        let table = PiiSeverityTable::new().with(PiiKind::Email, PiiSeverity::Critical);
        let raised = PiiRiskScorer::with_table(table);
        assert_ne!(raised.config_hash(), PiiRiskScorer::new().config_hash());
        assert!(run(text, raised).level > base.level);
    }
}
