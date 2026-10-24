//! Flatten windows across independent units without mixing their aggregates.
use super::*;
use std::ops::Range;

impl<S: DlSequenceScorer, C: ClassificationScoringStrategy> DocumentCategorizerDL<S, C> {
    /// Classify pre-tokenized units in one scorer batch, retaining per-unit errors.
    pub fn categorize_subword_batches(&self, inputs: &[&[i32]]) -> Vec<Result<Vec<f64>, DlDoccatError>> {
        let size = self.options.document_split_size.min(self.scorer.max_content_tokens().max(1)).max(1);
        let overlap = self.options.split_overlap_size.min(size.saturating_sub(1));
        let chunks = inputs
            .iter()
            .map(|ids| window_ranges(ids.len(), size, overlap).map(|rs| rs.into_iter().map(|r| &ids[r]).collect()))
            .collect();
        self.score_many(chunks, |windows| self.scorer.score_subword_windows(windows))
    }

    /// Classify raw texts in one scorer batch, with the configured preprocessing
    /// and scoring strategy applied independently to each text's windows.
    pub fn categorize_text_batches(&self, inputs: &[&str]) -> Vec<Result<Vec<f64>, DlDoccatError>> {
        let chunks = inputs
            .iter()
            .map(|text| whitespace_chunks(text, self.options.document_split_size, self.options.split_overlap_size))
            .collect();
        self.score_many(chunks, |windows| self.scorer.score_chunks(windows, &self.options))
    }

    fn score_many<W>(
        &self,
        chunks: Vec<Result<Vec<W>, DlDoccatError>>,
        score: impl Fn(&[W]) -> Result<Vec<Vec<f32>>, DlDoccatError>,
    ) -> Vec<Result<Vec<f64>, DlDoccatError>> {
        let mut windows = Vec::new();
        let ranges: Vec<Result<Range<usize>, DlDoccatError>> = chunks
            .into_iter()
            .map(|c| {
                let c = c?;
                if c.is_empty() {
                    return Err(DlDoccatError::EmptyDocument("no windows to classify".into()));
                }
                let start = windows.len();
                windows.extend(c);
                Ok(start..windows.len())
            })
            .collect();
        if windows.is_empty() {
            return ranges.into_iter().map(|r| r.map(|_| Vec::new())).collect();
        }
        let logits = score(&windows);
        ranges
            .into_iter()
            .map(|range| {
                let range = range?;
                match &logits {
                    Ok(rows) => {
                        if rows.len() != windows.len() {
                            return Err(DlDoccatError::InvalidLogits(
                                "scorer output count differs from batch windows".into(),
                            ));
                        }
                        self.combine_logit_rows(&rows[range])
                    }
                    // A backend can reject one item. Retry only after a failed batch
                    // to preserve valid neighbors, not on the successful path.
                    Err(_) => {
                        let rows = score(&windows[range.clone()])?;
                        if rows.len() != range.len() {
                            return Err(DlDoccatError::InvalidLogits(
                                "scorer output count differs from item windows".into(),
                            ));
                        }
                        self.combine_logit_rows(&rows)
                    }
                }
            })
            .collect()
    }
}
