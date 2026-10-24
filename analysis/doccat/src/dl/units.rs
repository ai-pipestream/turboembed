use super::*;
use turbo_analysis::{Error, Tag};
use turbo_analysis_layers::{Categories, CategoryScope};

pub(super) enum Unit {
    Empty,
    Pieces(Vec<i32>),
    Text(String),
}

impl<S: DlSequenceScorer, C: ClassificationScoringStrategy> DocumentCategorizerDL<S, C> {
    pub(super) fn score_units<L: CategoryScope>(
        &self,
        caller: &'static str,
        docs: Vec<Result<Vec<Unit>, Error>>,
    ) -> Vec<Result<(Categories<L>,), Error>> {
        let pieces: Vec<_> = docs
            .iter()
            .flatten()
            .flatten()
            .filter_map(|unit| if let Unit::Pieces(ids) = unit { Some(ids.as_slice()) } else { None })
            .collect();
        let texts: Vec<_> = docs
            .iter()
            .flatten()
            .flatten()
            .filter_map(|unit| if let Unit::Text(text) = unit { Some(text.as_str()) } else { None })
            .collect();
        let mut p = self.categorize_subword_batches(&pieces).into_iter();
        let mut t = self.categorize_text_batches(&texts).into_iter();
        docs.into_iter()
            .map(|doc| {
                let units = doc?;
                let mut out = Categories::with_capacity(Arc::clone(self.categories()), units.len());
                let mut error = None;
                // Consume every result even if an earlier unit failed, so later
                // documents cannot receive the failed document's remaining scores.
                for unit in units {
                    let result = match unit {
                        Unit::Empty => {
                            out.push(Tag::NONE, &vec![0.0; self.number_of_categories()]);
                            continue;
                        }
                        Unit::Pieces(_) => p.next().expect("one result per piece unit"),
                        Unit::Text(_) => t.next().expect("one result per text unit"),
                    };
                    match result.and_then(|scores| Ok((self.best_index(&scores)?, scores))) {
                        Ok((best, scores)) => out.push(Tag(best as u16), &scores),
                        Err(e) => {
                            error = Some(Error::Annotator { annotator: caller, detail: e.to_string() });
                        }
                    }
                }
                match error {
                    Some(e) => Err(e),
                    None => Ok((out,)),
                }
            })
            .collect()
    }
}
