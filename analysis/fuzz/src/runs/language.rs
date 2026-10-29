//! Retired classical language detector runs.

use turbo_analysis::Document;
use turbo_goldens::Golden;

use super::{Outcome, Run, RustError};
use crate::resources::{Resources, Unavailable};

pub(crate) struct Detector;

impl Detector {
    pub(crate) fn open(model: &str, _options: &[&str], _res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        Err(Unavailable::NotCovered(format!("retired classical langdetect/{model}")))
    }
}

impl Run for Detector {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        Ok(Document::new(text))
    }

    fn compare(&self, _doc: &Document, _golden: &Golden) -> Outcome {
        Outcome::Skipped("classical language detection is retired".into())
    }
}
