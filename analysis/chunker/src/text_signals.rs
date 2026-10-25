//! Optional lexical and discourse costs combined with an existing signal.

use turbo_analysis::{Annotator, Bound, Error, Layer, Refs};
use turbo_analysis_layers::{Boundaries, Sentences};

use crate::{DISCOURSE, LEXICAL, STRUCTURE, Weight, weigh_with_sentiment};

/// Combines a base (structure, or an already weighed signal) with the
/// two text signals. Missing inputs are allowed only at weight zero.
/// Defaults leave the base unchanged. Bind the base read to another
/// instance to add these to embedding drift or other signals.
#[derive(Clone, Copy, Debug)]
pub struct WeighTextBoundaries {
    pub base: Weight,
    pub lexical: Weight,
    pub discourse: Weight,
}

impl Default for WeighTextBoundaries {
    fn default() -> Self {
        Self { base: Weight::ONE, lexical: Weight(0), discourse: Weight(0) }
    }
}

impl WeighTextBoundaries {
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).read(1, STRUCTURE).read(2, LEXICAL).read(3, DISCOURSE)
    }
}

impl Annotator for WeighTextBoundaries {
    const ID: &'static str = "turbo:weigh-text-boundaries";
    type Reads = (Sentences, Boundaries, Option<Boundaries>, Option<Boundaries>);
    type Writes = (Boundaries,);

    fn annotate(&self, (sentences, base, lexical, discourse): Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        sentences.check().map_err(fail)?;
        let n = sentences.spans.len();
        let zero = Boundaries { cost: vec![0; n], reason: vec![turbo_analysis_layers::Cut::Sentence; n] };
        for (name, signal, weight) in [(LEXICAL, lexical, self.lexical), (DISCOURSE, discourse, self.discourse)] {
            if signal.is_none() && weight.0 != 0 {
                return Err(fail(format!("missing {name} signal with nonzero weight")));
            }
        }
        let (lexical, discourse) = (lexical.unwrap_or(&zero), discourse.unwrap_or(&zero));
        for signal in [base, lexical, discourse] {
            signal.check().map_err(fail)?;
            if signal.cost.len() != n {
                return Err(fail("one boundary cost per sentence required".into()));
            }
        }
        let out = weigh_with_sentiment(base, lexical, discourse, (self.base, self.lexical, self.discourse))
            .ok_or_else(|| fail("boundary column lengths differ".into()))?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        [self.base.0, self.lexical.0, self.discourse.0]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, &w| (h ^ u64::from(w as u32)).wrapping_mul(0x0100_0000_01b3))
    }
}
