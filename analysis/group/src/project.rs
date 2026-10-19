//! Boilerplate flags carried from one level to another: from structure
//! blocks to the sentences, chunks or tokens over them.

use std::marker::PhantomData;

use turbo_analysis::{Annotator, Error, Layer, Refs};
use turbo_analysis_layers::{Boilerplate, BoilerplateKind, Covered, Structure};

use crate::sketch::item_spans;

/// Reads the items of `S` (structure blocks by default), their
/// boilerplate flags and the items of `L`; writes `Boilerplate<L>`.
///
/// An item of `L` is boilerplate when at least half of its bytes (over
/// its fragments, for a discontinuous item) lie inside boilerplate items
/// of `S`, and then of the kind that covers the most of its bytes, added
/// up over every interval and fragment (the first in kind order on a
/// tie). Its score is the
/// byte-weighted mean of the covering items' scores, uncovered bytes
/// counting zero, so a chunk that is a third footer scores a third of the
/// footer's score, flagged or not. An empty item is boilerplate when its
/// offset is inside boilerplate, with that interval's kind and score.
/// Both layers must be in one view. Two binary searches per span.
pub struct BoilerplateProjector<L, S = Structure> {
    _of: PhantomData<fn() -> (L, S)>,
}

impl<L, S> Default for BoilerplateProjector<L, S> {
    fn default() -> Self {
        BoilerplateProjector { _of: PhantomData }
    }
}

impl<L: Layer, S: Layer> Annotator for BoilerplateProjector<L, S> {
    const ID: &'static str = "turbo:boilerplate-projector";
    type Reads = (S, Boilerplate<S>, L);
    type Writes = (Boilerplate<L>,);

    fn annotate(&self, (source, flags, units): Refs<'_, Self>) -> Result<(Boilerplate<L>,), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let from = source.text_spans().ok_or_else(|| fail(format!("{} has no spans", S::NAME)))?;
        let to = units.text_spans().ok_or_else(|| fail(format!("{} has no spans", L::NAME)))?;
        if from.view != to.view {
            return Err(fail(format!("{} is in view {}, {} in view {}", S::NAME, from.view.0, L::NAME, to.view.0)));
        }
        let covered = Covered::new(&from, flags).map_err(fail)?;
        let n = to.start.len();
        let (mut kind, mut score) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let (mut len, mut weighted, mut bytes) = (0u64, 0f64, [0u64; 5]);
            for s in item_spans(&to, i) {
                len += u64::from(s.end - s.start);
                weighted += covered.cover(s, &mut bytes);
            }
            if len == 0 {
                let at = covered.at(to.start[i]);
                kind.push(at.map(|k| covered.kind[k]));
                score.push(at.map_or(0.0, |k| covered.score[k]));
                continue;
            }
            // The kind with the most bytes over all fragments; the first in
            // kind order on a tie.
            let (best, most) = (0..5).fold((0, 0), |b, k| if bytes[k] > b.1 { (k, bytes[k]) } else { b });
            let total: u64 = bytes.iter().sum();
            kind.push((most > 0 && 2 * total >= len).then(|| BoilerplateKind::ALL[best]));
            score.push(((weighted / len as f64) as f32).clamp(0.0, 1.0));
        }
        Ok((Boilerplate::new(kind, score),))
    }
}
