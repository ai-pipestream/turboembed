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
/// of `S`, and then of the kind covering most of it. Its score is the
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
        let covered = Covered::new(&from, flags);
        let n = to.start.len();
        let (mut kind, mut score) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let (mut len, mut bytes, mut weighted, mut votes) = (0u64, 0u64, 0f64, Vec::new());
            for s in item_spans(&to, i) {
                let (b, w, k) = covered.cover(s);
                len += u64::from(s.end - s.start);
                bytes += b;
                weighted += w;
                votes.extend(k.map(|k| (b, k)));
            }
            if len == 0 {
                let k = covered.spans.partition_point(|c| c.end <= to.start[i]);
                let inside = covered.spans.get(k).is_some_and(|c| c.start <= to.start[i] && to.start[i] < c.end);
                kind.push(inside.then(|| covered.kind[k]));
                score.push(if inside { covered.score[k] } else { 0.0 });
                continue;
            }
            let best = votes.iter().fold(None, |best: Option<(u64, BoilerplateKind)>, &(b, k)| match best {
                Some((bb, _)) if bb >= b => best,
                _ => Some((b, k)),
            });
            kind.push(best.filter(|_| 2 * bytes >= len).map(|(_, k)| k));
            score.push(((weighted / len as f64) as f32).clamp(0.0, 1.0));
        }
        Ok((Boilerplate::new(kind, score),))
    }
}
