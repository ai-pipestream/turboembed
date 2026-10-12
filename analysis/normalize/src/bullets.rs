//! Replaces list bullets with a space.

use turbo_analysis::ViewBuilder;

use crate::Normalizer;
use crate::substitute::substitute;
use crate::tables::is_bullet;

/// Replaces each bullet of [`BULLETS`](crate::tables::BULLETS) with
/// U+0020, so a bullet reads as a separator. Each bullet becomes one
/// space; a whitespace normalizer run afterwards collapses the space with
/// its neighbours.
///
/// OpenNLP counterpart: `BulletCharSequenceNormalizer`, which has five of
/// the ten bullets; this set adds U+204C, U+204D, U+25D8, U+29BE and
/// U+29BF.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bullets;

impl Normalizer for Bullets {
    const ID: &'static str = "turbo:normalize:bullets";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        substitute(text, out, |c| is_bullet(c).then_some(" "));
    }
}
