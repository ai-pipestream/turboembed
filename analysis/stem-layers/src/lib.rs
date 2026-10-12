//! The stemmers' layer, so another add-on can read stems without depending
//! on the stemmers that make them.

use turbo_analysis::{Anchor, Layer};
use turbo_analysis_layers::{Strs, Tokens};

/// The stem of every token: item i is the stem of token i. OpenNLP's
/// `StemmerAnnotator.STEMS`, which the OpenNLP gRPC standard has no id
/// for.
#[derive(Default)]
pub struct Stems {
    pub stem: Strs,
}

impl Layer for Stems {
    const ID: &'static str = "turbo:stems";

    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }

    fn len(&self) -> usize {
        self.stem.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_to_tokens() {
        assert_eq!(Stems::anchor(), Anchor::aligned::<Tokens>());
        let mut stems = Stems::default();
        stems.stem.push("run");
        stems.stem.push("");
        assert_eq!(stems.len(), 2);
        assert_eq!(stems.stem.get(0), "run");
        assert_eq!(Stems::ID, "turbo:stems");
    }
}
