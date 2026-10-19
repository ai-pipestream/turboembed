//! The stemmers' layer, so another add-on can read stems without depending
//! on the stemmers that make them.

pub use turbo_analysis_layers::{Stems, TermSlot, Terms};

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Anchor, Layer};
    use turbo_analysis_layers::Tokens;

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
