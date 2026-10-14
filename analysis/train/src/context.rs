//! `opennlp.tools.ml.maxent.BasicContextGenerator`: the predicates of a
//! context given as one string, split on whitespace or on a literal
//! separator. OpenNLP's command line uses it to read event files.

use crate::event::split_on_unicode_whitespace;

/// What `BasicContextGenerator` splits on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Separator {
    /// Runs of Unicode `White_Space` (`StringUtil.splitOnUnicodeWhitespace`).
    Whitespace,
    /// A literal string; empty fields between two separators, before the
    /// first and after the last are left out.
    Literal(String),
}

/// The generator with its separator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicContextGenerator {
    separator: Separator,
}

impl Default for BasicContextGenerator {
    fn default() -> BasicContextGenerator {
        BasicContextGenerator::whitespace()
    }
}

impl BasicContextGenerator {
    /// `new BasicContextGenerator()`: whitespace.
    pub fn whitespace() -> BasicContextGenerator {
        BasicContextGenerator { separator: Separator::Whitespace }
    }

    /// `new BasicContextGenerator(sep)`. OpenNLP refuses a null or empty
    /// separator and one with an unpaired surrogate; a `&str` has no
    /// surrogate, so only the empty one is refused, with `None`.
    pub fn with_separator(separator: &str) -> Option<BasicContextGenerator> {
        if separator.is_empty() {
            return None;
        }
        Some(BasicContextGenerator { separator: Separator::Literal(separator.to_owned()) })
    }

    pub fn separator(&self) -> &Separator {
        &self.separator
    }

    /// `getContext(o)`: the predicates of `text`.
    pub fn context<'t>(&self, text: &'t str) -> Vec<&'t str> {
        match &self.separator {
            Separator::Whitespace => split_on_unicode_whitespace(text).collect(),
            Separator::Literal(sep) => text.split(sep.as_str()).filter(|p| !p.is_empty()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_splits_on_the_white_space_property() {
        let g = BasicContextGenerator::default();
        assert_eq!(g.context(" a\u{3000}b=1\tc "), ["a", "b=1", "c"]);
        assert!(g.context(" \u{A0}").is_empty());
        assert_eq!(g.context("a\u{200B}b"), ["a\u{200B}b"]);
    }

    #[test]
    fn a_literal_separator_drops_empty_fields() {
        let g = BasicContextGenerator::with_separator(",").unwrap();
        assert_eq!(g.context(",a,,b,"), ["a", "b"]);
        assert_eq!(g.context("abc"), ["abc"]);
        assert!(g.context("").is_empty());
        assert!(g.context(",,").is_empty());
        let g = BasicContextGenerator::with_separator("::").unwrap();
        assert_eq!(g.context("a::b:::c"), ["a", "b", ":c"]);
        assert_eq!(g.separator(), &Separator::Literal("::".into()));
        assert!(BasicContextGenerator::with_separator("").is_none());
    }
}
