//! Part-of-speech samples from Arvores Deitadas: `ADPOSSampleStream`.

use std::borrow::Cow;

use turbo_analysis::Span;

use super::sentence::{AdSentence, SentenceRead, SentenceStream};
use super::tree::{Leaf, is_ws};
use crate::error::FormatError;
use crate::sample::PosSample;
use crate::source::{Field, Lines, Piece};

/// Part-of-speech samples read from an Arvores Deitadas text, one per
/// sentence: `ADPOSSampleStream`.
///
/// A token is a leaf's word as written, and its tag the leaf's word class
/// (`n`, `v-fin`, ...); punctuation, which has none, is tagged with
/// itself. With `include_features` the morphological tag follows after a
/// space, and every run of whitespace in the tag becomes one `=`
/// (`n=M=S`). With `expand_me` a word with underscores is one token per
/// part, tagged `B-` and then `I-` before the tag.
///
/// Every token is a field of the text; a tag is one too, unless it was
/// joined with the features or prefixed, when it has a text of its own and
/// the span of what it was made from.
pub struct AdPosReader<'a> {
    sentences: SentenceStream<'a>,
    expand_me: bool,
    include_features: bool,
    done: bool,
}

impl<'a> AdPosReader<'a> {
    /// A reader of `source`, which must fit `u32` offsets.
    pub fn new(source: &'a str, expand_me: bool, include_features: bool) -> Result<AdPosReader<'a>, FormatError> {
        Ok(AdPosReader {
            sentences: SentenceStream::new(Lines::new(source)?),
            expand_me,
            include_features,
            done: false,
        })
    }

    fn sample(&self, sentence: AdSentence<'a>) -> Result<PosSample<'a>, FormatError> {
        let mut tokens = Vec::new();
        let mut tags = Vec::new();
        for leaf in sentence.tree.leaves() {
            self.leaf(leaf, &mut tokens, &mut tags);
        }
        PosSample::new(tokens, tags, sentence.source)
    }

    /// `processLeaf`.
    fn leaf(&self, leaf: &Leaf<'a>, tokens: &mut Vec<Field<'a>>, tags: &mut Vec<Field<'a>>) {
        let lexeme = leaf.lexeme;
        let base = leaf.functional.unwrap_or(lexeme);
        let tag = match leaf.morphological.filter(|_| self.include_features) {
            Some(m) => {
                let joined = format!("{} {}", base.text, m.text);
                let span = Span::new(base.span.start.min(m.span.start), base.span.end.max(m.span.end));
                Field::made(join_whitespace(&joined).into_owned(), span)
            }
            None => match join_whitespace(base.text) {
                Cow::Borrowed(_) => base.field(),
                Cow::Owned(t) => Field::made(t, base.span),
            },
        };
        let mut parts = lexeme.text.split('_').filter(|p| !p.is_empty());
        if self.expand_me && lexeme.text.contains('_') && parts.next().is_some() {
            let mut first = true;
            for part in underscore_runs(lexeme) {
                tokens.push(part.field());
                let prefix = if first { "B-" } else { "I-" };
                tags.push(Field::made(format!("{prefix}{}", tag.as_str()), tag.span));
                first = false;
            }
        } else {
            tokens.push(lexeme.field());
            tags.push(tag);
        }
    }
}

/// The parts of a word between underscores, as `StringTokenizer` gives
/// them.
fn underscore_runs(p: Piece<'_>) -> impl Iterator<Item = Piece<'_>> {
    let text = p.text;
    let mut at = 0;
    std::iter::from_fn(move || {
        let start = at + text[at..].bytes().position(|b| b != b'_')?;
        let end = text[start..].find('_').map_or(text.len(), |i| start + i);
        at = end;
        Some(p.slice(start, end))
    })
}

/// `replaceWhitespaceWithEquals`: every run of whitespace replaced by one
/// `=`.
fn join_whitespace(tag: &str) -> Cow<'_, str> {
    if !tag.chars().any(is_ws) {
        return Cow::Borrowed(tag);
    }
    let mut out = String::with_capacity(tag.len());
    let mut in_run = false;
    for c in tag.chars() {
        if is_ws(c) {
            if !in_run {
                out.push('=');
            }
            in_run = true;
        } else {
            out.push(c);
            in_run = false;
        }
    }
    Cow::Owned(out)
}

impl<'a> Iterator for AdPosReader<'a> {
    type Item = Result<PosSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.sentences.read() {
            Some(SentenceRead::Sentence(s)) => Some(self.sample(s)),
            Some(SentenceRead::Failed) | None => {
                self.done = true;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_runs_become_one_equals_sign() {
        assert_eq!(join_whitespace("n M  S"), "n=M=S");
        assert_eq!(join_whitespace(" a\u{2003}"), "=a=");
        assert!(matches!(join_whitespace("n"), Cow::Borrowed("n")));
    }

    #[test]
    fn samples() {
        let src = "<s>\nSOURCE: x\n1 Rio de Janeiro .\nSTA:fcl\n=H:prop('Rio_de_Janeiro' <civ> M S) Rio_de_Janeiro\n=.\n</s>\n";
        let plain: Vec<_> = AdPosReader::new(src, false, false).unwrap().map(|s| s.unwrap().to_string()).collect();
        assert_eq!(plain, ["Rio_de_Janeiro_prop ._."]);
        let expanded: Vec<_> = AdPosReader::new(src, true, true).unwrap().map(|s| s.unwrap().to_string()).collect();
        assert_eq!(expanded, ["Rio_B-prop=M=S de_I-prop=M=S Janeiro_I-prop=M=S ._."]);
    }
}
