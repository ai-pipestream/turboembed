//! The samples OpenNLP reads from CoNLL-U: `ConlluPOSSampleStream`,
//! `ConlluLemmaSampleStream`, `ConlluSentenceSampleStream` and
//! `ConlluTokenSampleStream`.
//!
//! - A part-of-speech or lemma sample is a sentence's word lines: each
//!   line's form, its tag in a [`Tagset`] and its lemma. Empty nodes are
//!   in it too, and a sentence of comments only is a sample with no
//!   tokens.
//! - A sentence sample is the `text` comments of up to a number of
//!   sentences, each followed by a space, the last space removed. A
//!   sentence without one stands in the document as `null`, Java's
//!   string of a missing value; that sentence's source is the sentence's
//!   lines. A number of zero gives no samples. An error in a sentence
//!   discards the sentences read before it into the same sample.
//! - A token sample is a sentence's `text` comment with `<SPLIT>`
//!   inserted after each word form that is followed by a character that
//!   is not whitespace (in the whitespace mode), parsed as the token
//!   format parses a line. Each form is searched for in the text from
//!   the end of the one before, in the text with the markers inserted so
//!   far, so a form can be found inside a marker; empty nodes are left
//!   out. A token's source is the bytes of the comment it holds; a token
//!   made only of marker bytes, or an empty one, has an empty source.

use std::borrow::Cow;

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use super::{ConlluReader, Sentence, Tagset};
use crate::error::FormatError;
use crate::native::token::parse_text;
use crate::sample::lemma::LemmaSample;
use crate::sample::pos::PosSample;
use crate::sample::sentence::SentenceSample;
use crate::sample::token::{SPLIT_MARKER, SplitMarker, TokenSample};
use crate::source::Piece;

/// Reads part-of-speech samples from CoNLL-U.
#[derive(Clone, Debug)]
pub struct ConlluPosReader<'a> {
    sentences: ConlluReader<'a>,
    tagset: Tagset,
}

impl<'a> ConlluPosReader<'a> {
    pub fn new(source: &'a str, tagset: Tagset) -> Result<ConlluPosReader<'a>, FormatError> {
        Ok(ConlluPosReader { sentences: ConlluReader::new(source)?, tagset })
    }
}

impl<'a> Iterator for ConlluPosReader<'a> {
    type Item = Result<PosSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sentences.next()?.and_then(|s| {
            let tokens = s.word_lines.iter().map(|l| l.form.clone()).collect();
            let tags = s.word_lines.iter().map(|l| l.tag(self.tagset).clone()).collect();
            PosSample::new(tokens, tags, s.source)
        }))
    }
}

/// Reads lemma samples from CoNLL-U.
#[derive(Clone, Debug)]
pub struct ConlluLemmaReader<'a> {
    sentences: ConlluReader<'a>,
    tagset: Tagset,
}

impl<'a> ConlluLemmaReader<'a> {
    pub fn new(source: &'a str, tagset: Tagset) -> Result<ConlluLemmaReader<'a>, FormatError> {
        Ok(ConlluLemmaReader { sentences: ConlluReader::new(source)?, tagset })
    }
}

impl<'a> Iterator for ConlluLemmaReader<'a> {
    type Item = Result<LemmaSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sentences.next()?.and_then(|s| {
            let tokens = s.word_lines.iter().map(|l| l.form.clone()).collect();
            let tags = s.word_lines.iter().map(|l| l.tag(self.tagset).clone()).collect();
            let lemmas = s.word_lines.iter().map(|l| l.lemma.clone()).collect();
            LemmaSample::new(tokens, tags, lemmas, s.source)
        }))
    }
}

/// Reads sentence samples from CoNLL-U, up to a number of sentences each.
#[derive(Clone, Debug)]
pub struct ConlluSentenceReader<'a> {
    sentences: ConlluReader<'a>,
    per_sample: usize,
}

impl<'a> ConlluSentenceReader<'a> {
    /// A reader of `source` with up to `per_sample` sentences a sample.
    pub fn new(source: &'a str, per_sample: usize) -> Result<ConlluSentenceReader<'a>, FormatError> {
        Ok(ConlluSentenceReader { sentences: ConlluReader::new(source)?, per_sample })
    }
}

impl<'a> Iterator for ConlluSentenceReader<'a> {
    type Item = Result<SentenceSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut document = String::new();
        let mut spans = Vec::new();
        let mut sources = Vec::new();
        for _ in 0..self.per_sample {
            let sentence = match self.sentences.next() {
                None => break,
                Some(Ok(s)) => s,
                Some(Err(e)) => return Some(Err(e)),
            };
            let (text, source) = match &sentence.text {
                Some(t) => (t.as_str(), t.span),
                None => ("null", sentence.source),
            };
            let start = document.len();
            document.push_str(text);
            let end = document.len();
            let (Ok(start), Ok(end)) = (u32::try_from(start), u32::try_from(end)) else {
                return Some(Err(FormatError::TooLong { len: end }));
            };
            spans.push(Span::new(start, end));
            sources.push(source);
            document.push(' ');
        }
        document.pop()?;
        let source = Span::new(sources.first()?.start, sources.last()?.end);
        Some(SentenceSample::new(Cow::Owned(document), spans, sources, source))
    }
}

/// Reads token samples from CoNLL-U.
#[derive(Clone, Debug)]
pub struct ConlluTokenReader<'a> {
    sentences: ConlluReader<'a>,
    mode: WhitespaceMode,
}

impl<'a> ConlluTokenReader<'a> {
    /// A reader of `source`, testing for whitespace and splitting the
    /// text in `mode`.
    pub fn new(source: &'a str, mode: WhitespaceMode) -> Result<ConlluTokenReader<'a>, FormatError> {
        Ok(ConlluTokenReader { sentences: ConlluReader::new(source)?, mode })
    }
}

impl<'a> Iterator for ConlluTokenReader<'a> {
    type Item = Result<TokenSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sentences.next()?.and_then(|s| token_sample(&s, self.mode)))
    }
}

/// `ConlluTokenSampleStream.read` of one sentence.
fn token_sample<'a>(sentence: &Sentence<'a>, mode: WhitespaceMode) -> Result<TokenSample<'a>, FormatError> {
    let Some(comment) = &sentence.text else {
        return Err(FormatError::NoSentenceText { line: sentence.line });
    };
    // The text with the markers, and for each of its bytes the source byte
    // it is, or none for a marker's byte.
    let mut text = comment.as_str().to_owned();
    let base = comment.span.start;
    let mut origin: Vec<Option<u32>> = (base..comment.span.end).map(Some).collect();
    let mut search = 0;
    for word in &sentence.word_lines {
        if word.id.as_str().contains('.') {
            continue;
        }
        let form = word.form.as_str();
        let Some(at) = text[search..].find(form).map(|i| search + i) else {
            return Err(FormatError::TokenNotInText {
                line: word.line,
                token: form.to_owned(),
                sentence: sentence.sentence_id.as_ref().map(|f| f.as_str().to_owned()),
                text,
            });
        };
        search = at + form.len();
        if let Some(c) = text[search..].chars().next()
            && !mode.is_whitespace(c as u32)
        {
            text.insert_str(search, SPLIT_MARKER);
            origin.splice(search..search, std::iter::repeat_n(None, SPLIT_MARKER.len()));
        }
    }
    let len = u32::try_from(text.len()).map_err(|_| FormatError::TooLong { len: text.len() })?;
    let end = comment.span.end;
    // The source byte of the first comment byte at or after `at`.
    let next = |at: usize| origin[at..].iter().find_map(|&o| o).unwrap_or(end);
    let to_source = |span: Span| {
        let (s, e) = (span.start as usize, span.end as usize);
        let start = next(s);
        let end = origin[s..e].iter().rev().find_map(|&o| o).map_or(start, |o| o + 1);
        Span::new(start, end)
    };
    let piece = Piece { text: &text, span: Span::new(0, len) };
    parse_text(piece, mode, SplitMarker::default(), to_source, sentence.source)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(id: &str, form: &str, lemma: &str, upos: &str, xpos: &str) -> String {
        format!("{id}\t{form}\t{lemma}\t{upos}\t{xpos}\t_\t_\t_\t_\t_\n")
    }

    #[test]
    fn pos_and_lemma_samples() {
        let src = format!(
            "# text = a b\n{}{}{}\n# only = comments\n",
            line("1", "a", "la", "X", "x"),
            line("1.1", "e", "le", "E", "_"),
            line("2", "b", "_", "Y", "y")
        );
        let pos: Vec<PosSample<'_>> = ConlluPosReader::new(&src, Tagset::X).unwrap().map(Result::unwrap).collect();
        assert_eq!(pos.len(), 2);
        assert_eq!(pos[0].to_string(), "a_x e__ b_y");
        assert!(pos[1].is_empty());
        let lemmas: Vec<LemmaSample<'_>> =
            ConlluLemmaReader::new(&src, Tagset::U).unwrap().map(Result::unwrap).collect();
        let l: Vec<&str> = lemmas[0].lemmas().iter().map(|f| f.as_str()).collect();
        assert_eq!(l, ["la", "le", "_"]);
    }

    #[test]
    fn sentence_samples_stand_null_for_a_missing_text() {
        let src = format!(
            "# text = One.\n{}\n{}\n# text = Three.\n{}\n",
            line("1", "One", "_", "_", "_"),
            line("1", "Two", "_", "_", "_"),
            line("1", "Three", "_", "_", "_")
        );
        let samples: Vec<SentenceSample<'_>> =
            ConlluSentenceReader::new(&src, 2).unwrap().map(Result::unwrap).collect();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].document(), "One. null");
        assert_eq!(samples[0].sentence(1), "null");
        assert_eq!(&src[samples[0].sources()[0].range()], "One.");
        assert_eq!(samples[1].document(), "Three.");
        assert_eq!(ConlluSentenceReader::new(&src, 0).unwrap().count(), 0);
    }

    #[test]
    fn token_samples_insert_splits() {
        let src = format!(
            "# sent_id = s\n# text = Vamos al mar.\n{}{}{}{}{}{}",
            line("1", "Vamos", "_", "_", "_"),
            line("2-3", "al", "_", "_", "_"),
            line("2", "a", "_", "_", "_"),
            line("3", "el", "_", "_", "_"),
            line("4", "mar", "_", "_", "_"),
            line("5", ".", "_", "_", "_"),
        );
        let s = ConlluTokenReader::new(&src, WhitespaceMode::Unicode).unwrap().next().unwrap().unwrap();
        assert_eq!(s.text(), "Vamos al mar. ");
        assert_eq!(s.to_string(), "Vamos al mar<SPLIT>.");
        let sources: Vec<&str> = s.sources().iter().map(|sp| &src[sp.range()]).collect();
        assert_eq!(sources, ["Vamos", "al", "mar", "."]);
    }

    #[test]
    fn a_form_can_be_found_inside_a_marker() {
        // After `a` the text is `a<SPLIT>b`, and `SPLIT` is found inside the
        // marker; `b` is found after it.
        let src = format!(
            "# text = ab\n{}{}{}",
            line("1", "a", "_", "_", "_"),
            line("2", "SPLIT", "_", "_", "_"),
            line("3", "b", "_", "_", "_")
        );
        let s = ConlluTokenReader::new(&src, WhitespaceMode::Unicode).unwrap().next().unwrap().unwrap();
        assert_eq!(s.text(), "a<SPLIT>b ");
        let tokens: Vec<&str> = (0..s.tokens().len()).map(|i| s.token(i)).collect();
        assert_eq!(tokens, ["a<SPLIT", ">b"]);
        let sources: Vec<&str> = s.sources().iter().map(|sp| &src[sp.range()]).collect();
        assert_eq!(sources, ["a", "b"]);
    }

    #[test]
    fn token_errors() {
        let src = format!(
            "# sent_id = s9\n# text = ab\n{}\n{}",
            line("1", "c", "_", "_", "_"),
            line("1", "c", "_", "_", "_")
        );
        let r: Vec<_> = ConlluTokenReader::new(&src, WhitespaceMode::Unicode).unwrap().collect();
        assert_eq!(r[0].as_ref().unwrap_err().to_string(), "Failed to match token [c] in sentence [s9] with text [ab]");
        assert_eq!(r[1], Err(FormatError::NoSentenceText { line: 5 }));
    }
}
