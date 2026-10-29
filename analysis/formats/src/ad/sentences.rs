//! Sentence samples from Arvores Deitadas: `ADSentenceSampleStream`.

use std::borrow::Cow;

use turbo_analysis::Span;

use super::metadata;
use super::sentence::{AdSentence, SentenceRead, SentenceStream};
use super::{AdError, NullAt};
use crate::error::FormatError;
use crate::sample::SentenceSample;
use crate::source::Lines;

/// The characters that end a Portuguese sentence
/// (`opennlp.tools.sentdetect.lang.Factory.ptEosCharacters`).
const PT_EOS: [char; 11] = ['.', '?', '!', ';', ':', '(', ')', '\u{AB}', '\u{BB}', '\'', '"'];

/// Sentence samples read from an Arvores Deitadas text, one per text:
/// `ADSentenceSampleStream`.
///
/// A sample is the sentences of one text (the text id of the metadata),
/// joined with a space. A sentence whose text does not end with one of
/// `.?!;:()«»'"` is left out, and so is one inside a title (`<t>`),
/// unless `include_titles` is set. The sentences of the next text start
/// the next sample.
///
/// The document is the sample's own text; each sentence keeps the span of
/// its text line in the source.
pub struct AdSentenceReader<'a> {
    sentences: SentenceStream<'a>,
    include_titles: bool,
    /// The sentence read last, which starts the next sample.
    sent: Option<AdSentence<'a>>,
    text: i32,
    paragraph: i32,
    same_text: bool,
    same_paragraph: bool,
    title: bool,
    done: bool,
}

impl<'a> AdSentenceReader<'a> {
    /// A reader of `source`, which must fit `u32` offsets.
    pub fn new(source: &'a str, include_titles: bool) -> Result<AdSentenceReader<'a>, FormatError> {
        Ok(AdSentenceReader {
            sentences: SentenceStream::new(Lines::new(source)?),
            include_titles,
            sent: None,
            text: -1,
            paragraph: -1,
            same_text: false,
            same_paragraph: false,
            title: false,
            done: false,
        })
    }

    /// The next sentence; a sentence the parser gave up on is Java's
    /// `null`, which ends the sample, but the reading goes on after it.
    fn read_sentence(&mut self) -> Option<AdSentence<'a>> {
        match self.sentences.read()? {
            SentenceRead::Sentence(s) => Some(s),
            SentenceRead::Failed => None,
        }
    }

    /// `updateMeta`: whether the sentence read last is in the same text
    /// and paragraph as the one before, and in a title.
    fn update_meta(&mut self) -> Result<(), AdError> {
        let Some(sent) = &self.sent else {
            self.same_text = false;
            self.same_paragraph = false;
            return Ok(());
        };
        let meta = sent.metadata.as_deref().ok_or(AdError::NullMetadata { at: NullAt::Length })?;
        let ids =
            metadata::text_and_paragraph(meta).ok_or_else(|| AdError::InvalidMetadata { metadata: meta.to_owned() })?;
        self.same_text = ids.text == self.text;
        self.same_paragraph = self.same_text && ids.paragraph == self.paragraph;
        self.title = meta.contains("title");
        self.text = ids.text;
        self.paragraph = ids.paragraph;
        Ok(())
    }

    fn read(&mut self) -> Option<Result<SentenceSample<'a>, FormatError>> {
        if self.sent.is_none() {
            self.sent = self.read_sentence();
            if let Err(e) = self.update_meta() {
                return Some(Err(e.into()));
            }
            self.sent.as_ref()?;
        }
        let mut document = String::new();
        let mut spans = Vec::new();
        let mut sources = Vec::new();
        let mut covered: Option<Span> = None;
        loop {
            let sent = self.sent.take().expect("a sentence to add");
            covered = Some(covered.map_or(sent.source, |c| Span::new(c.start, sent.source.end)));
            if !self.title || self.include_titles {
                let Some(text) = &sent.text else {
                    // Java's hasPunctuation throws on a null text, and
                    // throws again on every later read; this reader ends.
                    self.done = true;
                    return Some(Err(AdError::NullText.into()));
                };
                if has_punctuation(text.as_str()) {
                    let start = document.len() as u32;
                    document.push_str(text.as_str());
                    spans.push(Span::new(start, document.len() as u32));
                    sources.push(text.span);
                    document.push(' ');
                }
            }
            self.sent = self.read_sentence();
            if let Err(e) = self.update_meta() {
                return Some(Err(e.into()));
            }
            if !self.same_paragraph && !self.same_text {
                break;
            }
        }
        document.pop();
        let source = covered.expect("a sentence was read");
        Some(SentenceSample::new(Cow::Owned(document), spans, sources, source))
    }
}

/// `hasPunctuation`: whether the text, trimmed as `String.trim` trims,
/// ends with one of [`PT_EOS`].
fn has_punctuation(text: &str) -> bool {
    text.trim_matches(|c: char| c <= ' ').chars().next_back().is_some_and(|c| PT_EOS.contains(&c))
}

impl<'a> Iterator for AdSentenceReader<'a> {
    type Item = Result<SentenceSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let item = self.read();
        if item.is_none() {
            self.done = true;
        }
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texts_make_samples() {
        let src = "<s>\nSOURCE: a\n1 Um.\n</s>\n<s>\nSOURCE: b\n1 Dois\n</s>\n<p>\n<s>\nSOURCE: c\n1 Tres!\n</s>\n<s>\nSOURCE: d\n2 Quatro?\n</s>\n";
        let samples: Vec<_> = AdSentenceReader::new(src, false).unwrap().map(|s| s.unwrap()).collect();
        let docs: Vec<&str> = samples.iter().map(|s| s.document()).collect();
        assert_eq!(docs, ["Um. Tres!", "Quatro?"]);
        assert_eq!(&src[samples[0].sources()[1].range()], "Tres!");
    }
}
