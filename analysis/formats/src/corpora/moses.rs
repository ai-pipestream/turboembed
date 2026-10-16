//! Moses sentence files: `MosesSentenceSampleStream`.

use std::borrow::Cow;

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::native::sentence::EmptyLinePreprocessor;
use crate::sample::SentenceSample;
use crate::source::java_trim;

/// The sentences a Moses or LetsMT sample holds at most.
pub const SENTENCES_PER_SAMPLE: usize = 25;

/// Reads sentence samples from a Moses file: one sentence per line, read
/// through `EmptyLinePreprocessorStream`, 25 sentences per sample.
///
/// Each sentence is its line trimmed with `String.trim`, followed by a
/// space in the sample's document. An empty line (after the trim) is not
/// a break between samples: a run of them is one empty sentence, except
/// at the start of the file, where they are skipped. So a file that ends
/// with an empty line ends with an empty sentence.
#[derive(Clone, Debug)]
pub struct MosesReader<'a> {
    lines: EmptyLinePreprocessor<'a>,
}

impl<'a> MosesReader<'a> {
    pub fn new(source: &'a str) -> Result<MosesReader<'a>, FormatError> {
        Ok(MosesReader { lines: EmptyLinePreprocessor::new(source)? })
    }
}

impl<'a> Iterator for MosesReader<'a> {
    type Item = Result<SentenceSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut document = String::new();
        let mut sentences = Vec::new();
        let mut sources = Vec::new();
        let mut source: Option<Span> = None;
        // A trimmed line and its space take no more bytes than the line and
        // its terminator (one more for a last line without one), so no
        // offset below passes the source's length plus one.
        while sentences.len() < SENTENCES_PER_SAMPLE {
            let Some(line) = self.lines.next_line() else { break };
            let (line, trimmed) = match line {
                Ok(line) => (line, java_trim(line.piece)),
                // An empty line is an empty sentence; its source is the
                // line's start.
                Err(line) => (line, line.piece.slice(0, 0)),
            };
            let start = document.len() as u32;
            document.push_str(trimmed.text);
            sentences.push(Span::new(start, document.len() as u32));
            sources.push(trimmed.span);
            source = Some(source.map_or(line.span(), |s| Span::new(s.start, line.span().end)));
            document.push(' ');
        }
        let source = source?;
        Some(SentenceSample::new(Cow::Owned(document), sentences, sources, source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(src: &str) -> Vec<String> {
        MosesReader::new(src).unwrap().map(|s| s.unwrap().document().to_owned()).collect()
    }

    #[test]
    fn empty_lines_are_empty_sentences() {
        assert_eq!(read("\n \n a \n\n\nb\n\n"), ["a  b  "]);
        let s = MosesReader::new("\n a \n\n\nb\n\n").unwrap().next().unwrap().unwrap();
        let texts: Vec<&str> = (0..s.sentences().len()).map(|i| s.sentence(i)).collect();
        assert_eq!(texts, ["a", "", "b", ""]);
        assert_eq!(s.sources()[0], Span::new(2, 3));
        assert_eq!(s.source(), Span::new(1, 9));
    }

    #[test]
    fn twenty_five_sentences_per_sample() {
        let src: String = (0..26).map(|i| format!("s{i}\n")).collect();
        let samples: Vec<_> = MosesReader::new(&src).unwrap().map(Result::unwrap).collect();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].sentences().len(), 25);
        assert_eq!(samples[1].document(), "s25 ");
        assert!(read(" \n\n").is_empty());
    }
}
