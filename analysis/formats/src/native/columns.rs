//! The lemma and chunk formats: `LemmaSampleStream` and
//! `ChunkSampleStream` (of `opennlp.tools.chunker`).
//!
//! One token per line in three columns, a sentence ending at an empty
//! line. The lemma format separates its columns (token, tag, lemma) with
//! a tab; the chunk format (token, tag, chunk tag, as in CoNLL-2000)
//! with a single space:
//!
//! ```text
//! He PRP B-NP
//! reckons VBZ B-VP
//! ```
//!
//! The columns are split as `StringUtil.split` splits them: at every
//! separator, empty columns at the end removed. A line that does not
//! give exactly three columns is skipped ([`Warning::CorruptLine`]).
//!
//! Two details of OpenNLP's streams, reproduced here:
//!
//! - Only a line of length zero ends a sentence. A line of spaces is a
//!   line with one column (lemma) or no columns (chunk) and is skipped.
//! - A sentence with no valid line ends the stream: `read` returns null
//!   for it, so two empty lines in a row, an empty first line, or a
//!   sentence of corrupt lines stop the reading there. The reader then
//!   returns `None` for good.

use turbo_analysis::Span;

use crate::error::{FormatError, Warning};
use crate::sample::chunk::ChunkSample;
use crate::sample::lemma::LemmaSample;
use crate::source::{Field, Lines, split_fields};

/// The columns of one sentence and the bytes of its lines.
struct Block<'a> {
    columns: [Vec<Field<'a>>; 3],
    source: Span,
}

/// Reads blocks of three-column lines.
#[derive(Clone, Debug)]
struct Blocks<'a> {
    lines: Lines<'a>,
    separator: char,
    warnings: Vec<Warning>,
    done: bool,
}

impl<'a> Blocks<'a> {
    fn new(source: &'a str, separator: char) -> Result<Blocks<'a>, FormatError> {
        Ok(Blocks { lines: Lines::new(source)?, separator, warnings: Vec::new(), done: false })
    }

    fn next_block(&mut self) -> Option<Block<'a>> {
        if self.done {
            return None;
        }
        let mut columns: [Vec<Field<'a>>; 3] = Default::default();
        let mut source: Option<Span> = None;
        while let Some(line) = self.lines.next().filter(|l| !l.text().is_empty()) {
            let span = line.span();
            source = Some(source.map_or(span, |s| Span::new(s.start, span.end)));
            let fields = split_fields(line.piece, self.separator);
            if fields.len() == 3 {
                for (column, field) in columns.iter_mut().zip(fields) {
                    column.push(field.field());
                }
            } else {
                self.warnings.push(Warning::CorruptLine { line: line.number, columns: fields.len() });
            }
        }
        if columns[0].is_empty() {
            self.done = true;
            return None;
        }
        Some(Block { columns, source: source.expect("a block with a line has a span") })
    }
}

/// Reads lemma samples from the lemma format.
#[derive(Clone, Debug)]
pub struct LemmaReader<'a>(Blocks<'a>);

impl<'a> LemmaReader<'a> {
    pub fn new(source: &'a str) -> Result<LemmaReader<'a>, FormatError> {
        Ok(LemmaReader(Blocks::new(source, '\t')?))
    }

    /// The lines read so far that were skipped.
    pub fn warnings(&self) -> &[Warning] {
        &self.0.warnings
    }
}

impl<'a> Iterator for LemmaReader<'a> {
    type Item = Result<LemmaSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let Block { columns: [tokens, tags, lemmas], source } = self.0.next_block()?;
        Some(LemmaSample::new(tokens, tags, lemmas, source))
    }
}

/// Reads chunk samples from the chunk format.
#[derive(Clone, Debug)]
pub struct ChunkReader<'a>(Blocks<'a>);

impl<'a> ChunkReader<'a> {
    pub fn new(source: &'a str) -> Result<ChunkReader<'a>, FormatError> {
        Ok(ChunkReader(Blocks::new(source, ' ')?))
    }

    /// The lines read so far that were skipped.
    pub fn warnings(&self) -> &[Warning] {
        &self.0.warnings
    }
}

impl<'a> Iterator for ChunkReader<'a> {
    type Item = Result<ChunkSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let Block { columns: [tokens, tags, chunks], source } = self.0.next_block()?;
        Some(ChunkSample::new(tokens, tags, chunks, source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lemma_blocks_and_corrupt_lines() {
        let src = "He\tPRP\the\nruns\tVBZ\n \nfast\tRB\tfast\t\n\nx\ty\tz\n";
        let mut r = LemmaReader::new(src).unwrap();
        let samples: Vec<LemmaSample<'_>> = r.by_ref().map(Result::unwrap).collect();
        assert_eq!(samples.len(), 2);
        let lemmas: Vec<&str> = samples[0].lemmas().iter().map(|f| f.as_str()).collect();
        assert_eq!(lemmas, ["he", "fast"]);
        assert_eq!(samples[0].to_string(), "He\tPRP\the\nfast\tRB\tfast\n");
        assert_eq!(&src[samples[0].source().range()], "He\tPRP\the\nruns\tVBZ\n \nfast\tRB\tfast\t");
        assert_eq!(
            r.warnings(),
            [Warning::CorruptLine { line: 2, columns: 2 }, Warning::CorruptLine { line: 3, columns: 1 }]
        );
    }

    #[test]
    fn two_empty_lines_end_the_stream() {
        let src = "a b B-NP\n\n\nc d B-NP\n";
        let samples: Vec<_> = ChunkReader::new(src).unwrap().collect();
        assert_eq!(samples.len(), 1);
        let mut r = ChunkReader::new("\na b O\n").unwrap();
        assert!(r.next().is_none());
        assert!(r.next().is_none());
    }

    #[test]
    fn chunk_columns_split_on_single_spaces() {
        let src = "a  b O\nc d O \ne\tf g O\n";
        let mut r = ChunkReader::new(src).unwrap();
        let s = r.next().unwrap().unwrap();
        let tokens: Vec<&str> = s.tokens().iter().map(|f| f.as_str()).collect();
        assert_eq!(tokens, ["c", "e\tf"]);
        assert_eq!(r.warnings(), [Warning::CorruptLine { line: 1, columns: 4 }]);
    }
}
