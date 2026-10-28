//! The lemma and chunk formats: one token per line in three columns, a
//! sentence ending at an empty line. The lemma format separates its
//! columns (token, tag, lemma) with a tab; the chunk format (token, tag,
//! chunk tag) with a single space. A line that does not give exactly
//! three columns is skipped; a sentence with no valid line ends the read.

use turbo_analysis::Span;

use super::split_fields;
use turbo_formats::sample::{ChunkSample, LemmaSample};
use turbo_formats::{Field, FormatError, Lines};

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
    done: bool,
}

impl<'a> Blocks<'a> {
    fn new(source: &'a str, separator: char) -> Result<Blocks<'a>, FormatError> {
        Ok(Blocks { lines: Lines::new(source)?, separator, done: false })
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
    /// A reader of `source`, separating columns with a tab.
    pub fn new(source: &'a str) -> Result<LemmaReader<'a>, FormatError> {
        Ok(LemmaReader(Blocks::new(source, '\t')?))
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
    /// A reader of `source`, separating columns with a single space.
    pub fn new(source: &'a str) -> Result<ChunkReader<'a>, FormatError> {
        Ok(ChunkReader(Blocks::new(source, ' ')?))
    }
}

impl<'a> Iterator for ChunkReader<'a> {
    type Item = Result<ChunkSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let Block { columns: [tokens, tags, chunks], source } = self.0.next_block()?;
        Some(ChunkSample::new(tokens, tags, chunks, source))
    }
}
