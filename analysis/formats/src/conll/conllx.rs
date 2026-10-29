//! The CoNLL-X part-of-speech reader.

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::sample::PosSample;
use crate::source::{Line, Lines, split_fields};

/// The columns a CoNLL-X line needs: id, form, lemma, coarse and fine
/// part of speech.
const MIN_COLUMNS: usize = 5;

/// Reads part-of-speech samples from CoNLL-X (`ConllXPOSSampleStream`): a
/// sentence is a paragraph of lines (`ParagraphStream`), each line a word
/// in tab-separated columns, of which the form (the second) is the token
/// and the fine part of speech (the fifth) the tag. Columns after the
/// fifth are not read. With the tabs shown as spaces:
///
/// ```text
/// 1  Cathy  Cathy  N  NNP  _  2  SUB
/// 2  zag    zien   V  VBD  _  0  ROOT
/// ```
///
/// A line with fewer than five columns, after `StringUtil.split` drops
/// the empty columns at its end, fails the whole sentence
/// ([`FormatError::ConllXColumns`]), and the next read starts at the next
/// sentence.
#[derive(Clone, Debug)]
pub struct ConllXPosReader<'a> {
    lines: Lines<'a>,
}

impl<'a> ConllXPosReader<'a> {
    /// A reader of `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(source: &'a str) -> Result<ConllXPosReader<'a>, FormatError> {
        Ok(ConllXPosReader { lines: Lines::new(source)? })
    }

    /// The next paragraph: the lines up to a line of length zero or the end,
    /// skipping the empty lines before it; `None` at the end.
    fn paragraph(&mut self) -> Option<Vec<Line<'a>>> {
        let mut lines = Vec::new();
        loop {
            match self.lines.next() {
                Some(line) if !line.text().is_empty() => lines.push(line),
                Some(_) if lines.is_empty() => {}
                Some(_) => return Some(lines),
                None if lines.is_empty() => return None,
                None => return Some(lines),
            }
        }
    }
}

impl<'a> Iterator for ConllXPosReader<'a> {
    type Item = Result<PosSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let lines = self.paragraph()?;
        let mut tokens = Vec::with_capacity(lines.len());
        let mut tags = Vec::with_capacity(lines.len());
        for line in &lines {
            let columns = split_fields(line.piece, '\t');
            if columns.len() < MIN_COLUMNS {
                return Some(Err(FormatError::ConllXColumns { line: line.number, text: line.text().to_owned() }));
            }
            tokens.push(columns[1].field());
            tags.push(columns[4].field());
        }
        let source = Span::new(lines[0].span().start, lines[lines.len() - 1].span().end);
        Some(PosSample::new(tokens, tags, source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(source: &str) -> Vec<Result<String, String>> {
        ConllXPosReader::new(source).unwrap().map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string())).collect()
    }

    #[test]
    fn reads_the_form_and_the_fine_tag() {
        let text = "\n\n1\tCathy\tCathy\tN\tNNP\t_\t2\tSUB\n2\tzag\tzien\tV\tVBD\n\n\n\n1\tHij\thij\tPron\tPRP\t_\r\n";
        assert_eq!(read(text), [Ok("Cathy_NNP zag_VBD".to_owned()), Ok("Hij_PRP".to_owned())]);
    }

    #[test]
    fn a_short_line_fails_its_sentence() {
        let text = "1\ta\ta\tX\tY\n2\tb\tb\tX\t\t\t\n3\tc\tc\tX\tY\n\n1\td\td\tX\tZ";
        assert_eq!(
            read(text),
            [
                Err("Every non-empty line must have at least 5 fields: '2\tb\tb\tX\t\t\t'!".to_owned()),
                Ok("d_Z".to_owned()),
            ]
        );
    }

    #[test]
    fn empty_columns_inside_a_line_count() {
        assert_eq!(
            read("1\t\t\t\t\n"),
            [Err("Every non-empty line must have at least 5 fields: '1\t\t\t\t'!".to_owned())]
        );
        assert_eq!(read("1\t\t\t\tT\n"), [Ok("_T".to_owned())]);
    }

    #[test]
    fn tokens_and_tags_are_fields_of_the_source() {
        let text = "1\tCathy\tCathy\tN\tNNP\t_\n";
        let sample = ConllXPosReader::new(text).unwrap().next().unwrap().unwrap();
        assert_eq!(&text[sample.tokens()[0].span.range()], "Cathy");
        assert_eq!(&text[sample.tags()[0].span.range()], "NNP");
        assert_eq!(&text[sample.source().range()], "1\tCathy\tCathy\tN\tNNP\t_");
    }
}
