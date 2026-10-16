//! `DocumentToLineStream`: a document cut into lines for the parse
//! reader.

use turbo_analysis::Span;

use super::OntoNotesError;
use crate::source::{Piece, java_trim, split_fields};

/// The lines `DocumentToLineStream` makes of a document:
/// `StringUtil.split(document, '\n')`, which drops the empty lines at the
/// end, and an empty line after them when the last line left is not blank
/// (`trim` leaves something), so every document ends with a line that
/// ends a sentence. A carriage return stays in its line. An empty document
/// is one empty line. The added line is empty, at the end of the document.
///
/// # Errors
///
/// [`OntoNotesError::NoLines`] when the document is nothing but line feeds,
/// so no line is left, and [`OntoNotesError::TooLong`] when it is too long
/// for spans.
pub fn document_lines(document: &str) -> Result<Vec<Piece<'_>>, OntoNotesError> {
    let len = u32::try_from(document.len()).map_err(|_| OntoNotesError::TooLong { len: document.len() })?;
    let mut lines = split_fields(Piece { text: document, span: Span::new(0, len) }, '\n');
    let last = lines.last().ok_or(OntoNotesError::NoLines)?;
    if !java_trim(*last).text.is_empty() {
        lines.push(Piece { text: &document[document.len()..], span: Span::new(len, len) });
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(s: &str) -> Result<Vec<&str>, String> {
        document_lines(s).map(|l| l.iter().map(|p| p.text).collect()).map_err(|e| e.to_string())
    }

    #[test]
    fn documents_end_with_an_empty_line() {
        assert_eq!(lines(""), Ok(vec![""]));
        assert_eq!(lines("(A)"), Ok(vec!["(A)", ""]));
        assert_eq!(lines("(A)\n\n\n"), Ok(vec!["(A)", ""]));
        assert_eq!(lines("(A)\n \n"), Ok(vec!["(A)", " "]));
        assert_eq!(lines("(A)\r\n(B)\r\n"), Ok(vec!["(A)\r", "(B)\r", ""]));
        assert_eq!(lines("\n(A)\n\n(B)"), Ok(vec!["", "(A)", "", "(B)", ""]));
        assert_eq!(lines("\n\n"), Err("Index -1 out of bounds for length 0".to_owned()));
        let spans: Vec<Span> = document_lines("a\nbc").unwrap().iter().map(|p| p.span).collect();
        assert_eq!(spans, [Span::new(0, 1), Span::new(2, 4), Span::new(4, 4)]);
    }
}
