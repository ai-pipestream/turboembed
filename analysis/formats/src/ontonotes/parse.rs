//! `OntoNotesParseSampleStream` and the part-of-speech samples
//! `ParseToPOSSampleStream` makes of its parses.

use turbo_analysis::Span;
use turbo_parser::{Parse, TokenSource};

use super::{OntoNotesError, document_lines};
use crate::sample::PosSample;
use crate::source::{Field, Piece, java_trim};

/// A parse read from an OntoNotes `.parse` document, with where it came
/// from.
#[derive(Clone, Debug)]
pub struct OntoNotesParse {
    parse: Parse,
    document: usize,
    source: Span,
    tokens: Vec<TokenSource>,
}

impl OntoNotesParse {
    /// The parse, as `Parse.parseParse` reads the sentence's lines joined
    /// with spaces.
    pub fn parse(&self) -> &Parse {
        &self.parse
    }

    pub fn into_parse(self) -> Parse {
        self.parse
    }

    /// The index of the document the parse was read from, counting from 0.
    pub fn document(&self) -> usize {
        self.document
    }

    /// The bytes of the sentence's lines in the document, from the first
    /// to the last, trimmed.
    pub fn source(&self) -> Span {
        self.source
    }

    /// Where each token of the parse came from, in the order of the
    /// parse's text: its span in that text, and the bytes in the document
    /// of the token as written and of the type of the bracket around it.
    pub fn tokens(&self) -> &[TokenSource] {
        &self.tokens
    }

    /// The part-of-speech sample `ParseToPOSSampleStream` makes of the
    /// parse: the covered text and type of each tag node
    /// (`Parse.getTagNodes`). Each token and tag has a text of its own and
    /// the span in the document of the token as written or of its
    /// bracket's type.
    pub fn pos_sample(&self) -> PosSample<'static> {
        let pairs = self.parse.tag_nodes().into_iter().map(|node| {
            let at = node.span().start;
            let source = self.tokens.binary_search_by_key(&at, |t| t.text.start).ok().map(|i| self.tokens[i]);
            let (token, label) = source.map_or((self.source, self.source), |s| (s.token, s.label));
            (Field::made(node.covered_text().to_owned(), token), Field::made(node.kind().to_owned(), label))
        });
        PosSample::from_pairs(pairs, self.source)
    }
}

/// The parses of OntoNotes `.parse` documents read one after another, as
/// `OntoNotesParseSampleStream` reads them over a `DocumentToLineStream`
/// of the factory's files.
///
/// Each line ([`document_lines`]) is trimmed; the lines up to an empty one
/// are joined, each followed by a space, and read with `Parse.parseParse`
/// (no function tags, no gap labeler). An empty line with no line before
/// it ends the stream, so do two empty lines in a row and an empty
/// document. A sentence that cannot be read is an error, and the stream
/// goes on after it; so does a document of nothing but line feeds
/// ([`OntoNotesError::NoLines`]).
pub struct ParseStream<'a, I> {
    documents: I,
    document: Option<usize>,
    lines: std::vec::IntoIter<Piece<'a>>,
    done: bool,
}

impl<'a, I: Iterator<Item = &'a str>> ParseStream<'a, I> {
    pub fn new(documents: impl IntoIterator<IntoIter = I>) -> ParseStream<'a, I> {
        ParseStream { documents: documents.into_iter(), document: None, lines: Vec::new().into_iter(), done: false }
    }

    /// The next line of the documents, reading the next document when the
    /// lines of one run out.
    fn next_line(&mut self) -> Option<Result<Piece<'a>, OntoNotesError>> {
        loop {
            if let Some(line) = self.lines.next() {
                return Some(Ok(line));
            }
            let document = self.documents.next()?;
            self.document = Some(self.document.map_or(0, |d| d + 1));
            match document_lines(document) {
                Ok(lines) => self.lines = lines.into_iter(),
                Err(e) => return Some(Err(e)),
            }
        }
    }

    fn read(&self, text: &str, segments: &[(u32, Span)]) -> Result<OntoNotesParse, OntoNotesError> {
        let document = self.document.unwrap_or(0);
        let source = Span::new(segments[0].1.start, segments[segments.len() - 1].1.end);
        let (parse, mut tokens) = Parse::parse_with_sources(text, false).map_err(|error| OntoNotesError::Parse {
            document,
            sentence: source,
            error,
        })?;
        // A token or type is inside one line: the lines are joined with
        // spaces, which end both.
        let to_document = |span: Span| {
            let i = segments.partition_point(|&(at, _)| at <= span.start).saturating_sub(1);
            let (at, line) = segments[i];
            Span::new(line.start + (span.start - at), line.start + (span.end - at))
        };
        for t in &mut tokens {
            t.token = to_document(t.token);
            t.label = to_document(t.label);
        }
        Ok(OntoNotesParse { parse, document, source, tokens })
    }
}

impl<'a, I: Iterator<Item = &'a str>> Iterator for ParseStream<'a, I> {
    type Item = Result<OntoNotesParse, OntoNotesError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let mut text = String::new();
        // Where each line starts in `text`, and its bytes in the document.
        let mut segments: Vec<(u32, Span)> = Vec::new();
        loop {
            let line = match self.next_line() {
                Some(Ok(line)) => Some(java_trim(line)),
                Some(Err(e)) => return Some(Err(e)),
                None => None,
            };
            match line {
                Some(line) if !line.text.is_empty() => {
                    let Ok(at) = u32::try_from(text.len()) else {
                        return Some(Err(OntoNotesError::TooLong { len: text.len() }));
                    };
                    segments.push((at, line.span));
                    text.push_str(line.text);
                    text.push(' ');
                }
                _ if !text.is_empty() => return Some(self.read(&text, &segments)),
                _ => {
                    self.done = true;
                    return None;
                }
            }
        }
    }
}

/// The part-of-speech samples of the parses of a [`ParseStream`]
/// ([`OntoNotesParse::pos_sample`]), as `OntoNotesPOSSampleStreamFactory`
/// reads them.
pub struct PosStream<'a, I> {
    parses: ParseStream<'a, I>,
}

impl<'a, I: Iterator<Item = &'a str>> PosStream<'a, I> {
    pub fn new(documents: impl IntoIterator<IntoIter = I>) -> PosStream<'a, I> {
        PosStream { parses: ParseStream::new(documents) }
    }
}

impl<'a, I: Iterator<Item = &'a str>> Iterator for PosStream<'a, I> {
    type Item = Result<PosSample<'static>, OntoNotesError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.parses.next()?.map(|p| p.pos_sample()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(docs: &[&str]) -> Vec<Result<String, String>> {
        ParseStream::new(docs.iter().copied())
            .map(|r| r.map(|p| p.parse().to_string_penn_treebank()).map_err(|e| e.to_string()))
            .collect()
    }

    #[test]
    fn sentences_run_over_lines_and_documents_end_them() {
        let doc = "(TOP (S (NP (NNP John))\n   (VP (VBD ran))))\n\n(TOP (NN x))";
        assert_eq!(
            read(&[doc]),
            [Ok("(TOP (S (NP (NNP John)) (VP (VBD ran))) )".to_owned()), Ok("(TOP (NN x) )".to_owned())]
        );
        assert_eq!(read(&["(TOP (NN a))", "(TOP (NN b))\n"]).len(), 2);
    }

    #[test]
    fn an_empty_line_without_a_sentence_ends_the_stream() {
        assert_eq!(read(&["(TOP (NN a))\n\n\n(TOP (NN b))"]).len(), 1);
        assert_eq!(read(&["", "(TOP (NN b))"]).len(), 0);
        assert_eq!(
            read(&["\n\n", "(TOP (NN b))"]),
            [Err("Index -1 out of bounds for length 0".to_owned()), Ok("(TOP (NN b) )".to_owned())]
        );
    }

    #[test]
    fn tokens_map_back_to_the_document() {
        let doc = "  (TOP (S (NP (-LRB- -LRB-) (NN dog))\n (VP (VB runs))))\n";
        let p = ParseStream::new([doc]).next().unwrap().unwrap();
        assert_eq!(&doc[p.source().range()], "(TOP (S (NP (-LRB- -LRB-) (NN dog))\n (VP (VB runs))))");
        let pos = p.pos_sample();
        let toks: Vec<(&str, &str, &str, &str)> = pos
            .tokens()
            .iter()
            .zip(pos.tags())
            .map(|(t, g)| (t.as_str(), &doc[t.span.range()], g.as_str(), &doc[g.span.range()]))
            .collect();
        assert_eq!(toks, [("(", "-LRB-", "-LRB-", "-LRB-"), ("dog", "dog", "NN", "NN"), ("runs", "runs", "VB", "VB")]);
        assert_eq!(pos.to_string(), "(_-LRB- dog_NN runs_VB");
    }
}
