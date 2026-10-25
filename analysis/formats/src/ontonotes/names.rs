//! `OntoNotesNameSampleStream`: the name samples of `.name` documents.

use std::collections::HashMap;
use std::ops::Range;

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;
use turbo_unicode::java::lowercase_code_points;

use super::OntoNotesError;
use crate::sample::{Name, NameSample};
use crate::source::{Field, Lines, Piece, whitespace_tokens};

const DOC_OPEN: &str = "<DOC";
const DOC_CLOSE: &str = "</DOC>";
const ENAMEX_OPEN: &str = "<ENAMEX";
const ENAMEX_CLOSE: &str = "</ENAMEX>";
const TYPE: &str = "TYPE=\"";

/// The tokens OpenNLP replaces once their tags are cut off.
fn converted(token: &str) -> Option<&'static str> {
    Some(match token {
        "-LRB-" => "(",
        "-RRB-" => ")",
        "-LSB-" => "[",
        "-RSB-" => "]",
        "-LCB-" => "{",
        "-RCB-" => "}",
        "-AMP-" => "&",
        _ => return None,
    })
}

/// A name read from a line: its tokens, the index of its type in the
/// document's types, and the bytes from its start tag to its end tag.
#[derive(Clone, Copy, Debug)]
struct RawName {
    start: u32,
    end: u32,
    kind: Option<u32>,
    source: Span,
}

/// A sample read from a line: its tokens and names as ranges of the
/// document's.
#[derive(Clone, Debug)]
struct RawSample {
    tokens: Range<usize>,
    names: Range<usize>,
    clear_adaptive_data: bool,
    source: Span,
}

/// One OntoNotes `.name` document read as `OntoNotesNameSampleStream`
/// reads it: a name sample per line, and the error that stopped the
/// reading, if one did.
///
/// OpenNLP reads a whole document before it returns its first sample.
/// The samples are made by [`NameDocument::samples`]; the document keeps
/// the lower-cased name types they point to, which are not the source's
/// bytes.
#[derive(Debug)]
pub struct NameDocument<'a> {
    tokens: Vec<Field<'a>>,
    names: Vec<RawName>,
    samples: Vec<RawSample>,
    types: Vec<String>,
    error: Option<OntoNotesError>,
}

impl<'a> NameDocument<'a> {
    /// Reads `source`, splitting each line into tokens at the whitespace
    /// of `mode` (`WhitespaceTokenizer`).
    ///
    /// Each line that does not start with `<DOC` is a sample, up to a line
    /// that is exactly `</DOC>`; the first sample clears adaptive data. An
    /// error on a line stops the reading and is kept
    /// ([`NameDocument::error`]) with the samples of the lines before it.
    pub fn read(source: &'a str, mode: WhitespaceMode) -> NameDocument<'a> {
        let mut doc =
            NameDocument { tokens: Vec::new(), names: Vec::new(), samples: Vec::new(), types: Vec::new(), error: None };
        let lines = match Lines::new(source) {
            Ok(lines) => lines,
            Err(_) => {
                doc.error = Some(OntoNotesError::TooLong { len: source.len() });
                return doc;
            }
        };
        let mut type_ids: HashMap<String, u32> = HashMap::new();
        let mut clear_adaptive_data = true;
        for line in lines {
            if line.text().starts_with(DOC_OPEN) {
                continue;
            }
            if line.text() == DOC_CLOSE {
                break;
            }
            let tokens_start = doc.tokens.len();
            let names_start = doc.names.len();
            if let Err(e) = doc.read_line(line.piece, line.number, mode, &mut type_ids) {
                doc.tokens.truncate(tokens_start);
                doc.names.truncate(names_start);
                doc.error = Some(e);
                break;
            }
            doc.samples.push(RawSample {
                tokens: tokens_start..doc.tokens.len(),
                names: names_start..doc.names.len(),
                clear_adaptive_data,
                source: line.span(),
            });
            clear_adaptive_data = false;
        }
        doc
    }

    /// Reads the tokens and names of one line onto the document's.
    fn read_line(
        &mut self,
        line: Piece<'a>,
        number: u32,
        mode: WhitespaceMode,
        type_ids: &mut HashMap<String, u32>,
    ) -> Result<(), OntoNotesError> {
        let mut token_index = 0u32;
        let mut begin: Option<u32> = None;
        let mut kind: Option<u32> = None;
        let mut inside_start_tag = false;
        let mut tag_start = line.span.start;
        for token in whitespace_tokens(line, mode) {
            let text = token.text;
            if text.starts_with(ENAMEX_OPEN) {
                inside_start_tag = true;
                tag_start = token.span.start;
                continue;
            }
            if inside_start_tag {
                if let Some(rest) = text.strip_prefix(TYPE) {
                    let close = rest.find('"').ok_or_else(|| OntoNotesError::UnclosedType {
                        line: number,
                        token: token.span,
                        len: text.encode_utf16().count(),
                    })?;
                    let lower: String = lowercase_code_points(&rest[..close]);
                    let next = self.types.len() as u32;
                    let id = *type_ids.entry(lower).or_insert_with_key(|k| {
                        self.types.push(k.clone());
                        next
                    });
                    kind = Some(id);
                }
                if text.contains('>') {
                    begin = Some(token_index);
                    inside_start_tag = false;
                } else {
                    continue;
                }
            }
            if text.ends_with(ENAMEX_CLOSE) {
                let start = begin.ok_or(OntoNotesError::EndWithoutStart { line: number, token: token.span })?;
                let source = Span::new(tag_start, token.span.end);
                self.names.push(RawName { start, end: token_index + 1, kind, source });
                begin = None;
            }
            self.tokens.push(convert_token(token, number)?);
            token_index += 1;
        }
        Ok(())
    }

    /// The samples, in the order of their lines.
    pub fn samples(&self) -> impl ExactSizeIterator<Item = NameSample<'_>> + '_ {
        self.samples.iter().map(|s| self.sample(s))
    }

    fn sample(&self, s: &RawSample) -> NameSample<'_> {
        let names = self.names[s.names.clone()]
            .iter()
            .map(|n| Name {
                start: n.start,
                end: n.end,
                kind: n.kind.map(|k| self.types[k as usize].as_str()),
                source: n.source,
            })
            .collect();
        // A line's names follow each other without overlapping: a name
        // starts at the token after its start tag, which comes after the
        // last token of the name before it, and an end tag with no name
        // open is an error. So NameSample's checks always pass.
        NameSample::from_ordered_names(self.tokens[s.tokens.clone()].to_vec(), names, s.clear_adaptive_data, s.source)
    }

    /// The number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The error that stopped the reading, after the samples.
    pub fn error(&self) -> Option<&OntoNotesError> {
        self.error.as_ref()
    }

    /// The distinct name types of the document, lower-cased, in the order
    /// they first appear.
    pub fn types(&self) -> &[String] {
        &self.types
    }
}

/// `convertToken`: the token with a start tag cut off its front when it
/// holds `="` and a `>`, then the part from its first `<` to its first `>`
/// cut out, then `-LRB-` and the other bracket escapes and `-AMP-`
/// replaced.
fn convert_token<'a>(token: Piece<'a>, line: u32) -> Result<Field<'a>, OntoNotesError> {
    let text = token.text;
    let from = match text.find('>') {
        Some(gt) if text.contains("=\"") => gt + 1,
        _ => 0,
    };
    let rest = &text[from..];
    let mut kept = token.slice(from, text.len());
    let mut joined: Option<String> = None;
    if let (Some(lt), Some(gt)) = (rest.find('<'), rest.find('>')) {
        let cut_end = gt + 1;
        if lt > cut_end {
            let units = |s: &str| s.encode_utf16().count();
            return Err(OntoNotesError::TagOrder {
                line,
                token: token.span,
                start: units(&rest[..lt]),
                end: units(&rest[..cut_end]),
                len: units(rest),
            });
        }
        if lt < cut_end {
            if lt == 0 {
                kept = kept.slice(cut_end, rest.len());
            } else if cut_end == rest.len() {
                kept = kept.slice(0, lt);
            } else {
                joined = Some([&rest[..lt], &rest[cut_end..]].concat());
            }
        }
    }
    let cleaned = joined.as_deref().unwrap_or(kept.text);
    if let Some(replaced) = converted(cleaned) {
        return Ok(Field::made(replaced.to_owned(), token.span));
    }
    Ok(match joined {
        Some(text) => Field::made(text, token.span),
        None => kept.field(),
    })
}

/// The samples of OntoNotes name documents read one after another, as
/// `OntoNotesNameSampleStream` reads them over the files of the factory.
///
/// A document that stopped on an error gives the error first and then the
/// samples of the lines before it, as OpenNLP throws from the read that
/// reads the document and returns those samples from the reads after it.
/// A document that gives no sample and no error ends the stream: OpenNLP
/// returns null for it, and the documents after it are not read.
#[derive(Clone, Debug)]
pub struct NameStream<'d, 'a> {
    documents: std::slice::Iter<'d, NameDocument<'a>>,
    current: Option<(&'d NameDocument<'a>, usize)>,
    done: bool,
}

impl<'d, 'a> NameStream<'d, 'a> {
    pub fn new(documents: &'d [NameDocument<'a>]) -> NameStream<'d, 'a> {
        NameStream { documents: documents.iter(), current: None, done: false }
    }
}

impl<'d> Iterator for NameStream<'d, '_> {
    type Item = Result<NameSample<'d>, &'d OntoNotesError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((doc, next)) = &mut self.current {
                if let Some(raw) = doc.samples.get(*next) {
                    *next += 1;
                    return Some(Ok(doc.sample(raw)));
                }
                self.current = None;
            }
            if self.done {
                return None;
            }
            let Some(doc) = self.documents.next() else {
                self.done = true;
                return None;
            };
            self.current = Some((doc, 0));
            match doc.error() {
                Some(e) => return Some(Err(e)),
                None if doc.is_empty() => {
                    self.done = true;
                    return None;
                }
                None => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(source: &str) -> (Vec<String>, Option<(String, &'static str)>) {
        let doc = NameDocument::read(source, WhitespaceMode::Unicode);
        let samples = doc.samples().map(|s| s.to_string()).collect();
        let error = doc.error().map(|e| (e.to_string(), e.java_class().unwrap()));
        (samples, error)
    }

    #[test]
    fn reads_names_and_skips_doc_lines() {
        let doc = "<DOC DOCNO=\"x\">\n<ENAMEX TYPE=\"PERSON\">John Smith</ENAMEX> met \
                   <ENAMEX TYPE=\"GPE\" S_OFF=\"1\">-LRB-Paris-RRB-</ENAMEX> .\n\nnext\n</DOC>\nafter\n";
        let (samples, error) = read(doc);
        assert_eq!(error, None);
        assert_eq!(samples, ["\n<START:person> John Smith <END> met <START:gpe> -LRB-Paris-RRB- <END> .", "", "next "]);
        let d = NameDocument::read(doc, WhitespaceMode::Unicode);
        let first = d.samples().next().unwrap();
        let toks: Vec<&str> = first.tokens().iter().map(Field::as_str).collect();
        assert_eq!(toks, ["John", "Smith", "met", "-LRB-Paris-RRB-", "."]);
        assert!(first.tokens().iter().all(Field::is_borrowed));
        assert_eq!(&doc[first.tokens()[0].span.range()], "John");
        assert_eq!(&doc[first.names()[0].source.range()], "<ENAMEX TYPE=\"PERSON\">John Smith</ENAMEX>");
        assert_eq!(d.types(), ["person", "gpe"]);
    }

    #[test]
    fn converts_tokens_as_java() {
        let piece = |s: &'static str| Piece { text: s, span: Span::new(0, s.len() as u32) };
        let c = |s: &'static str| convert_token(piece(s), 1).map(|f| (f.as_str().to_owned(), f.is_borrowed()));
        assert_eq!(c("TYPE=\"PER\">-LRB-").unwrap(), ("(".to_owned(), false));
        assert_eq!(c("a<b>c").unwrap(), ("ac".to_owned(), false));
        assert_eq!(c("x</ENAMEX>").unwrap(), ("x".to_owned(), true));
        assert_eq!(c("a><b").unwrap(), ("a><b".to_owned(), true));
        assert_eq!(c("TYPE=\"P\">").unwrap(), (String::new(), true));
        let e = c("a>bc<d").unwrap_err();
        assert_eq!(e.to_string(), "Range [4, 2) out of bounds for length 6");
    }

    #[test]
    fn errors_keep_the_samples_before_them() {
        let (samples, error) = read("a b\nx</ENAMEX>\nc\n");
        assert_eq!(samples, ["\na b"]);
        assert_eq!(
            error,
            Some(("start index must be zero or greater: -1".to_owned(), "java.lang.IllegalArgumentException"))
        );
        let (_, error) = read("<ENAMEX TYPE=\"PER>x</ENAMEX>");
        assert_eq!(error.unwrap().0, "Range [6, -1) out of bounds for length 20");
    }

    #[test]
    fn a_type_carries_over_and_open_names_are_dropped() {
        let (samples, _) =
            read("<ENAMEX TYPE=\"ORG\">a</ENAMEX> <ENAMEX>b</ENAMEX> <ENAMEX S=\"1\">c</ENAMEX> <ENAMEX TYPE=\"X\">d");
        assert_eq!(samples, ["\n<START:org> a <END> <START:org> c <END> d"]);
    }

    #[test]
    fn the_stream_orders_errors_and_stops_at_an_empty_document() {
        let docs = [
            NameDocument::read("a\nb</ENAMEX>", WhitespaceMode::Unicode),
            NameDocument::read("c", WhitespaceMode::Unicode),
            NameDocument::read("<DOC>\n</DOC>", WhitespaceMode::Unicode),
            NameDocument::read("d", WhitespaceMode::Unicode),
        ];
        let items: Vec<Result<String, String>> =
            NameStream::new(&docs).map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string())).collect();
        assert_eq!(
            items,
            [Err("start index must be zero or greater: -1".to_owned()), Ok("\na ".to_owned()), Ok("\nc ".to_owned())]
        );
    }
}
