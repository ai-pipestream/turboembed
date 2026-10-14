//! The MUC-6 reader: `MucNameSampleStream` with its `SgmlParser` and
//! `MucNameContentHandler`.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use super::{NerError, TokenSpans};
use crate::error::FormatError;
use crate::sample::{Name, NameSample};
use crate::source::{Field, check_len};

/// The registry name of the MUC-6 format.
pub const MUC6_FORMAT: &str = "muc6";

/// The `TYPE` values a name element may have, in the order
/// `MucNameContentHandler` lists them.
pub const MUC_TYPES: [&str; 7] = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT"];

/// The elements that are names (`ENAMEX`, `TIMEX`, `NUMEX`).
const NAME_ELEMENTS: [&str; 3] = ["ENAMEX", "TIMEX", "NUMEX"];

/// The elements whose text is read, each a sample (`MucElementNames.CONTENT_ELEMENTS`).
const CONTENT_ELEMENTS: [&str; 4] = ["HL", "DATELINE", "DD", "s"];

/// The element that starts a document (`MucElementNames.DOC_ELEMENT`).
const DOC_ELEMENT: &str = "DOC";

/// The name type of an expected `TYPE`: the value lower-cased.
fn name_type(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "PERSON" => "person",
        "ORGANIZATION" => "organization",
        "LOCATION" => "location",
        "DATE" => "date",
        "TIME" => "time",
        "MONEY" => "money",
        "PERCENT" => "percent",
        _ => return None,
    })
}

/// The source with every `>>` replaced by `>`, as `MucNameSampleStream`
/// replaces them before parsing, and where each replacement is.
struct Collapsed {
    text: String,
    /// The position in `text` of each `>` that stands for `>>`.
    pairs: Vec<usize>,
}

impl Collapsed {
    fn new(source: &str) -> Collapsed {
        let mut text = String::with_capacity(source.len());
        let mut pairs = Vec::new();
        let mut rest = source;
        while let Some(i) = rest.find(">>") {
            text.push_str(&rest[..i]);
            pairs.push(text.len());
            text.push('>');
            rest = &rest[i + 2..];
        }
        text.push_str(rest);
        Collapsed { text, pairs }
    }

    /// The source offset of offset `at` of the text.
    fn source(&self, at: usize) -> u32 {
        (at + self.pairs.partition_point(|&p| p < at)) as u32
    }
}

/// The line numbers of a source, as `BufferedReader.readLine` counts
/// lines: a line ends at `\n`, `\r` or `\r\n`.
struct LineStarts(Vec<u32>);

impl LineStarts {
    fn new(source: &str) -> LineStarts {
        let bytes = source.as_bytes();
        let mut starts = vec![0];
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'\r' if bytes.get(i + 1) == Some(&b'\n') => {
                    i += 2;
                    starts.push(i as u32);
                }
                b'\r' | b'\n' => {
                    i += 1;
                    starts.push(i as u32);
                }
                _ => i += 1,
            }
        }
        LineStarts(starts)
    }

    /// The number, from 1, of the line holding source offset `at`.
    fn line(&self, at: u32) -> u32 {
        self.0.partition_point(|&s| s <= at) as u32
    }
}

/// The tag name `SgmlParser.extractTagName` takes from a tag: the text
/// after `<` or `</` up to the first `>` or whitespace.
fn tag_name(tag: &str, mode: WhitespaceMode) -> &str {
    let from = if tag.as_bytes().get(1) == Some(&b'/') { 2 } else { 1 };
    for (i, c) in tag.char_indices().skip(1) {
        if c == '>' || mode.is_whitespace(c as u32) {
            return &tag[from..i];
        }
    }
    // the parser hands over a tag only at its `>`, so the loop returns;
    // OpenNLP's "Failed to extract tag name!" cannot be reached
    unreachable!("a tag ends with >")
}

/// The value of the `TYPE` attribute as `SgmlParser.getAttributes` reads
/// a tag: whitespace starts a key, which runs to `=` or whitespace; a
/// value is the text between two double quotes, and a later value of a
/// key replaces an earlier one.
fn type_attribute(tag: &str, mode: WhitespaceMode) -> Option<String> {
    let mut attributes = HashMap::new();
    let mut key = String::new();
    let mut value = String::new();
    let (mut in_key, mut in_value) = (false, false);
    for c in tag.chars() {
        let space = mode.is_whitespace(c as u32);
        if space && !in_value {
            in_key = true;
        } else if in_key && (c == '=' || space) {
            in_key = false;
        } else if in_key {
            key.push(c);
        } else if c == '"' {
            if in_value {
                attributes.insert(std::mem::take(&mut key), std::mem::take(&mut value));
            }
            in_value = !in_value;
        } else if in_value {
            value.push(c);
        }
    }
    attributes.remove("TYPE")
}

/// `MucNameContentHandler` and the source it reads.
struct Handler<'s, 'a, T: ?Sized> {
    source: &'a str,
    doc: &'s Collapsed,
    lines: &'s LineStarts,
    tokenizer: &'s T,
    spans: Vec<Span>,
    inside_content: bool,
    content_start: Option<u32>,
    text: Vec<Field<'a>>,
    clear_adaptive_data: bool,
    open_names: Vec<(u32, &'static str)>,
    names: Vec<Name<'a>>,
    samples: Vec<NameSample<'a>>,
}

impl<'a, T: TokenSpans + ?Sized> Handler<'_, 'a, T> {
    fn line(&self, at: usize) -> u32 {
        self.lines.line(self.doc.source(at))
    }

    fn start_element(&mut self, name: &str, tag: &str, at: usize, mode: WhitespaceMode) -> Result<(), FormatError> {
        if name == DOC_ELEMENT {
            self.clear_adaptive_data = true;
        }
        if CONTENT_ELEMENTS.contains(&name) {
            self.inside_content = true;
            if self.content_start.is_none() {
                self.content_start = Some(self.doc.source(at));
            }
        }
        if NAME_ELEMENTS.contains(&name) {
            let line = self.line(at);
            let kind = type_attribute(tag, mode).ok_or(NerError::MucMissingType { line })?;
            let kind = name_type(&kind).ok_or(NerError::MucUnknownType { line, kind })?;
            self.open_names.push((self.text.len() as u32, kind));
        }
        Ok(())
    }

    /// The trimmed text between two tags, which starts at offset `at`.
    fn characters(&mut self, chars: &str, at: usize) {
        if !self.inside_content {
            return;
        }
        self.spans.clear();
        self.tokenizer.token_spans(chars, &mut self.spans);
        for s in &self.spans {
            let token = &chars[s.range()];
            let span = Span::new(self.doc.source(at + s.start as usize), self.doc.source(at + s.end as usize));
            // text between tags holds no `>`, which the parser refuses
            // outside a tag, so no token spans a replaced `>>`
            debug_assert_eq!(&self.source[span.range()], token);
            self.text.push(Field::of(self.source, span));
        }
    }

    fn end_element(&mut self, name: &str, at: usize, end: usize) -> Result<(), FormatError> {
        if NAME_ELEMENTS.contains(&name) {
            let line = self.line(at);
            let (start, kind) = self.open_names.pop().ok_or(NerError::MucUnopenedName { line })?;
            self.names.push(Name::new(start, self.text.len() as u32, Some(kind))?);
        }
        if CONTENT_ELEMENTS.contains(&name) {
            let start = self.content_start.take().unwrap_or_else(|| self.doc.source(at));
            let source = Span::new(start, self.doc.source(end));
            let line = self.line(at);
            let text = std::mem::take(&mut self.text);
            let names = std::mem::take(&mut self.names);
            self.samples.push(NameSample::new(text, names, self.clear_adaptive_data, source, line)?);
            self.clear_adaptive_data = false;
            self.inside_content = false;
        }
        Ok(())
    }
}

/// `SgmlParser.parse` of the collapsed text into `handler`.
fn parse<T: TokenSpans + ?Sized>(handler: &mut Handler<'_, '_, T>, mode: WhitespaceMode) -> Result<(), FormatError> {
    let doc = &handler.doc.text;
    let mut buffer = 0;
    let mut inside_tag = false;
    let mut start_tag = true;
    let mut last = None;
    for (i, c) in doc.char_indices() {
        if c == '<' {
            if inside_tag {
                return Err(NerError::MucUnexpectedOpen { line: handler.line(i) }.into());
            }
            let text = &doc[buffer..i];
            let start = text.len() - text.trim_start_matches(|c| c <= ' ').len();
            let trimmed = text[start..].trim_end_matches(|c| c <= ' ');
            if !trimmed.is_empty() {
                handler.characters(trimmed, buffer + start);
            }
            buffer = i;
            inside_tag = true;
            start_tag = true;
        }
        if c == '/' && last == Some('<') {
            start_tag = false;
        }
        if c == '>' {
            if !inside_tag {
                return Err(NerError::MucUnexpectedClose { line: handler.line(i) }.into());
            }
            let tag = &doc[buffer..=i];
            let name = tag_name(tag, mode);
            if start_tag {
                handler.start_element(name, tag, buffer, mode)?;
            } else {
                handler.end_element(name, buffer, i + 1)?;
            }
            buffer = i + 1;
            inside_tag = false;
        }
        last = Some(c);
    }
    if inside_tag {
        return Err(NerError::MucUnclosedTag { line: handler.line(buffer) }.into());
    }
    Ok(())
}

/// Reads name samples from a MUC-6 document file (`MucNameSampleStream`,
/// which `Muc6NameSampleStreamFactory` gives each `.sgm` file of a
/// directory as one text): SGML in which the text of the `HL`,
/// `DATELINE`, `DD` and `s` elements is split into tokens with a
/// tokenizer, each element a sample, and `ENAMEX`, `TIMEX` and `NUMEX`
/// elements are names of their `TYPE`, lower-cased. A `DOC` element
/// clears adaptive data for the next sample.
///
/// ```text
/// <DOC>
/// <s> <ENAMEX TYPE="PERSON">Ada Lovelace</ENAMEX> wrote notes . </s>
/// </DOC>
/// ```
///
/// The file is parsed when the reader is made. What OpenNLP's stream does
/// with it:
///
/// - Every `>>` of the file is replaced by `>` before it is parsed, a fix
///   for tags such as `<s>>` in some MUC files. A `>` outside a tag is an
///   error ([`NerError::MucUnexpectedClose`]), so `a >> b` in the text is
///   one too.
/// - The text between two tags is trimmed (`String.trim`) and split on
///   its own, so a word with a tag inside it is two tokens. Text outside
///   the content elements is not read, nor is text after the last tag.
/// - Every end of a content element makes a sample, with no tokens if it
///   had none. A content element inside another makes its sample at its
///   end, and the text of the outer one after it is not read; the outer
///   one's end makes another sample.
/// - A name closed outside a content element joins the next sample; a
///   name open at the end of one ends in the next, which refuses it
///   ([`FormatError::BackwardsSpan`]) when it starts past that sample's
///   tokens so far, and otherwise holds a name over other tokens. Nested
///   names overlap, which `NameSample` refuses
///   ([`FormatError::OverlappingNames`]).
/// - A value is read only between double quotes, so `TYPE=PERSON` gives
///   a name element no type ([`NerError::MucMissingType`]).
/// - An error ends the parse of the file. OpenNLP's `read` throws it, and
///   the samples made before it are returned by the reads after: so this
///   reader gives the error first, then those samples.
#[derive(Debug)]
pub struct MucReader<'a> {
    items: std::vec::IntoIter<Result<NameSample<'a>, FormatError>>,
}

impl<'a> MucReader<'a> {
    /// A reader of the document file `source`, split into tokens by
    /// `tokenizer`; `whitespace` is what the SGML parser takes as
    /// whitespace in tags (`StringUtil.isWhitespace`).
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new<T: TokenSpans + ?Sized>(
        source: &'a str,
        tokenizer: &T,
        whitespace: WhitespaceMode,
    ) -> Result<MucReader<'a>, FormatError> {
        check_len(source)?;
        let doc = Collapsed::new(source);
        let lines = LineStarts::new(source);
        let mut handler = Handler {
            source,
            doc: &doc,
            lines: &lines,
            tokenizer,
            spans: Vec::new(),
            inside_content: false,
            content_start: None,
            text: Vec::new(),
            clear_adaptive_data: false,
            open_names: Vec::new(),
            names: Vec::new(),
            samples: Vec::new(),
        };
        let result = parse(&mut handler, whitespace);
        let mut items: Vec<_> = result.err().map(Err).into_iter().collect();
        items.extend(handler.samples.into_iter().map(Ok));
        Ok(MucReader { items: items.into_iter() })
    }
}

impl<'a> Iterator for MucReader<'a> {
    type Item = Result<NameSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.next()
    }
}

/// The files `Muc6NameSampleStreamFactory` reads from the directory
/// `dir`, in the order it reads them: the files (not the directories,
/// which it does not enter) whose name ends with `.sgm` in any case. The
/// factory's `DirectorySampleStream` sorts them by name, in Java's string
/// order, and pushes them on a stack, so they come in the reverse of that
/// order: `b.sgm` before `a.sgm`.
///
/// # Errors
///
/// What listing the directory or reading an entry's metadata gives.
pub fn muc_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        if name.ends_with(".sgm") && std::fs::metadata(&path)?.is_file() {
            files.push(path);
        }
    }
    files.sort_by(|a, b| turbo_jvm::chars::compare(&a.to_string_lossy(), &b.to_string_lossy()));
    files.reverse();
    Ok(files)
}

/// The samples of MUC document files read one after the other, as the
/// factory's stream reads them ([`muc_files`] gives the files, each
/// decoded as UTF-8 by the caller). OpenNLP's `read` returns null for a
/// file that gives no sample and no error, so the samples end at the
/// first such file, and the files after it are not read. A file that
/// gives an error and no sample does not end them.
pub struct MucSamples<'a, I, T: ?Sized> {
    documents: I,
    tokenizer: &'a T,
    whitespace: WhitespaceMode,
    current: Option<MucReader<'a>>,
    done: bool,
}

impl<'a, I: Iterator<Item = &'a str>, T: TokenSpans + ?Sized> MucSamples<'a, I, T> {
    /// The samples of `documents`, split into tokens by `tokenizer`.
    pub fn new(
        documents: impl IntoIterator<IntoIter = I>,
        tokenizer: &'a T,
        whitespace: WhitespaceMode,
    ) -> MucSamples<'a, I, T> {
        MucSamples { documents: documents.into_iter(), tokenizer, whitespace, current: None, done: false }
    }
}

impl<'a, I: Iterator<Item = &'a str>, T: TokenSpans + ?Sized> Iterator for MucSamples<'a, I, T> {
    type Item = Result<NameSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.current.as_mut().and_then(Iterator::next) {
            return Some(item);
        }
        if self.done {
            return None;
        }
        let Some(document) = self.documents.next() else {
            self.done = true;
            return None;
        };
        match MucReader::new(document, self.tokenizer, self.whitespace) {
            Err(e) => Some(Err(e)),
            Ok(mut reader) => {
                let first = reader.next();
                if first.is_none() {
                    self.done = true;
                }
                self.current = Some(reader);
                first
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::RuleTokenizer;

    fn read(source: &str) -> Vec<Result<String, String>> {
        MucReader::new(source, &RuleTokenizer::default(), WhitespaceMode::Unicode)
            .unwrap()
            .map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string()))
            .collect()
    }

    #[test]
    fn reads_content_elements_and_names() {
        let text = "<DOC><HL>Big <ENAMEX TYPE=\"ORGANIZATION\">Acme</ENAMEX> news</HL>\n<s> On <TIMEX TYPE=\"DATE\">Monday</TIMEX> . </s></DOC>";
        assert_eq!(
            read(text),
            [
                Ok("\nBig <START:organization> Acme <END> news".to_owned()),
                Ok("On <START:date> Monday <END> .".to_owned())
            ]
        );
    }

    #[test]
    fn text_outside_content_and_after_the_last_tag_is_not_read() {
        assert_eq!(read("lost <s>a b</s> lost <s></s> lost"), [Ok("a b".to_owned()), Ok("".to_owned())]);
    }

    #[test]
    fn double_closing_brackets_are_one() {
        // a `>` outside a tag is an error, so the replacement matters in
        // tags, and the spans after it still name the source
        let text = "<s>>a <ENAMEX TYPE=\"PERSON\">>b</ENAMEX></s>";
        let samples: Vec<_> = MucReader::new(text, &RuleTokenizer::default(), WhitespaceMode::Unicode)
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(samples[0].to_string(), "a <START:person> b <END>");
        let tokens = samples[0].tokens();
        assert_eq!(&text[tokens[1].span.range()], "b");
        assert!(tokens.iter().all(Field::is_borrowed));
        assert_eq!(&text[samples[0].source().range()], text);
        assert_eq!(read("<s>a >> b</s>"), [Err("Did not expect > char!".to_owned())]);
    }

    #[test]
    fn attributes_need_double_quotes() {
        assert_eq!(type_attribute("<ENAMEX TYPE=\"PERSON\">", WhitespaceMode::Unicode).as_deref(), Some("PERSON"));
        assert_eq!(type_attribute("<ENAMEX TYPE=PERSON>", WhitespaceMode::Unicode), None);
        assert_eq!(
            type_attribute("<ENAMEX TYPE=\"DATE\" TYPE=\"TIME\">", WhitespaceMode::Unicode).as_deref(),
            Some("TIME")
        );
        // the space after `=` starts another key, which takes the quote
        assert_eq!(type_attribute("<ENAMEX TYPE = \"MONEY\">", WhitespaceMode::Unicode), None);
        assert_eq!(type_attribute("<ENAMEX TYPE =\"MONEY\">", WhitespaceMode::Unicode).as_deref(), Some("MONEY"));
        assert_eq!(type_attribute("<ENAMEX\u{a0}TYPE=\"MONEY\">", WhitespaceMode::Unicode).as_deref(), Some("MONEY"));
        assert_eq!(type_attribute("<ENAMEX\u{a0}TYPE=\"MONEY\">", WhitespaceMode::Legacy).as_deref(), Some("MONEY"));
        assert_eq!(type_attribute("<ENAMEX\u{85}TYPE=\"MONEY\">", WhitespaceMode::Legacy), None);
    }

    #[test]
    fn tag_names() {
        let m = WhitespaceMode::Unicode;
        assert_eq!(tag_name("<s>", m), "s");
        assert_eq!(tag_name("</s>", m), "s");
        assert_eq!(tag_name("<ENAMEX TYPE=\"X\">", m), "ENAMEX");
        assert_eq!(tag_name("<>", m), "");
        assert_eq!(tag_name("</>", m), "");
        assert_eq!(tag_name("< s>", m), "");
        assert_eq!(tag_name("<a/>", m), "a/");
    }

    #[test]
    fn an_error_comes_before_the_samples_made_ahead_of_it() {
        assert_eq!(
            read("<s>a</s><s>b <ENAMEX TYPE=\"THING\">c</ENAMEX></s><s>d</s>"),
            [
                Err("Unknown timex, numex or namex type: THING, expected one of [PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT]".to_owned()),
                Ok("a ".to_owned()),
            ]
        );
        assert_eq!(read("<s>a</s> > <s>b</s>"), [Err("Did not expect > char!".to_owned()), Ok("a ".to_owned())]);
        assert_eq!(read("<s <s>"), [Err("Did not expect < char!".to_owned())]);
        assert_eq!(read("<s>a</s><s"), [Err("Did not find matching > char!".to_owned()), Ok("a ".to_owned())]);
        assert_eq!(read("<s>a</ENAMEX></s>"), [Err("".to_owned())]);
        assert_eq!(read("<s><ENAMEX>a</ENAMEX></s>"), [Err("".to_owned())]);
    }

    #[test]
    fn names_across_samples() {
        assert_eq!(
            read("<s>a b <ENAMEX TYPE=\"PERSON\">c</s><s>d</ENAMEX></s>"),
            [Err("start index must not be larger than end index: start=2, end=1".to_owned()), Ok("a b c".to_owned())]
        );
        assert_eq!(
            read("<s><ENAMEX TYPE=\"PERSON\">a</s><s>b c</ENAMEX></s>"),
            [Ok("a ".to_owned()), Ok("<START:person> b c <END>".to_owned())]
        );
        assert_eq!(
            read("<s>a</s><ENAMEX TYPE=\"PERSON\">x</ENAMEX><s>b</s>"),
            [Ok("a ".to_owned()), Ok("<START:person> <END> b ".to_owned())]
        );
    }

    #[test]
    fn nested_names_overlap() {
        let text = "<s><ENAMEX TYPE=\"ORGANIZATION\"><ENAMEX TYPE=\"LOCATION\">a</ENAMEX> b</ENAMEX></s>";
        assert_eq!(
            read(text),
            [Err("name spans [0..2) organization and [0..1) location are overlapped in file: null, sentence: [a, b]"
                .to_owned())]
        );
    }

    #[test]
    fn documents_end_at_one_without_samples() {
        let t = RuleTokenizer::default();
        let read = |docs: &[&'static str]| -> Vec<Result<String, String>> {
            MucSamples::new(docs.iter().copied(), &t, WhitespaceMode::Unicode)
                .map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string()))
                .collect()
        };
        assert_eq!(
            read(&["<s>a</s>", "<s", "<s>b</s>"]),
            [Ok("a ".to_owned()), Err("Did not find matching > char!".to_owned()), Ok("b ".to_owned())]
        );
        assert_eq!(read(&["<s>a</s>", "<DOC></DOC>", "<s>b</s>"]), [Ok("a ".to_owned())]);
    }

    #[test]
    fn files_come_in_reverse_name_order() {
        let dir = std::env::temp_dir().join(format!("turbo-muc-files-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("d.sgm")).unwrap();
        for name in ["a.sgm", "B.SGM", "c.Sgm", "e.txt", "sgm"] {
            std::fs::write(dir.join(name), "").unwrap();
        }
        let names: Vec<_> =
            muc_files(&dir).unwrap().iter().map(|p| p.file_name().unwrap().to_str().unwrap().to_owned()).collect();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(names, ["c.Sgm", "a.sgm", "B.SGM"]);
    }

    #[test]
    fn errors_name_their_lines() {
        let text = "<s>a</s>\r\n\n<s>\rb <ENAMEX TYPE=\"X\">";
        let e = MucReader::new(text, &RuleTokenizer::default(), WhitespaceMode::Unicode).unwrap().next().unwrap();
        assert_eq!(e.unwrap_err().line(), Some(4));
    }
}
