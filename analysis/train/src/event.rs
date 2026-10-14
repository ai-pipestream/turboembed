//! Training events and the two text formats OpenNLP reads them from.
//!
//! An [`Event`] is `opennlp.tools.ml.model.Event`: an outcome, the context
//! (the predicates that were true) and, for real valued features, one
//! `float` per predicate. Both are [`JavaString`]s: the tokenizer's
//! context generator makes predicates that hold half a surrogate pair,
//! and the lemmatizer's classes, its outcomes, edit the UTF-16 units of a
//! word and can hold one too. [`FileEventStream`] reads the format `FileEventStream`
//! reads (`outcome pred pred ...` per line), and
//! [`RealValueFileEventStream`] the one `RealValueFileEventStream` reads,
//! where a predicate may end in `=value`. [`EventChecksum`] is
//! `ChecksumEventStream`, the CRC-32C over every event's `toString` that a
//! trained model's manifest records as `Training-Eventhash`.

use crate::crc32c::Crc32c;
use crate::error::{Error, Result};
use crate::java_string::JavaString;
use turbo_jvm::text;
use turbo_unicode::props::white_space;

/// One training event.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    outcome: JavaString,
    context: Vec<JavaString>,
    /// One value per context, or none when every predicate counts 1.
    values: Option<Vec<f32>>,
}

impl Event {
    /// An event whose predicates all count 1.
    pub fn new(outcome: impl Into<JavaString>, context: Vec<JavaString>) -> Event {
        Event { outcome: outcome.into(), context, values: None }
    }

    /// An event with predicates given as text.
    pub fn from_strs(outcome: impl Into<JavaString>, context: &[&str]) -> Event {
        Event::new(outcome, context.iter().map(|c| JavaString::new(c)).collect())
    }

    /// An event with a value per predicate. `values` must be as long as
    /// `context`; OpenNLP does not check that and fails later, out of
    /// bounds. `None` values make a plain event.
    pub fn with_values(
        outcome: impl Into<JavaString>,
        context: Vec<JavaString>,
        values: Option<Vec<f32>>,
    ) -> Result<Event> {
        if let Some(v) = &values
            && v.len() != context.len()
        {
            return Err(Error::ValuesLength { contexts: context.len(), values: v.len() });
        }
        Ok(Event { outcome: outcome.into(), context, values })
    }

    pub fn outcome(&self) -> &JavaString {
        &self.outcome
    }

    pub fn context(&self) -> &[JavaString] {
        &self.context
    }

    pub fn values(&self) -> Option<&[f32]> {
        self.values.as_deref()
    }

    /// What `Event.toString` returns, encoded as `getBytes(UTF_8)` encodes
    /// it: `outcome [pred pred ...]`, each predicate followed by `=` and
    /// `Float.toString` of its value when the event has values, and a
    /// lone surrogate in the outcome or a predicate written as `?`. This
    /// is the text the event checksum hashes.
    pub fn java_string(&self) -> String {
        let mut s =
            String::with_capacity(self.outcome.len() + 3 + self.context.iter().map(|c| c.len() + 1).sum::<usize>());
        self.outcome.push_java_utf8(&mut s);
        s.push_str(" [");
        for (i, c) in self.context.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            c.push_java_utf8(&mut s);
            if let Some(values) = &self.values {
                s.push('=');
                s.push_str(&text::float_to_string(values[i]));
            }
        }
        s.push(']');
        s
    }

    /// The line `FileEventStream.toLine` writes, without its line
    /// separator and as a UTF-8 writer encodes it (a lone surrogate as
    /// `?`): the outcome and the predicates separated by single spaces.
    /// Values are not written; `RealValueFileEventStream` has no writer in
    /// OpenNLP. [`Event::real_value_line`] writes them.
    pub fn to_line(&self) -> String {
        let mut s = String::with_capacity(self.outcome.len() + self.context.iter().map(|c| c.len() + 1).sum::<usize>());
        self.outcome.push_java_utf8(&mut s);
        for c in &self.context {
            s.push(' ');
            c.push_java_utf8(&mut s);
        }
        s
    }

    /// A line [`RealValueFileEventStream`] reads back as this event: the
    /// outcome and the predicates, each followed by `=` and `Float.toString`
    /// of its value when the event has values.
    pub fn real_value_line(&self) -> String {
        let Some(values) = &self.values else {
            return self.to_line();
        };
        let mut s = String::with_capacity(self.outcome.len() + self.context.iter().map(|c| c.len() + 5).sum::<usize>());
        self.outcome.push_java_utf8(&mut s);
        for (c, v) in self.context.iter().zip(values) {
            s.push(' ');
            c.push_java_utf8(&mut s);
            s.push('=');
            s.push_str(&text::float_to_string(*v));
        }
        s
    }
}

/// The lines of `text` as `BufferedReader.readLine` returns them: ended by
/// `\n`, `\r` or `\r\n`, the end itself left out, and no empty last line
/// after a final line end.
pub struct Lines<'a> {
    rest: &'a str,
    /// The 1-based number of the line returned last.
    line: usize,
}

impl<'a> Lines<'a> {
    pub fn new(text: &'a str) -> Lines<'a> {
        Lines { rest: text, line: 0 }
    }

    /// The 1-based number of the line returned last, 0 before the first.
    pub fn line_number(&self) -> usize {
        self.line
    }
}

impl<'a> Iterator for Lines<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.rest.is_empty() {
            return None;
        }
        self.line += 1;
        let bytes = self.rest.as_bytes();
        let Some(end) = bytes.iter().position(|&b| b == b'\n' || b == b'\r') else {
            let line = self.rest;
            self.rest = "";
            return Some(line);
        };
        let line = &self.rest[..end];
        let skip = if bytes[end] == b'\r' && bytes.get(end + 1) == Some(&b'\n') { 2 } else { 1 };
        self.rest = &self.rest[end + skip..];
        Some(line)
    }
}

/// `StringTokenizer`'s default delimiters: space, tab, newline, carriage
/// return and form feed. Not Unicode whitespace.
fn is_tokenizer_delimiter(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{C}')
}

/// Parses one line as `FileEventStream.read` does: the tokens between
/// `StringTokenizer`'s default delimiters, the first the outcome, the rest
/// the context. A line with no token is an error, where OpenNLP throws
/// `NoSuchElementException`.
pub fn parse_event(line: &str, line_number: usize) -> Result<Event> {
    let mut tokens = line.split(is_tokenizer_delimiter).filter(|t| !t.is_empty());
    let Some(outcome) = tokens.next() else {
        return Err(Error::EmptyEventLine { line: line_number });
    };
    Ok(Event::new(outcome, tokens.map(JavaString::new).collect()))
}

/// `StringUtil.splitOnUnicodeWhitespace`: the runs of characters without
/// the Unicode `White_Space` property, in order.
pub fn split_on_unicode_whitespace(line: &str) -> impl Iterator<Item = &str> {
    line.split(white_space).filter(|t| !t.is_empty())
}

/// `RealValueFileEventStream.parseContexts`: for each context whose last
/// `=` is neither its first character nor its last, the text after it is
/// read with `Float.parseFloat`, which trims the characters up to U+0020
/// at both ends and reads a decimal or hexadecimal literal; when that
/// succeeds the context is cut before the `=` and the value kept (a
/// negative value is an error), and when it fails the context is left
/// whole and its value is 1 (OpenNLP logs an error and goes on). Returns
/// `None` when no context had a value, as OpenNLP returns a null array.
pub fn parse_contexts(contexts: &mut [JavaString], line_number: usize) -> Result<Option<Vec<f32>>> {
    let mut values = vec![1.0f32; contexts.len()];
    let mut has_real_value = false;
    for (context, value) in contexts.iter_mut().zip(values.iter_mut()) {
        // `=` is ASCII, and every byte of a non-ASCII unit is 0x80 or more,
        // so a byte search finds the last `=` character.
        let bytes = context.as_bytes();
        let Some(eq) = bytes.iter().rposition(|&b| b == b'=') else { continue };
        if eq == 0 || eq + 1 >= bytes.len() {
            continue;
        }
        // The trimmed text may hold a lone surrogate, which is not a
        // number (`String::from_utf16` refuses it).
        let units: Vec<u16> = context.utf16_units_from(eq + 1).collect();
        let Ok(v) =
            String::from_utf16(java_trim(&units)).map_err(|_| ()).and_then(|t| text::parse_float(&t).map_err(|_| ()))
        else {
            continue;
        };
        if v < 0.0 {
            return Err(Error::NegativeValue { line: line_number, context: std::mem::take(context) });
        }
        *value = v;
        context.truncate(eq);
        has_real_value = true;
    }
    Ok(has_real_value.then_some(values))
}

/// `String.trim`: the units up to U+0020 taken off both ends.
fn java_trim(units: &[u16]) -> &[u16] {
    let start = units.iter().position(|&u| u > 0x20).unwrap_or(units.len());
    let end = units.iter().rposition(|&u| u > 0x20).map_or(start, |i| i + 1);
    &units[start..end]
}

/// Parses one line as `RealValueFileEventStream.parseEvent` does: fields
/// split on Unicode `White_Space`, the first the outcome, the rest the
/// contexts read by [`parse_contexts`]. A line with no field is an error
/// (`InvalidFormatException`).
pub fn parse_real_value_event(line: &str, line_number: usize) -> Result<Event> {
    let mut fields = split_on_unicode_whitespace(line);
    let Some(outcome) = fields.next() else {
        return Err(Error::NoOutcome { line: line_number, text: line.to_owned() });
    };
    let mut contexts: Vec<JavaString> = fields.map(JavaString::new).collect();
    let values = parse_contexts(&mut contexts, line_number)?;
    Event::with_values(outcome, contexts, values)
}

/// `FileEventStream` over a text: one event per line.
pub struct FileEventStream<'a> {
    lines: Lines<'a>,
}

impl<'a> FileEventStream<'a> {
    pub fn new(text: &'a str) -> FileEventStream<'a> {
        FileEventStream { lines: Lines::new(text) }
    }
}

impl Iterator for FileEventStream<'_> {
    type Item = Result<Event>;

    fn next(&mut self) -> Option<Result<Event>> {
        let line = self.lines.next()?;
        Some(parse_event(line, self.lines.line_number()))
    }
}

/// `RealValueFileEventStream` over a text: one event per line.
pub struct RealValueFileEventStream<'a> {
    lines: Lines<'a>,
}

impl<'a> RealValueFileEventStream<'a> {
    pub fn new(text: &'a str) -> RealValueFileEventStream<'a> {
        RealValueFileEventStream { lines: Lines::new(text) }
    }
}

impl Iterator for RealValueFileEventStream<'_> {
    type Item = Result<Event>;

    fn next(&mut self) -> Option<Result<Event>> {
        let line = self.lines.next()?;
        Some(parse_real_value_event(line, self.lines.line_number()))
    }
}

/// `ChecksumEventStream`: the CRC-32C of the UTF-8 of every event's
/// `toString`, in event order. OpenNLP writes the value into the manifest
/// of a trained model as `Training-Eventhash`.
#[derive(Debug, Clone, Default)]
pub struct EventChecksum {
    crc: Crc32c,
}

impl EventChecksum {
    pub fn new() -> EventChecksum {
        EventChecksum::default()
    }

    pub fn update(&mut self, event: &Event) {
        self.crc.update(event.java_string().as_bytes());
    }

    /// The checksum so far; `Training-Eventhash` is its decimal form.
    pub fn value(&self) -> u32 {
        self.crc.value()
    }
}

/// The checksum of a sequence of events.
pub fn checksum<'e>(events: impl IntoIterator<Item = &'e Event>) -> u32 {
    let mut c = EventChecksum::new();
    for e in events {
        c.update(e);
    }
    c.value()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strs(items: &[&str]) -> Vec<JavaString> {
        items.iter().map(|s| JavaString::new(s)).collect()
    }

    fn valued(outcome: &str, context: &[&str], values: &[f32]) -> Event {
        Event::with_values(outcome, strs(context), Some(values.to_vec())).unwrap()
    }

    #[test]
    fn lines_split_as_read_line_splits_them() {
        let lines: Vec<&str> = Lines::new("a\nb\r\nc\rd\n\ne").collect();
        assert_eq!(lines, ["a", "b", "c", "d", "", "e"]);
        let lines: Vec<&str> = Lines::new("a\n").collect();
        assert_eq!(lines, ["a"]);
        assert_eq!(Lines::new("").count(), 0);
        let lines: Vec<&str> = Lines::new("\n").collect();
        assert_eq!(lines, [""]);
    }

    #[test]
    fn file_event_stream_splits_on_the_tokenizer_delimiters_only() {
        let events: Vec<Event> = FileEventStream::new(" s\tp=1  w=a\u{C}x\n n\u{A0}y\n").map(Result::unwrap).collect();
        assert_eq!(events[0], Event::from_strs("s", &["p=1", "w=a", "x"]));
        // A no-break space is not a StringTokenizer delimiter.
        assert_eq!(events[1], Event::from_strs("n\u{A0}y", &[]));
    }

    #[test]
    fn an_empty_event_line_is_an_error() {
        let mut stream = FileEventStream::new("a b\n\nc d");
        assert!(stream.next().unwrap().is_ok());
        assert_eq!(stream.next().unwrap(), Err(Error::EmptyEventLine { line: 2 }));
        assert!(stream.next().unwrap().is_ok());
        assert_eq!(parse_event(" \t ", 7), Err(Error::EmptyEventLine { line: 7 }));
    }

    #[test]
    fn to_line_round_trips() {
        let e = Event::from_strs("s", &["a", "b=1"]);
        assert_eq!(e.to_line(), "s a b=1");
        assert_eq!(parse_event(&e.to_line(), 1).unwrap(), e);
        assert_eq!(Event::from_strs("s", &[]).to_line(), "s");
    }

    #[test]
    fn java_string_is_event_to_string() {
        assert_eq!(Event::from_strs("s", &["a", "b"]).java_string(), "s [a b]");
        assert_eq!(Event::from_strs("s", &[]).java_string(), "s []");
        let e = valued("n", &["a", "b"], &[1.0, 0.25]);
        assert_eq!(e.java_string(), "n [a=1.0 b=0.25]");
        assert_eq!(e.real_value_line(), "n a=1.0 b=0.25");
        assert_eq!(parse_real_value_event(&e.real_value_line(), 1).unwrap(), e);
    }

    #[test]
    fn a_lone_surrogate_is_written_as_a_question_mark() {
        // The tokenizer's "p=" feature with the first half of an emoji.
        let e = Event::new("T", vec![JavaString::from_utf16(&[0x70, 0x3D, 0xD83D]), JavaString::new("😀")]);
        assert_eq!(e.java_string(), "T [p=? 😀]");
        assert_eq!(e.to_line(), "T p=? 😀");
        assert_eq!(checksum([&e]), crate::crc32c::crc32c("T [p=? 😀]".as_bytes()));
        // The lemmatizer's class of "😀" with the lemma "_": R1<low>_D0<high>.
        let class = JavaString::from_utf16(&[0x52, 0x31, 0xDE00, 0x5F, 0x44, 0x30, 0xD83D]);
        let e = Event::new(class.clone(), vec![JavaString::new("w0=😀")]);
        assert_eq!(e.outcome(), &class);
        assert_eq!(e.java_string(), "R1?_D0? [w0=😀]");
        assert_eq!(e.to_line(), "R1?_D0? w0=😀");
        assert_eq!(e.real_value_line(), "R1?_D0? w0=😀");
    }

    #[test]
    fn values_must_be_one_per_context() {
        assert_eq!(
            Event::with_values("s", strs(&["a"]), Some(vec![1.0, 2.0])),
            Err(Error::ValuesLength { contexts: 1, values: 2 })
        );
        assert_eq!(Event::with_values("s", strs(&["a"]), None).unwrap(), Event::from_strs("s", &["a"]));
    }

    #[test]
    fn real_value_lines_split_on_unicode_whitespace() {
        // The no-break space and the ideographic space have White_Space; the
        // zero width space and the word joiner do not.
        let e = parse_real_value_event("s\u{A0}a=2\u{3000}b\u{200B}c=0.5 d", 1).unwrap();
        assert_eq!(*e.outcome(), "s");
        assert_eq!(e.context(), strs(&["a", "b\u{200B}c", "d"]));
        assert_eq!(e.values(), Some(&[2.0f32, 0.5, 1.0][..]));
    }

    #[test]
    fn parse_contexts_follows_the_last_equals_sign() {
        let mut c = strs(&["a=b=2", "x=", "=3", "y", "z=1e2", "w=abc", "k==4", "é=\0"]);
        let values = parse_contexts(&mut c, 1).unwrap().unwrap();
        assert_eq!(c, strs(&["a=b", "x=", "=3", "y", "z", "w=abc", "k=", "é=\0"]));
        assert_eq!(values, [2.0, 1.0, 1.0, 1.0, 100.0, 1.0, 4.0, 1.0]);
    }

    #[test]
    fn a_value_is_trimmed_as_parse_float_trims_it() {
        // U+0000 is not Unicode whitespace, so a field may hold it; parseFloat
        // trims it with every other character up to U+0020. "x=1\0" is 1 and
        // "x=\0" is the empty string, which is not a number.
        let mut c = strs(&["x=1\0", "x=\0", "y=\t2 ", "z=1\0 2", "w=\u{1}3\u{1f}", "v=\u{A0}4", "u=0x1p3"]);
        let values = parse_contexts(&mut c, 1).unwrap().unwrap();
        assert_eq!(c, strs(&["x", "x=\0", "y", "z=1\0 2", "w", "v=\u{A0}4", "u"]));
        assert_eq!(values, [1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 8.0]);
        // A lone surrogate after the `=` is not a number, trimmed or not.
        let mut c = vec![
            JavaString::from_utf16(&[0x61, 0x3D, 0x31, 0xD83D]),
            JavaString::from_utf16(&[0x61, 0x3D, 0xD83D, 0x20]),
        ];
        assert_eq!(parse_contexts(&mut c, 1).unwrap(), None);
        assert_eq!(c[0].utf16_len(), 4);
        assert_eq!(java_trim(&[0x20, 0x61, 0x00]), [0x61]);
        assert_eq!(java_trim(&[0x20, 0x00]), [] as [u16; 0]);
        assert_eq!(java_trim(&[]), [] as [u16; 0]);
    }

    #[test]
    fn an_unreadable_value_keeps_the_context_and_counts_one() {
        let e = parse_real_value_event("s w=abc", 1).unwrap();
        assert_eq!(e.context(), strs(&["w=abc"]));
        // No context had a value, so the event has none, as OpenNLP returns null.
        assert_eq!(e.values(), None);
        // A value read with a Java suffix, and one that is zero.
        let e = parse_real_value_event("s w=2f v=0", 1).unwrap();
        assert_eq!(e.values(), Some(&[2.0f32, 0.0][..]));
    }

    #[test]
    fn negative_values_are_refused() {
        assert_eq!(
            parse_real_value_event("s a=1 b=-0.5", 3),
            Err(Error::NegativeValue { line: 3, context: JavaString::new("b=-0.5") })
        );
        // Negative zero and NaN are not below zero, as in Java.
        let e = parse_real_value_event("s a=-0 b=NaN", 1).unwrap();
        let v = e.values().unwrap();
        assert!(v[0] == 0.0 && v[0].is_sign_negative());
        assert!(v[1].is_nan());
    }

    #[test]
    fn a_line_without_fields_has_no_outcome() {
        assert_eq!(parse_real_value_event(" \u{3000}", 4), Err(Error::NoOutcome { line: 4, text: " \u{3000}".into() }));
        let mut stream = RealValueFileEventStream::new("a b\n\n");
        assert!(stream.next().unwrap().is_ok());
        assert_eq!(stream.next().unwrap(), Err(Error::NoOutcome { line: 2, text: String::new() }));
        assert!(stream.next().is_none());
    }

    #[test]
    fn checksum_hashes_the_java_strings() {
        let events = [Event::from_strs("s", &["a", "b"]), Event::from_strs("n", &["c"])];
        let expected = crate::crc32c::crc32c("s [a b]n [c]".as_bytes());
        assert_eq!(checksum(&events), expected);
        let mut c = EventChecksum::new();
        assert_eq!(c.value(), 0);
        c.update(&events[0]);
        c.update(&events[1]);
        assert_eq!(c.value(), expected);
    }
}
