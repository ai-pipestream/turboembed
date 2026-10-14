//! The brat annotation file (`.ann`) and annotation configuration
//! (`annotation.conf`) parsers.

use std::collections::BTreeMap;
use std::fmt;

use turbo_unicode::WhitespaceMode;

use crate::error::FormatError;
use crate::masc::java::parse_int;
use crate::ner::NerError;
use crate::source::{Lines, Piece, java_trim, split_fields, utf16_cut, whitespace_tokens};

/// The type of the annotator notes (`BratAnnotationParser.NOTES_TYPE`).
const NOTES_TYPE: &str = "AnnotatorNotes";

/// One fragment of a text-bound annotation: UTF-16 code units
/// `start..end` of the document's text, as brat's offsets are read.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fragment {
    pub start: i32,
    pub end: i32,
}

/// What an annotation of an `.ann` file holds besides its id and type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnnotationValue {
    /// A text-bound annotation (`T`, `SpanAnnotation`): its fragments,
    /// sorted as `Span.compareTo` sorts them, and the text the line gives
    /// after the offsets.
    Span { fragments: Vec<Fragment>, covered_text: String },
    /// A relation (`R`): the two arguments after their five-character
    /// prefix (`Arg1:`), trimmed.
    Relation { arg1: String, arg2: String },
    /// An attribute (`A`): the annotation it is attached to and its value,
    /// if it has one.
    Attribute { attached_to: String, value: Option<String> },
    /// An event (`E`): its trigger and its arguments, role to annotation;
    /// a role given twice keeps the last.
    Event { trigger: String, arguments: BTreeMap<String, String> },
}

/// An annotation of a brat document (`BratAnnotation`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BratAnnotation {
    pub id: String,
    /// The annotation's type: the name type of a text-bound annotation.
    pub kind: String,
    /// The text of the last annotator note attached to it, empty if none.
    pub note: String,
    pub value: AnnotationValue,
}

impl BratAnnotation {
    /// The fragments of a text-bound annotation, or none.
    pub fn fragments(&self) -> &[Fragment] {
        match &self.value {
            AnnotationValue::Span { fragments, .. } => fragments,
            _ => &[],
        }
    }
}

/// An annotator note (`#`, `AnnotatorNoteAnnotation`), which
/// `BratDocument` attaches to the annotation it names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Note {
    pub(crate) attached_to: String,
    pub(crate) note: String,
}

/// An annotation or a note, as the annotation stream reads them.
pub(crate) enum Parsed {
    Annotation(BratAnnotation),
    Note(Note),
}

/// Why a line of an annotation file is refused: the cause of the
/// `IOException` OpenNLP's annotation stream wraps it in
/// ([`NerError::BratAnnotation`]). `Display` is the cause's message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BratCause {
    /// A text-bound annotation line of fewer than five fields.
    SpanFields,
    /// An offset `Integer.parseInt` refuses.
    NotANumber(String),
    /// A relation argument of fewer than five UTF-16 code units.
    Argument(String),
    /// An event type without exactly one `:`; `line` is the line.
    EventType(String),
    /// An event argument without exactly one `:`; `line` is the line.
    EventArgument(String),
    /// An attribute line of other than three or four fields.
    AttributeFields,
}

impl fmt::Display for BratCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BratCause::SpanFields => f.write_str("Line must have at least 5 fields"),
            BratCause::NotANumber(s) => write!(f, "java.lang.NumberFormatException: For input string: \"{s}\""),
            BratCause::Argument(s) => write!(f, "Failed to parse argument: {s}"),
            BratCause::EventType(line) => {
                write!(f, "Failed to parse [{line}], type part must be in the format type:trigger")
            }
            BratCause::EventArgument(line) => {
                write!(f, "Failed to parse [{line}], argument parts must be in form argument:value")
            }
            BratCause::AttributeFields => f.write_str("Line must have 3 or 4 fields"),
        }
    }
}

/// The error of one line, before the document's id is known to it.
enum LineError {
    /// An `IOException` of the parsers, which the stream wraps.
    Wrapped(BratCause),
    /// An array index out of bounds.
    Index { index: i64, len: usize },
    /// `Span`'s constructor refusing the offsets.
    Span { start: i32, end: i32 },
}

impl LineError {
    fn error(self, file: &str, line: u32) -> NerError {
        let file = file.to_owned();
        match self {
            LineError::Wrapped(cause) => NerError::BratAnnotation { file, line, cause },
            LineError::Index { index, len } => NerError::BratIndex { file, line, index, len },
            LineError::Span { start, end } => NerError::BratSpan { file, line, start, end },
        }
    }
}

fn number(s: &str) -> Result<i32, LineError> {
    parse_int(s).ok_or_else(|| LineError::Wrapped(BratCause::NotANumber(s.to_owned())))
}

/// `values[i]`, or the `ArrayIndexOutOfBoundsException` Java throws.
fn at<'v, 'a>(values: &'v [Piece<'a>], i: i64) -> Result<&'v Piece<'a>, LineError> {
    usize::try_from(i).ok().and_then(|i| values.get(i)).ok_or(LineError::Index { index: i, len: values.len() })
}

/// A fragment, as `new Span(begin, end, type)` checks it.
fn fragment(start: i32, end: i32) -> Result<Fragment, LineError> {
    if start < 0 || end < 0 || start > end {
        return Err(LineError::Span { start, end });
    }
    Ok(Fragment { start, end })
}

/// `SpanAnnotationParser.parse`. `line` is the line, which starts at
/// offset `base` of the file the pieces are of.
fn span_annotation(values: &[Piece<'_>], line: &str, base: u32) -> Result<BratAnnotation, LineError> {
    if values.len() <= 4 {
        return Err(LineError::Wrapped(BratCause::SpanFields));
    }
    let mut begin = number(values[2].text)?;
    let mut fragments = Vec::new();
    let mut first_text: i64 = -1;
    for (i, v) in values.iter().enumerate().skip(3) {
        if v.text.contains(';') {
            let parts = split_fields(*v, ';');
            let end = number(at(&parts, 0)?.text)?;
            fragments.push(fragment(begin, end)?);
            begin = number(at(&parts, 1)?.text)?;
        } else {
            let end = number(v.text)?;
            first_text = i as i64 + 1;
            fragments.push(fragment(begin, end)?);
            break;
        }
    }
    let from = at(values, first_text)?.span.start;
    let to = values[values.len() - 1].span.end;
    let covered_text = line[(from - base) as usize..(to - base) as usize].to_owned();
    // `Span.compareTo`: by start, the longer first (the type is the same)
    fragments.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    Ok(BratAnnotation {
        id: values[0].text.to_owned(),
        kind: values[1].text.to_owned(),
        note: String::new(),
        value: AnnotationValue::Span { fragments, covered_text },
    })
}

/// `RelationAnnotationParser.parseArg`: the argument after its first five
/// UTF-16 code units, trimmed (`String.trim`).
fn relation_argument(arg: &str) -> Result<String, LineError> {
    let start = match utf16_cut(arg, 5) {
        Ok(Some(i)) => i,
        // Java cuts the surrogate pair and keeps its second half, which a
        // Rust string cannot hold: the argument starts after the pair
        Ok(None) => {
            let mut units = 0;
            arg.char_indices()
                .find_map(|(i, c)| {
                    units += c.len_utf16();
                    (units > 5).then_some(i + c.len_utf8())
                })
                .unwrap_or(arg.len())
        }
        Err(len) if len > 4 => unreachable!("an argument of five units or more can be cut after five"),
        Err(_) => return Err(LineError::Wrapped(BratCause::Argument(arg.to_owned()))),
    };
    Ok(arg[start..].trim_matches(|c| c <= ' ').to_owned())
}

/// `EventAnnotationParser.parse`.
fn event_annotation(values: &[Piece<'_>], line: &str) -> Result<BratAnnotation, LineError> {
    let type_parts = split_fields(values[1], ':');
    if type_parts.len() != 2 {
        return Err(LineError::Wrapped(BratCause::EventType(line.to_owned())));
    }
    let mut arguments = BTreeMap::new();
    for v in &values[2..] {
        let parts = split_fields(*v, ':');
        if parts.len() != 2 {
            return Err(LineError::Wrapped(BratCause::EventArgument(line.to_owned())));
        }
        arguments.insert(parts[0].text.to_owned(), parts[1].text.to_owned());
    }
    Ok(BratAnnotation {
        id: values[0].text.to_owned(),
        kind: type_parts[0].text.to_owned(),
        note: String::new(),
        value: AnnotationValue::Event { trigger: type_parts[1].text.to_owned(), arguments },
    })
}

/// One line of an annotation file, as `BratAnnotationStream.read` reads
/// it: `Ok(None)` for a line the stream skips. `line` starts at offset
/// `base` of the file.
fn parse_line(values: &[Piece<'_>], line: &str, base: u32) -> Result<Option<Parsed>, LineError> {
    let id = values[0].text.to_owned();
    let kind = values[1].text.to_owned();
    let annotation = |value| Parsed::Annotation(BratAnnotation { id, kind, note: String::new(), value });
    Ok(Some(match values[0].text.as_bytes()[0] {
        b'T' => Parsed::Annotation(span_annotation(values, line, base)?),
        b'R' => {
            let arg1 = relation_argument(values[2].text)?;
            let arg2 = relation_argument(at(values, 3)?.text)?;
            annotation(AnnotationValue::Relation { arg1, arg2 })
        }
        b'A' => {
            if values.len() != 3 && values.len() != 4 {
                return Err(LineError::Wrapped(BratCause::AttributeFields));
            }
            let attached_to = values[2].text.to_owned();
            let value = values.get(3).map(|v| v.text.to_owned());
            annotation(AnnotationValue::Attribute { attached_to, value })
        }
        b'E' => Parsed::Annotation(event_annotation(values, line)?),
        b'#' if values[1].text == NOTES_TYPE => {
            let from = at(values, 3)?;
            let to = values[values.len() - 1].span.end;
            let note = line[(from.span.start - base) as usize..(to - base) as usize].to_owned();
            Parsed::Note(Note { attached_to: values[2].text.to_owned(), note })
        }
        _ => return Ok(None),
    }))
}

/// The annotations and notes of the annotation file `text` of the
/// document `file`, in the order of the file, as `BratAnnotationStream`
/// reads them: each line split on whitespace (`mode`), the first field
/// the id, whose first character says what the line is. A line of two
/// fields or fewer ends the reading, and the lines after it are not read.
pub(crate) fn parse_annotations(text: &str, file: &str, mode: WhitespaceMode) -> Result<Vec<Parsed>, FormatError> {
    let mut out = Vec::new();
    let lines = Lines::new(text)?;
    for line in lines {
        let values: Vec<Piece<'_>> = whitespace_tokens(line.piece, mode).collect();
        if values.len() <= 2 {
            break;
        }
        match parse_line(&values, line.text(), line.span().start) {
            Ok(Some(p)) => out.push(p),
            Ok(None) => {}
            Err(e) => return Err(e.error(file, line.number).into()),
        }
    }
    Ok(out)
}

/// The class of an annotation type in an annotation configuration.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AnnotationClass {
    Entity,
    Relation,
    Attribute,
    Event,
}

impl AnnotationClass {
    /// The name OpenNLP gives the class (`AnnotationConfiguration.ENTITY_TYPE` and so on).
    pub fn java_name(self) -> &'static str {
        match self {
            AnnotationClass::Entity => "Entity",
            AnnotationClass::Relation => "Relation",
            AnnotationClass::Attribute => "Attribute",
            AnnotationClass::Event => "Event",
        }
    }
}

/// A brat annotation configuration (`AnnotationConfiguration`): the
/// class of each type it lists. The brat factory requires one, though
/// nothing it reads depends on it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AnnotationConfiguration {
    types: BTreeMap<String, AnnotationClass>,
}

impl AnnotationConfiguration {
    /// Parses `annotation.conf` as `AnnotationConfiguration.parse` does:
    /// each line trimmed (`String.trim`), empty lines and lines starting
    /// with `#` skipped, a line `[name]` starting a section, and the first
    /// whitespace-separated field of any other line a type of the
    /// section's class: `entities`, `relations`, `attributes` or `events`
    /// (the types of other sections are not kept). A type listed twice
    /// keeps its last class.
    ///
    /// # Errors
    ///
    /// [`NerError::BratConfigNoSection`] for a type before any section,
    /// [`NerError::BratConfigNoType`] for a line with no field (only
    /// whitespace that `String.trim` keeps), and [`FormatError::TooLong`]
    /// for a text longer than spans can index.
    pub fn parse(text: &str, mode: WhitespaceMode) -> Result<AnnotationConfiguration, FormatError> {
        let mut types = BTreeMap::new();
        let mut section: Option<&str> = None;
        let lines = Lines::new(text)?;
        for line in lines {
            let number = line.number;
            let trimmed = java_trim(line.piece);
            let t = trimmed.text;
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            if t.starts_with('[') && t.ends_with(']') {
                let open = t.find('[').unwrap_or(0);
                let close = t.find(']').unwrap_or(t.len());
                section = Some(&t[open + 1..close]);
                continue;
            }
            let name = whitespace_tokens(trimmed, mode).next().ok_or(NerError::BratConfigNoType { line: number })?;
            let class = match section.ok_or(NerError::BratConfigNoSection { line: number })? {
                "entities" => AnnotationClass::Entity,
                "relations" => AnnotationClass::Relation,
                "attributes" => AnnotationClass::Attribute,
                "events" => AnnotationClass::Event,
                _ => continue,
            };
            types.insert(name.text.to_owned(), class);
        }
        Ok(AnnotationConfiguration { types })
    }

    /// The class of `kind` (`getTypeClass`).
    pub fn class(&self, kind: &str) -> Option<AnnotationClass> {
        self.types.get(kind).copied()
    }

    /// Every type and its class, by type.
    pub fn types(&self) -> impl Iterator<Item = (&str, AnnotationClass)> {
        self.types.iter().map(|(k, &c)| (k.as_str(), c))
    }
}
