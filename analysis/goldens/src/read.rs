//! The parser: a golden file's text into a typed `Golden`. It works line
//! by line, splitting each on tabs, and refuses anything a conforming
//! writer would not write. Every error names its line (counted from 1)
//! and what was expected there.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use turbo_analysis_layers::Strs;

use crate::compare::{Shape, known_shapes};
use crate::escape::{EscapeError, unescape_into};
use crate::format::{
    AnnotatorRef, Column, ColumnData, ColumnType, ConfigEntry, Generator, Golden, GoldenAnchor, GoldenLayer, Header,
    Input, Jdk, JvmError, Model, OpenNlp, UnicodeVersion, VERSION,
};
use crate::hex::{digit, parse_hex, to_hex};
use crate::sha::sha256;

/// Why a golden file, or the input text it names, was refused.
#[derive(Debug)]
pub enum GoldenError {
    /// The file could not be read.
    Io { path: PathBuf, error: io::Error },
    /// A parse error in the named file.
    File { path: PathBuf, error: Box<GoldenError> },
    /// The bytes of this line are not UTF-8.
    NotUtf8 { line: usize },
    /// A carriage return: lines end with LF alone.
    CarriageReturn { line: usize },
    /// The last line has no line end.
    NoFinalLineEnd { line: usize },
    /// The first line is not `golden 1`.
    Version { line: usize, found: String },
    /// The file ends where a line was expected.
    Eof { line: usize, expected: String },
    /// A line that is not the record expected there.
    Record { line: usize, expected: String, found: String },
    /// A record or row with the wrong number of fields.
    Fields { line: usize, expected: String, found: usize },
    /// A field that does not have the form its record or column needs.
    Field { line: usize, field: String, expected: &'static str, found: String },
    /// A `str` field with a bad escape.
    Escape { line: usize, field: String, error: EscapeError },
    /// A config record whose key is not after the one before it: keys are
    /// sorted, each once.
    ConfigOrder { line: usize, key: String, previous: String },
    /// A layer id this crate has no layer type for.
    UnknownLayer { line: usize, layer: String, known: Vec<String> },
    /// A layer whose anchor is not its type's.
    Anchor { line: usize, layer: String, expected: String, found: String },
    /// A layer whose columns are not its type's, by name and type.
    Columns { line: usize, layer: String, expected: Vec<String>, found: Vec<String> },
    /// A second block for one layer id.
    Duplicate { line: usize, layer: String },
    /// An aligned layer whose anchor is not an earlier layer of the file.
    AnchorLayer { line: usize, layer: String, anchor: String },
    /// An aligned layer whose row count is not its anchor's.
    AlignedRows { line: usize, layer: String, anchor: String, expected: usize, found: usize },
    /// The `end` line came before the declared number of rows.
    Rows { line: usize, layer: String, declared: usize, found: usize },
    /// A span that ends before it starts, ends past the input, or starts
    /// before the row before it; or a UTF-16 span that ends before it
    /// starts or past the input.
    Span { line: usize, layer: String, detail: String },
    /// A file with a header and no layer, and no error record.
    NoLayers { line: usize },
    /// A line after the error record: a run that threw has no layers.
    AfterError { line: usize, found: String },
    /// The input text does not match the header.
    Input { path: PathBuf, what: &'static str, expected: String, found: String },
}

impl fmt::Display for GoldenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use GoldenError::*;
        match self {
            Io { path, error } => write!(f, "{}: {error}", path.display()),
            File { path, error } => write!(f, "{}: {error}", path.display()),
            NotUtf8 { line } => write!(f, "line {line}: expected UTF-8 text"),
            CarriageReturn { line } => {
                write!(f, "line {line}: a carriage return, expected lines that end with LF alone")
            }
            NoFinalLineEnd { line } => write!(f, "line {line}: expected a line end after the last line"),
            Version { line, found } => write!(f, "line {line}: expected \"golden {VERSION}\", found {found:?}"),
            Eof { line, expected } => write!(f, "line {line}: the file ends, expected {expected}"),
            Record { line, expected, found } => write!(f, "line {line}: expected {expected}, found {found:?}"),
            Fields { line, expected, found } => {
                write!(f, "line {line}: expected the fields {expected}, found {found} fields")
            }
            Field { line, field, expected, found } => {
                write!(f, "line {line}: {field}: expected {expected}, found {found:?}")
            }
            Escape { line, field, error } => write!(f, "line {line}: {field}: {error}"),
            ConfigOrder { line, key, previous } => write!(
                f,
                "line {line}: the config key {key:?} comes after {previous:?}, expected keys sorted and each once"
            ),
            UnknownLayer { line, layer, known } => {
                write!(f, "line {line}: no layer type has the id {layer}, expected one of {}", known.join(", "))
            }
            Anchor { line, layer, expected, found } => {
                write!(f, "line {line}: {layer} is anchored to {found}, expected {expected}")
            }
            Columns { line, layer, expected, found } => write!(
                f,
                "line {line}: {layer} has the columns [{}], expected [{}]",
                found.join(", "),
                expected.join(", ")
            ),
            Duplicate { line, layer } => write!(f, "line {line}: a second block for {layer}, expected one"),
            AnchorLayer { line, layer, anchor } => {
                write!(f, "line {line}: {layer} is aligned to {anchor}, expected {anchor} earlier in the file")
            }
            AlignedRows { line, layer, anchor, expected, found } => {
                write!(f, "line {line}: {layer} declares {found} rows, expected {expected}, as many as {anchor} has")
            }
            Rows { line, layer, declared, found } => {
                write!(f, "line {line}: {layer} ends after {found} rows, expected the {declared} it declares")
            }
            Span { line, layer, detail } => write!(f, "line {line}: {layer}: {detail}"),
            NoLayers { line } => write!(f, "line {line}: the file ends, expected at least one layer block"),
            AfterError { line, found } => write!(
                f,
                "line {line}: expected the end of the file after the error record (a run that threw has no layers), \
                 found {found:?}"
            ),
            Input { path, what, expected, found } => {
                write!(f, "{}: the input's {what} is {found}, expected {expected} as the header says", path.display())
            }
        }
    }
}

impl std::error::Error for GoldenError {}

impl Golden {
    /// Reads and parses a golden file, checking its layer blocks against
    /// `known_shapes()`, the layer types of turbo-analysis and
    /// turbo-analysis-layers.
    pub fn read(path: impl AsRef<Path>) -> Result<Golden, GoldenError> {
        Golden::read_with(path, &known_shapes())
    }

    /// Reads and parses a golden file, checking its layer blocks against
    /// `shapes`. A crate that owns a layer type with a golden shape passes
    /// `known_shapes()` with `shape_of::<ItsLayer>()` added.
    pub fn read_with(path: impl AsRef<Path>, shapes: &[Shape]) -> Result<Golden, GoldenError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| GoldenError::Io { path: path.into(), error })?;
        Golden::parse_bytes_with(&bytes, shapes)
            .map_err(|e| GoldenError::File { path: path.into(), error: Box::new(e) })
    }

    /// Parses a golden file's bytes, against `known_shapes()`.
    pub fn parse_bytes(bytes: &[u8]) -> Result<Golden, GoldenError> {
        Golden::parse_bytes_with(bytes, &known_shapes())
    }

    /// Parses a golden file's bytes, against `shapes`.
    pub fn parse_bytes_with(bytes: &[u8], shapes: &[Shape]) -> Result<Golden, GoldenError> {
        let text = std::str::from_utf8(bytes).map_err(|e| GoldenError::NotUtf8 {
            line: 1 + bytes[..e.valid_up_to()].iter().filter(|&&b| b == b'\n').count(),
        })?;
        Golden::parse_with(text, shapes)
    }

    /// Parses a golden file's text, against `known_shapes()`.
    pub fn parse(text: &str) -> Result<Golden, GoldenError> {
        Golden::parse_with(text, &known_shapes())
    }

    /// Parses a golden file's text, checking each layer block's id, anchor
    /// and columns against `shapes`: a block whose id none of them has is
    /// refused.
    pub fn parse_with(text: &str, shapes: &[Shape]) -> Result<Golden, GoldenError> {
        if let Some(at) = text.find('\r') {
            return Err(GoldenError::CarriageReturn { line: 1 + text[..at].matches('\n').count() });
        }
        let Some(body) = text.strip_suffix('\n') else {
            return Err(GoldenError::NoFinalLineEnd { line: 1 + text.matches('\n').count() });
        };
        let mut lines = Lines { iter: body.split('\n').peekable(), line: 0 };
        let header = read_header(&mut lines)?;
        if header.error.is_some() {
            if let Some(found) = lines.next() {
                return Err(GoldenError::AfterError { line: lines.line, found: found.into() });
            }
            return Ok(Golden { header, layers: Vec::new() });
        }
        let mut layers: Vec<GoldenLayer> = Vec::new();
        while let Some(line) = lines.next() {
            let layer = read_layer(&mut lines, line, &header, shapes, &layers)?;
            layers.push(layer);
        }
        if layers.is_empty() {
            return Err(GoldenError::NoLayers { line: lines.line + 1 });
        }
        Ok(Golden { header, layers })
    }

    /// Reads the input text from `inputs/<set>/<id>.txt` under `data_dir`
    /// and checks its length in UTF-8 bytes and in UTF-16 units and its
    /// SHA-256 against the header, refusing a file that does not match.
    pub fn input_text(&self, data_dir: impl AsRef<Path>) -> Result<String, GoldenError> {
        let i = &self.header.input;
        let path = data_dir.as_ref().join("inputs").join(&i.set).join(format!("{}.txt", i.id));
        let bytes = std::fs::read(&path).map_err(|error| GoldenError::Io { path: path.clone(), error })?;
        let mismatch =
            |what, expected: String, found: String| GoldenError::Input { path: path.clone(), what, expected, found };
        if bytes.len() != i.utf8_len as usize {
            return Err(mismatch("length in UTF-8 bytes", i.utf8_len.to_string(), bytes.len().to_string()));
        }
        let digest = sha256(&bytes);
        if digest != i.sha256 {
            return Err(mismatch("SHA-256", to_hex(&i.sha256), to_hex(&digest)));
        }
        let text = String::from_utf8(bytes).map_err(|e| {
            mismatch("encoding", "UTF-8".into(), format!("invalid UTF-8 at byte {}", e.utf8_error().valid_up_to()))
        })?;
        let units = text.encode_utf16().count();
        if units != i.utf16_len as usize {
            return Err(mismatch("length in UTF-16 units", i.utf16_len.to_string(), units.to_string()));
        }
        Ok(text)
    }
}

/// The lines of the file, counting as it goes.
struct Lines<'a> {
    iter: std::iter::Peekable<std::str::Split<'a, char>>,
    /// The number of the line last returned.
    line: usize,
}

impl<'a> Lines<'a> {
    fn next(&mut self) -> Option<&'a str> {
        let l = self.iter.next()?;
        self.line += 1;
        Some(l)
    }

    /// The next line, without taking it.
    fn peek(&mut self) -> Option<&'a str> {
        self.iter.peek().copied()
    }

    fn expect(&mut self, expected: &str) -> Result<&'a str, GoldenError> {
        self.next().ok_or_else(|| GoldenError::Eof { line: self.line + 1, expected: expected.into() })
    }
}

/// A header record: the keyword, then exactly the named fields.
fn record<'a, const N: usize>(
    line: usize,
    text: &'a str,
    keyword: &str,
    names: [&str; N],
) -> Result<[&'a str; N], GoldenError> {
    let mut fields = text.split('\t');
    let first = fields.next().unwrap_or("");
    if first != keyword {
        return Err(GoldenError::Record { line, expected: format!("the {keyword} record"), found: text.into() });
    }
    let mut out = [""; N];
    let mut n = 0;
    for f in fields {
        if n < N {
            out[n] = f;
        }
        n += 1;
    }
    if n != N {
        return Err(GoldenError::Fields { line, expected: format!("{keyword} <{}>", names.join("> <")), found: n + 1 });
    }
    Ok(out)
}

fn read_header(lines: &mut Lines<'_>) -> Result<Header, GoldenError> {
    let first = lines.expect("\"golden 1\"")?;
    if first != format!("golden {VERSION}") {
        return Err(GoldenError::Version { line: lines.line, found: first.into() });
    }

    let l = lines.expect("the generator record")?;
    let [name, version] = record(lines.line, l, "generator", ["name", "version"])?;
    let generator = Generator {
        name: text_field(lines.line, "generator name", name)?,
        version: u32_field(lines.line, "generator version", version)?,
    };

    let l = lines.expect("the opennlp record")?;
    let [commit, version] = record(lines.line, l, "opennlp", ["commit", "maven version"])?;
    let opennlp = OpenNlp {
        commit: hex_field(lines.line, "opennlp commit", "40 lowercase hex digits", commit)?,
        version: text_field(lines.line, "opennlp version", version)?,
    };

    let l = lines.expect("the jdk record")?;
    let [version, vm, vendor] = record(lines.line, l, "jdk", ["java.version", "java.vm.version", "java.vendor"])?;
    let jdk = Jdk {
        version: text_field(lines.line, "java.version", version)?,
        vm_version: text_field(lines.line, "java.vm.version", vm)?,
        vendor: text_field(lines.line, "java.vendor", vendor)?,
    };

    let l = lines.expect("the unicode record")?;
    let [u] = record(lines.line, l, "unicode", ["version"])?;
    let unicode = u
        .split_once('.')
        .and_then(|(a, b)| Some(UnicodeVersion { major: parse_u32(a)?, minor: parse_u32(b)? }))
        .ok_or_else(|| field_error(lines.line, "unicode version", "<major>.<minor>, such as 15.0", u))?;

    let l = lines.expect("the input record")?;
    let [name, utf8, utf16, sha] = record(lines.line, l, "input", ["set/id", "utf8 length", "utf16 length", "sha256"])?;
    let (set, id) = name
        .split_once('/')
        .filter(|(s, i)| !s.is_empty() && !i.is_empty())
        .ok_or_else(|| field_error(lines.line, "input name", "<set>/<id>", name))?;
    let input = Input {
        set: set.into(),
        id: id.into(),
        utf8_len: u32_field(lines.line, "input length in UTF-8 bytes", utf8)?,
        utf16_len: u32_field(lines.line, "input length in UTF-16 units", utf16)?,
        sha256: hex_field(lines.line, "input sha256", "64 lowercase hex digits", sha)?,
    };

    let mut models = Vec::new();
    let annotator = loop {
        let l = lines.expect("a model record or the annotator record")?;
        if l.split('\t').next() == Some("model") {
            let [file, sha] = record(lines.line, l, "model", ["file name", "sha512"])?;
            models.push(Model {
                file: text_field(lines.line, "model file name", file)?,
                sha512: hex_field(lines.line, "model sha512", "128 lowercase hex digits", sha)?,
            });
            continue;
        }
        if l.split('\t').next() != Some("annotator") {
            return Err(GoldenError::Record {
                line: lines.line,
                expected: "a model record or the annotator record".into(),
                found: l.into(),
            });
        }
        let [class, method] = record(lines.line, l, "annotator", ["class", "method"])?;
        break AnnotatorRef {
            class: text_field(lines.line, "annotator class", class)?,
            method: text_field(lines.line, "annotator method", method)?,
        };
    };
    // Zero or more options, sorted by key, each key once.
    let mut config: Vec<ConfigEntry> = Vec::new();
    while let Some(l) = lines.peek().filter(|l| l.split('\t').next() == Some("config")) {
        lines.next();
        let line = lines.line;
        let [key, value] = record(line, l, "config", ["key", "value"])?;
        let mut buf = String::new();
        unescape_into(key, &mut buf).map_err(|error| GoldenError::Escape {
            line,
            field: "config key".into(),
            error,
        })?;
        let key = std::mem::take(&mut buf);
        unescape_into(value, &mut buf).map_err(|error| GoldenError::Escape {
            line,
            field: "config value".into(),
            error,
        })?;
        if let Some(previous) = config.last()
            && previous.key >= key
        {
            return Err(GoldenError::ConfigOrder { line, key, previous: previous.key.clone() });
        }
        config.push(ConfigEntry { key, value: buf });
    }
    // The exception the annotator threw, if it threw: after the options,
    // and the last line of the file.
    let mut error = None;
    if let Some(l) = lines.peek().filter(|l| l.split('\t').next() == Some("error")) {
        lines.next();
        let line = lines.line;
        let [class, message] = record(line, l, "error", ["class", "message"])?;
        let class = text_field(line, "error class", class)?;
        let mut buf = String::new();
        unescape_into(message, &mut buf).map_err(|error| GoldenError::Escape {
            line,
            field: "error message".into(),
            error,
        })?;
        error = Some(JvmError { class, message: buf });
    }
    Ok(Header { generator, opennlp, jdk, unicode, input, models, annotator, config, error })
}

/// The most rows reserved for ahead of reading them. The declared count
/// is only a claim until the rows are there, so a file declaring u32::MAX
/// rows is refused when it runs out of lines, not by allocating for them.
const RESERVE_CAP: usize = 4096;

fn read_layer(
    lines: &mut Lines<'_>,
    head: &str,
    header: &Header,
    shapes: &[Shape],
    earlier: &[GoldenLayer],
) -> Result<GoldenLayer, GoldenError> {
    let line = lines.line;
    let fields: Vec<&str> = head.split('\t').collect();
    if fields[0] != "layer" {
        return Err(GoldenError::Record { line, expected: "a layer record".into(), found: head.into() });
    }
    if fields.len() < 5 {
        return Err(GoldenError::Fields {
            line,
            expected: "layer <id> <anchor> <row count> <name:type>...".into(),
            found: fields.len(),
        });
    }
    let id = fields[1];
    let anchor = match fields[2] {
        "text" => GoldenAnchor::Text,
        "document" => GoldenAnchor::Document,
        a => match a.strip_prefix("aligned:") {
            Some(of) if !of.is_empty() => GoldenAnchor::Aligned(of.into()),
            _ => return Err(field_error(line, "layer anchor", "text, document or aligned:<layer id>", a)),
        },
    };
    let rows = u32_field(line, "layer row count", fields[3])? as usize;
    let mut specs = Vec::with_capacity(fields.len() - 4);
    for f in &fields[4..] {
        let spec = f.split_once(':').and_then(|(n, t)| Some((n, ColumnType::parse(t)?))).filter(|(n, _)| !n.is_empty());
        specs.push(spec.ok_or_else(|| field_error(line, "column", "<name>:<u32|f64|str>", f))?);
    }

    // The layer's type decides its anchor and columns.
    let Some(shape) = shapes.iter().find(|s| s.id == id) else {
        return Err(GoldenError::UnknownLayer {
            line,
            layer: id.into(),
            known: shapes.iter().map(|s| s.id.clone()).collect(),
        });
    };
    if earlier.iter().any(|l| l.id == id) {
        return Err(GoldenError::Duplicate { line, layer: id.into() });
    }
    if shape.anchor != anchor {
        return Err(GoldenError::Anchor {
            line,
            layer: id.into(),
            expected: shape.anchor.to_string(),
            found: anchor.to_string(),
        });
    }
    let found: Vec<String> = specs.iter().map(|(n, t)| format!("{n}:{t}")).collect();
    let expected: Vec<String> = shape.columns.iter().map(|c| c.to_string()).collect();
    if found != expected {
        return Err(GoldenError::Columns { line, layer: id.into(), expected, found });
    }
    if let GoldenAnchor::Aligned(of) = &anchor {
        let Some(target) = earlier.iter().find(|l| &l.id == of) else {
            return Err(GoldenError::AnchorLayer { line, layer: id.into(), anchor: of.clone() });
        };
        if target.len() != rows {
            return Err(GoldenError::AlignedRows {
                line,
                layer: id.into(),
                anchor: of.clone(),
                expected: target.len(),
                found: rows,
            });
        }
    }

    let mut columns: Vec<Column> = specs
        .iter()
        .map(|&(name, ty)| {
            let mut data = ColumnData::empty(ty);
            match &mut data {
                ColumnData::U32(v) => v.reserve(rows.min(RESERVE_CAP)),
                ColumnData::F64(v) => v.reserve(rows.min(RESERVE_CAP)),
                ColumnData::Str(_) => {}
            }
            Column { name: name.into(), data }
        })
        .collect();
    // Spans are checked as they are read, for the layers that have them.
    let span_cols = match anchor {
        GoldenAnchor::Text => {
            let start = specs.iter().position(|s| s.0 == "start");
            let end = specs.iter().position(|s| s.0 == "end");
            start.zip(end)
        }
        _ => None,
    };
    // A layer of UTF-16 spans (`turbo:utf16<L>`) has its spans in UTF-16
    // units of the input, in the order of the layer it is aligned to.
    let unit_cols = {
        let start = specs.iter().position(|s| s == &("start16", ColumnType::U32));
        let end = specs.iter().position(|s| s == &("end16", ColumnType::U32));
        start.zip(end)
    };
    let end_line = format!("end\t{id}");
    let mut buf = String::new();
    let mut last_start = 0;
    for r in 0..rows {
        let Some(row) = lines.next() else {
            return Err(GoldenError::Eof { line: lines.line + 1, expected: format!("row {r} of {id}") });
        };
        let line = lines.line;
        if let Err(e) = read_row(line, row, &mut columns, &mut buf) {
            // An early end line is the clearer error.
            return Err(if row == end_line {
                GoldenError::Rows { line, layer: id.into(), declared: rows, found: r }
            } else {
                e
            });
        }
        if let Some((s, e)) = span_cols {
            let (ColumnData::U32(starts), ColumnData::U32(ends)) = (&columns[s].data, &columns[e].data) else {
                unreachable!("start and end are u32 columns, as the shape says");
            };
            let (start, end) = (starts[r], ends[r]);
            let detail = if start > end {
                Some(format!("the span {start}..{end} ends before it starts"))
            } else if end > header.input.utf8_len {
                Some(format!("the span {start}..{end} ends past the input's {} bytes", header.input.utf8_len))
            } else if start < last_start {
                Some(format!("the span {start}..{end} starts before the row before it, expected text order"))
            } else {
                None
            };
            if let Some(detail) = detail {
                return Err(GoldenError::Span { line, layer: id.into(), detail });
            }
            last_start = start;
        }
        if let Some((s, e)) = unit_cols {
            let (ColumnData::U32(starts), ColumnData::U32(ends)) = (&columns[s].data, &columns[e].data) else {
                unreachable!("start16 and end16 are u32 columns, as found");
            };
            let (start, end) = (starts[r], ends[r]);
            let detail = if start > end {
                Some(format!("the UTF-16 span {start}..{end} ends before it starts"))
            } else if end > header.input.utf16_len {
                Some(format!(
                    "the UTF-16 span {start}..{end} ends past the input's {} UTF-16 units",
                    header.input.utf16_len
                ))
            } else {
                None
            };
            if let Some(detail) = detail {
                return Err(GoldenError::Span { line, layer: id.into(), detail });
            }
        }
    }
    let l = lines.expect(&format!("\"end\t{id}\""))?;
    if l != end_line {
        return Err(GoldenError::Record {
            line: lines.line,
            expected: format!("\"end\\t{id}\" after {rows} rows"),
            found: l.into(),
        });
    }
    Ok(GoldenLayer { id: id.into(), anchor, columns })
}

/// One row, a field per column, each appended to its column.
fn read_row(line: usize, row: &str, columns: &mut [Column], buf: &mut String) -> Result<(), GoldenError> {
    let n = row.split('\t').count();
    if n != columns.len() {
        let names: Vec<String> = columns.iter().map(|c| format!("<{}:{}>", c.name, c.data.ty())).collect();
        return Err(GoldenError::Fields { line, expected: names.join(" "), found: n });
    }
    for (field, col) in row.split('\t').zip(columns.iter_mut()) {
        match &mut col.data {
            ColumnData::U32(v) => v.push(
                parse_u32(field)
                    .ok_or_else(|| field_error(line, &col.name, "a u32 in decimal, no sign or leading zero", field))?,
            ),
            ColumnData::F64(v) => v.push(parse_f64(field).ok_or_else(|| {
                field_error(line, &col.name, "16 lowercase hex digits of bits, a space, a decimal", field)
            })?),
            ColumnData::Str(v) => push_str(line, &col.name, field, v, buf)?,
        }
    }
    Ok(())
}

fn push_str(line: usize, name: &str, field: &str, out: &mut Strs, buf: &mut String) -> Result<(), GoldenError> {
    if !field.contains('\\') && !field.bytes().any(|b| b < 0x20) {
        out.push(field);
        return Ok(());
    }
    buf.clear();
    unescape_into(field, buf).map_err(|error| GoldenError::Escape { line, field: name.into(), error })?;
    out.push(buf);
    Ok(())
}

/// Decimal, no sign, no leading zero except `0` itself, at most u32::MAX.
fn parse_u32(s: &str) -> Option<u32> {
    let b = s.as_bytes();
    if b.is_empty() || !b.iter().all(u8::is_ascii_digit) || (b.len() > 1 && b[0] == b'0') {
        return None;
    }
    s.parse().ok()
}

/// 16 lowercase hex digits of the raw bits, then a space. What follows the
/// space is the decimal for a person, and is not read.
fn parse_f64(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    if b.len() < 17 || b[16] != b' ' {
        return None;
    }
    let bits = b[..16].iter().try_fold(0u64, |v, &d| Some((v << 4) | digit(d)? as u64))?;
    Some(f64::from_bits(bits))
}

fn u32_field(line: usize, field: &str, s: &str) -> Result<u32, GoldenError> {
    parse_u32(s).ok_or_else(|| field_error(line, field, "a u32 in decimal, no sign or leading zero", s))
}

fn hex_field<const N: usize>(
    line: usize,
    field: &str,
    expected: &'static str,
    s: &str,
) -> Result<[u8; N], GoldenError> {
    parse_hex::<N>(s).ok_or_else(|| field_error(line, field, expected, s))
}

/// A header text field: written as it is, and never empty.
fn text_field(line: usize, field: &str, s: &str) -> Result<String, GoldenError> {
    if s.is_empty() {
        return Err(field_error(line, field, "a non-empty field", s));
    }
    Ok(s.into())
}

fn field_error(line: usize, field: &str, expected: &'static str, found: &str) -> GoldenError {
    GoldenError::Field { line, field: field.into(), expected, found: found.into() }
}
