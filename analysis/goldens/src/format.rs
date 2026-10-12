//! A golden file as typed values: the header records and the layer
//! blocks, each layer stored by column. `read` parses the text form into
//! these and `write` writes it back byte for byte.

use std::fmt;

use turbo_analysis_layers::Strs;

/// The format version the first line names (`golden 1`). A reader refuses
/// any other.
pub const VERSION: u32 = 1;

/// A whole golden file.
#[derive(Clone, Debug, PartialEq)]
pub struct Golden {
    pub header: Header,
    /// In file order: an aligned layer comes after the layer it is
    /// aligned to.
    pub layers: Vec<GoldenLayer>,
}

impl Golden {
    /// The layer with this id, if the file has it.
    pub fn layer(&self, id: &str) -> Option<&GoldenLayer> {
        self.layers.iter().find(|l| l.id == id)
    }
}

/// Every header record, typed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub generator: Generator,
    pub opennlp: OpenNlp,
    pub jdk: Jdk,
    /// The Unicode version the JDK's character tables implement.
    pub unicode: UnicodeVersion,
    pub input: Input,
    /// One per model file the annotator loaded, in the order loaded.
    pub models: Vec<Model>,
    pub annotator: AnnotatorRef,
    /// The options the annotator was run with, sorted by key, so two
    /// goldens of one class run with different options are told apart.
    pub config: Vec<ConfigEntry>,
}

/// What wrote the file: `generator <name> <version>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generator {
    pub name: String,
    pub version: u32,
}

/// The OpenNLP build that ran: `opennlp <commit> <maven version>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenNlp {
    /// The git commit, 40 lowercase hex digits in the file.
    pub commit: [u8; 20],
    pub version: String,
}

/// The JVM that ran: `jdk <java.version> <java.vm.version> <java.vendor>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Jdk {
    pub version: String,
    pub vm_version: String,
    pub vendor: String,
}

/// `unicode <major>.<minor>`, such as 15.0 for JDK 21.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnicodeVersion {
    pub major: u32,
    pub minor: u32,
}

impl fmt::Display for UnicodeVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// The input text: `input <set>/<id> <utf8 len> <utf16 len> <sha256>`. Its
/// file is `inputs/<set>/<id>.txt` under the data directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    pub set: String,
    pub id: String,
    /// Its length in UTF-8 bytes, the unit of every offset in the file.
    pub utf8_len: u32,
    /// Its length in UTF-16 units, the unit OpenNLP's spans were in.
    pub utf16_len: u32,
    pub sha256: [u8; 32],
}

/// A model file the annotator loaded: `model <file name> <sha512>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    pub file: String,
    pub sha512: [u8; 64],
}

/// The Java method that produced the layers: `annotator <class> <method>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnotatorRef {
    /// The fully qualified class, such as
    /// `opennlp.tools.sentdetect.SentenceDetectorME`.
    pub class: String,
    pub method: String,
}

/// One option the annotator was run with: `config <key> <value>`, both
/// `str` fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigEntry {
    pub key: String,
    pub value: String,
}

/// What a golden layer's rows hang on, as the `layer` record writes it:
/// `text`, `aligned:<layer id>` or `document`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoldenAnchor {
    Text,
    /// Row i describes row i of the named layer.
    Aligned(String),
    Document,
}

impl fmt::Display for GoldenAnchor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GoldenAnchor::Text => f.write_str("text"),
            GoldenAnchor::Aligned(of) => write!(f, "aligned:{of}"),
            GoldenAnchor::Document => f.write_str("document"),
        }
    }
}

/// One layer block: its id, its anchor and its columns, every column as
/// long as the layer has rows.
#[derive(Clone, Debug, PartialEq)]
pub struct GoldenLayer {
    pub id: String,
    pub anchor: GoldenAnchor,
    pub columns: Vec<Column>,
}

impl GoldenLayer {
    /// How many rows it has: the length of its first column. A layer
    /// read from a file has at least one column, all of one length.
    pub fn len(&self) -> usize {
        self.columns.first().map_or(0, |c| c.data.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// The column with this name, if it is a `u32` column.
    pub fn u32s(&self, name: &str) -> Option<&[u32]> {
        match &self.column(name)?.data {
            ColumnData::U32(v) => Some(v),
            _ => None,
        }
    }

    /// The column with this name, if it is an `f64` column.
    pub fn f64s(&self, name: &str) -> Option<&[f64]> {
        match &self.column(name)?.data {
            ColumnData::F64(v) => Some(v),
            _ => None,
        }
    }

    /// The column with this name, if it is a `str` column.
    pub fn strs(&self, name: &str) -> Option<&Strs> {
        match &self.column(name)?.data {
            ColumnData::Str(v) => Some(v),
            _ => None,
        }
    }

    /// The columns as `name:type`, in order, for messages.
    pub fn column_names(&self) -> Vec<String> {
        self.columns.iter().map(|c| format!("{}:{}", c.name, c.data.ty())).collect()
    }
}

/// A named, typed column.
#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    pub name: String,
    pub data: ColumnData,
}

/// A column's values. Strings are stored by column too, one buffer for
/// the whole column.
#[derive(Clone, Debug)]
pub enum ColumnData {
    U32(Vec<u32>),
    F64(Vec<f64>),
    Str(Strs),
}

impl ColumnData {
    pub fn ty(&self) -> ColumnType {
        match self {
            ColumnData::U32(_) => ColumnType::U32,
            ColumnData::F64(_) => ColumnType::F64,
            ColumnData::Str(_) => ColumnType::Str,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ColumnData::U32(v) => v.len(),
            ColumnData::F64(v) => v.len(),
            ColumnData::Str(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An empty column of a type.
    pub fn empty(ty: ColumnType) -> ColumnData {
        match ty {
            ColumnType::U32 => ColumnData::U32(Vec::new()),
            ColumnType::F64 => ColumnData::F64(Vec::new()),
            ColumnType::Str => ColumnData::Str(Strs::default()),
        }
    }
}

/// Equal when every value is: f64 values by their bits, so a file read
/// back is equal to the one written, NaN and -0.0 included.
impl PartialEq for ColumnData {
    fn eq(&self, other: &ColumnData) -> bool {
        match (self, other) {
            (ColumnData::U32(a), ColumnData::U32(b)) => a == b,
            (ColumnData::F64(a), ColumnData::F64(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (ColumnData::Str(a), ColumnData::Str(b)) => a.len() == b.len() && a.iter().eq(b.iter()),
            _ => false,
        }
    }
}

/// The three column types.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColumnType {
    /// Decimal, no sign, no leading zeros except `0` itself.
    U32,
    /// 16 lowercase hex digits of the IEEE 754 bits, a space, then Java's
    /// `Double.toString` for a person to read. The bits are the value.
    F64,
    /// UTF-8 text, escaped (see `escape`).
    Str,
}

impl ColumnType {
    pub fn name(self) -> &'static str {
        match self {
            ColumnType::U32 => "u32",
            ColumnType::F64 => "f64",
            ColumnType::Str => "str",
        }
    }

    pub fn parse(s: &str) -> Option<ColumnType> {
        match s {
            "u32" => Some(ColumnType::U32),
            "f64" => Some(ColumnType::F64),
            "str" => Some(ColumnType::Str),
            _ => None,
        }
    }
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A column a layer type expects: its name and type.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ColumnSpec {
    pub name: &'static str,
    pub ty: ColumnType,
}

impl ColumnSpec {
    pub const fn new(name: &'static str, ty: ColumnType) -> ColumnSpec {
        ColumnSpec { name, ty }
    }
}

impl fmt::Display for ColumnSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.name, self.ty)
    }
}
