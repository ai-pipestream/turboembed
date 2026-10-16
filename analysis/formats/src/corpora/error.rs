//! What the sentence, language and document corpus readers refuse, with
//! the exception OpenNLP throws in the same place.

use std::fmt;

/// An error of the readers of [`crate::corpora`].
///
/// [`CorpusError::java_class`] is the class of the exception OpenNLP
/// throws in the same place, and `Display` is its message. The messages
/// of `NullPointerException` are the JDK's helpful messages as JDK 21
/// writes them for OpenNLP's classes compiled with local variable names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CorpusError {
    /// A LetsMT document that is not well-formed XML: `LetsmtDocument`
    /// wraps the `SAXException`. `detail` is what the XML reader found,
    /// which Java's message does not hold.
    LetsmtXml { detail: String },
    /// An NKJP file that is not well-formed XML, or that misses what the
    /// reader asks for: `NKJPSegmentationDocument` and `NKJPTextDocument`
    /// wrap every `SAXException` and `IOException` of their parse in one
    /// message. `detail` says what was wrong, which Java's message does
    /// not hold.
    NkjpDocument { detail: String },
    /// A paragraph of an NKJP text file with no content:
    /// `NKJPTextDocument` asks for the name of its first child, which is
    /// null.
    NkjpEmptyParagraph,
    /// A number of an NKJP `corresp` pointer or an Irish Sentence Bank
    /// `slot` that `Integer.parseInt` refuses. `text` is the string.
    NotANumber { text: String },
    /// An NKJP pointer whose offset and length make no span: the
    /// `IllegalArgumentException` of `Span`'s constructor.
    NkjpSpan { start: i32, end: i32 },
    /// An NKJP pointer into a paragraph the text file does not have.
    NkjpMissingParagraph,
    /// An NKJP sentence whose segments cut a paragraph outside its text,
    /// or backwards: `String.substring` throws. The numbers are UTF-16
    /// code units.
    NkjpSubstring { begin: i32, end: i32, length: usize },
    /// An NKJP segment boundary between the two halves of a surrogate
    /// pair. Java cuts the pair and keeps half of it in the sentence; a
    /// Rust string cannot hold half a pair, so this port refuses the
    /// sentence instead.
    SplitsSurrogatePair,
    /// An Irish Sentence Bank document that is not well-formed XML:
    /// `IrishSentenceBankDocument` wraps the `SAXException`. `detail` is
    /// what the XML reader found.
    IrishXml { detail: String },
    /// An Irish Sentence Bank root element not named `sentences` in any
    /// case.
    IrishRoot { name: String },
    /// A node the Irish Sentence Bank reader does not expect: an element
    /// other than `sentence` in the root, other than `flex`,
    /// `translation` and `original` in a sentence, other than `token` in
    /// an original, or a CDATA section, comment or processing
    /// instruction (by its target) where only text is read.
    IrishUnexpectedNode { name: String },
    /// An Irish Sentence Bank element without an attribute the reader
    /// asks for (`source`, `slot` or `lemma`).
    IrishMissingAttribute,
    /// A processing instruction with the name of a `sentence`, `flex` or
    /// `token` element, which has no attributes to read.
    IrishNoAttributes,
    /// An Irish Sentence Bank `translation` or `token` element with no
    /// content.
    IrishNoContent,
    /// Text between the tokens of an Irish Sentence Bank original that is
    /// two or more spaces: the span the reader makes for it ends before
    /// it starts. The numbers are UTF-16 code units.
    IrishBackwardsSpan { start: usize, end: usize },
    /// An Irish Sentence Bank token whose slot is zero or less, so the
    /// reader writes its flex out of the array's bounds.
    IrishFlexIndex { index: i64, len: usize },
    /// An Irish Sentence Bank token span that lands inside a surrogate
    /// pair of the original (a second `original` element starts its
    /// offsets again from zero). Java keeps the span; a Rust string
    /// cannot be cut there, so this port refuses the sentence.
    IrishTokenSplitsSurrogatePair,
    /// A Leipzig sentences file with fewer lines than its language's
    /// samples need. `path` is the file as Java names it.
    LeipzigTooFewLines { path: String, lines: usize, required: u64 },
    /// Leipzig parameters whose product does not fit a Java `int`. Java
    /// computes it with overflow; this port refuses it.
    LeipzigParameters,
}

impl CorpusError {
    /// The class of the exception OpenNLP throws in the same place, for
    /// the errors that have one.
    pub fn java_class(&self) -> Option<&'static str> {
        use CorpusError::*;
        match self {
            LetsmtXml { .. } | NkjpDocument { .. } | IrishXml { .. } | IrishRoot { .. } => Some("java.io.IOException"),
            IrishUnexpectedNode { .. } => Some("java.io.IOException"),
            NkjpEmptyParagraph | NkjpMissingParagraph | IrishNoAttributes | IrishMissingAttribute | IrishNoContent => {
                Some("java.lang.NullPointerException")
            }
            NotANumber { .. } => Some("java.lang.NumberFormatException"),
            NkjpSpan { .. } | IrishBackwardsSpan { .. } => Some("java.lang.IllegalArgumentException"),
            NkjpSubstring { .. } => Some("java.lang.StringIndexOutOfBoundsException"),
            IrishFlexIndex { .. } => Some("java.lang.ArrayIndexOutOfBoundsException"),
            LeipzigTooFewLines { .. } => Some("opennlp.tools.util.InvalidFormatException"),
            SplitsSurrogatePair | IrishTokenSplitsSurrogatePair | LeipzigParameters => None,
        }
    }
}

impl fmt::Display for CorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use CorpusError::*;
        match self {
            LetsmtXml { .. } => f.write_str("Failed to parse letsmt xml!"),
            NkjpDocument { .. } => f.write_str("Failed to parse NKJP document"),
            NkjpEmptyParagraph => f.write_str(
                "Cannot invoke \"org.w3c.dom.Node.getNodeName()\" because the return value of \
                 \"org.w3c.dom.Node.getFirstChild()\" is null",
            ),
            NotANumber { text } => write!(f, "For input string: \"{text}\""),
            NkjpSpan { start, end } => {
                if *start < 0 {
                    write!(f, "start index must be zero or greater: {start}")
                } else if *end < 0 {
                    write!(f, "end index must be zero or greater: {end}")
                } else {
                    write!(f, "start index must not be larger than end index: start={start}, end={end}")
                }
            }
            NkjpMissingParagraph => {
                f.write_str("Cannot invoke \"String.substring(int, int)\" because \"currentParagraph\" is null")
            }
            NkjpSubstring { begin, end, length } => {
                write!(f, "Range [{begin}, {end}) out of bounds for length {length}")
            }
            SplitsSurrogatePair => f.write_str("an NKJP segment boundary falls inside a surrogate pair"),
            IrishXml { .. } => f.write_str("Failed to parse IrishSentenceBank document"),
            IrishRoot { name } => write!(f, "Expected root node {name}"),
            IrishUnexpectedNode { name } => write!(f, "Unexpected node: {name}"),
            IrishMissingAttribute => f.write_str(
                "Cannot invoke \"org.w3c.dom.Node.getNodeValue()\" because the return value of \
                 \"org.w3c.dom.NamedNodeMap.getNamedItem(String)\" is null",
            ),
            IrishNoAttributes => f.write_str(
                "Cannot invoke \"org.w3c.dom.NamedNodeMap.getNamedItem(String)\" because the return value of \
                 \"org.w3c.dom.Node.getAttributes()\" is null",
            ),
            IrishNoContent => f.write_str(
                "Cannot invoke \"org.w3c.dom.Node.getTextContent()\" because the return value of \
                 \"org.w3c.dom.Node.getFirstChild()\" is null",
            ),
            IrishBackwardsSpan { start, end } => {
                write!(f, "start index must not be larger than end index: start={start}, end={end}")
            }
            IrishFlexIndex { index, len } => write!(f, "Index {index} out of bounds for length {len}"),
            IrishTokenSplitsSurrogatePair => {
                f.write_str("an Irish Sentence Bank token span falls inside a surrogate pair")
            }
            LeipzigTooFewLines { path, lines, required } => {
                write!(f, "{path} does not contain enough lines ({lines} lines < {required} required lines).")
            }
            LeipzigParameters => f.write_str("the Leipzig sample counts do not fit a Java int"),
        }
    }
}

impl std::error::Error for CorpusError {}
