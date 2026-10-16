//! Stream readers and sample converters for Penn Treebank parses (`opennlp.tools.parser.ParseSampleStream`,
//! `opennlp.tools.parser.PosSampleStream`, and `opennlp.tools.parser.ChunkSampleStream`).

use turbo_sequence::Features;
use turbo_sequence::postag::DefaultPosContext;
use turbo_train::{Event, JavaString};

use crate::head_rules::{GapLabeler, JavaLineIter};
use crate::parse::{CONT, OTHER, Parse, ParseError, START};

/// Reads non-empty Penn Treebank lines from plain text and parses each into a [`Parse`] tree
/// (`opennlp.tools.parser.ParseSampleStream`).
pub struct ParseSampleStream<'a> {
    lines: JavaLineIter<'a>,
    use_function_tags: bool,
    gap_labeler: Option<&'a dyn GapLabeler>,
}

impl<'a> ParseSampleStream<'a> {
    /// Creates a new stream over the lines of `input` with default options
    /// (`use_function_tags = false`, no gap labeler).
    #[inline]
    pub fn new(input: &'a str) -> Self {
        Self { lines: JavaLineIter::new(input), use_function_tags: false, gap_labeler: None }
    }

    /// Creates a new stream over `input` with explicit `use_function_tags` and optional
    /// `gap_labeler`.
    #[inline]
    pub fn with_options(input: &'a str, use_function_tags: bool, gap_labeler: Option<&'a dyn GapLabeler>) -> Self {
        Self { lines: JavaLineIter::new(input), use_function_tags, gap_labeler }
    }
}

impl<'a> Iterator for ParseSampleStream<'a> {
    type Item = Result<Parse, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        for line in self.lines.by_ref() {
            // Match Java's String.trim().isEmpty(): trim ASCII control chars and space (<= U+0020).
            if !line.trim_matches(|c: char| c <= ' ').is_empty() {
                return Some(Parse::parse_with_options(line, self.use_function_tags, self.gap_labeler));
            }
        }
        None
    }
}

/// POS-tagged sentence extracted from a [`Parse`] tree (`opennlp.tools.parser.PosSampleStream`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParserPosSample {
    /// Token surface strings in left-to-right order.
    pub sentence: Vec<String>,
    /// POS tags corresponding 1-to-1 with `sentence`.
    pub tags: Vec<String>,
}

impl ParserPosSample {
    /// Extracts a [`ParserPosSample`] from the POS tag nodes of `parse` (`PosSampleStream.read`).
    pub fn from_parse(parse: &Parse) -> Self {
        let nodes = parse.tag_nodes();
        let mut sentence = Vec::with_capacity(nodes.len());
        let mut tags = Vec::with_capacity(nodes.len());
        for node in nodes {
            sentence.push(node.covered_text().to_string());
            tags.push(node.kind().to_string());
        }
        Self { sentence, tags }
    }
}

/// Basal chunk sample extracted from a [`Parse`] tree (`opennlp.tools.parser.ChunkSampleStream`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParserChunkSample {
    /// Token surface strings in left-to-right order.
    pub sentence: Vec<String>,
    /// POS tags corresponding 1-to-1 with `sentence`.
    pub tags: Vec<String>,
    /// Basal chunk outcomes (`S-<type>`, `C-<type>`, or `O`) corresponding 1-to-1 with `sentence`.
    pub preds: Vec<String>,
}

impl ParserChunkSample {
    /// Extracts basal chunk outcomes from `parse` (`ChunkSampleStream.read`).
    pub fn from_parse(parse: &Parse) -> Self {
        let chunks = parse.initial_chunks();
        let mut sentence = Vec::new();
        let mut tags = Vec::new();
        let mut preds = Vec::new();
        for c in chunks {
            if c.is_pos_tag() {
                sentence.push(c.covered_text().to_string());
                tags.push(c.kind().to_string());
                preds.push(OTHER.to_string());
            } else {
                let mut start = true;
                let ctype = c.kind();
                for tok in c.children() {
                    sentence.push(tok.covered_text().to_string());
                    tags.push(tok.kind().to_string());
                    if start {
                        let mut pred = String::with_capacity(START.len() + ctype.len());
                        pred.push_str(START);
                        pred.push_str(ctype);
                        preds.push(pred);
                        start = false;
                    } else {
                        let mut pred = String::with_capacity(CONT.len() + ctype.len());
                        pred.push_str(CONT);
                        pred.push_str(ctype);
                        preds.push(pred);
                    }
                }
            }
        }
        Self { sentence, tags, preds }
    }
}

/// Adds the part-of-speech events of `p`, a parse pruned and with its heads updated, to `events`
/// (`AbstractParserEventStream.addTagEvents`): one per token of its initial chunks, the outcome
/// the token's tag and the context what `DefaultPOSContextGenerator(null)` makes of the tokens and
/// all of the sentence's tags.
pub(crate) fn add_tag_events(p: &Parse, events: &mut Vec<Event>) {
    let mut tokens = Vec::new();
    let mut tags = Vec::new();
    for chunk in p.initial_chunks() {
        let nodes = if chunk.is_pos_tag() { std::slice::from_ref(chunk) } else { chunk.children() };
        for node in nodes {
            tokens.push(node.covered_text());
            tags.push(node.kind());
        }
    }
    let mut generator = DefaultPosContext::new(None);
    let mut features = Features::new();
    for (i, tag) in tags.iter().enumerate() {
        features.clear();
        generator.features(&mut features, i, &tokens, &tags);
        let context =
            features.iter().map(|f| JavaString::from_modified_utf8(f).expect("a feature is modified UTF-8")).collect();
        events.push(Event::new(*tag, context));
    }
}
