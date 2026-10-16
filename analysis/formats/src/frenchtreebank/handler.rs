//! `ConstitDocumentHandler`: the SAX handler that builds a parse of each
//! `SENT` element.

use std::sync::Arc;

use turbo_analysis::Span;
use turbo_maxent::xml::{Attribute, Handler, Names, parse_with};
use turbo_parser::{Parse, TOK_NODE, TOP_NODE, TreeBuilder};

use super::{ConstitParse, FtbError};

const SENT_ELEMENT: &str = "SENT";
const WORD_ELEMENT: &str = "w";
const SENT_TYPE: &str = "S";

/// An element whose node is not made yet: its type and where it starts in
/// the sentence's text.
struct Open {
    label: String,
    start: u32,
}

/// A node of the sentence: its type and span in the sentence's text.
struct Node {
    label: String,
    span: Span,
}

/// What `characters` collected since the last tag, and the bytes it was
/// read from.
#[derive(Default)]
struct Text {
    chars: String,
    /// The bytes from the start of the first run of text or CDATA section
    /// read into `chars` to the end of the last.
    source: Option<Span>,
    /// How many runs and sections were read into `chars`.
    runs: u32,
    /// The length of `chars` when the last run or section ended.
    mark: usize,
    /// Whether every run read was written as it reads, so a part of
    /// `chars` is the same part of the source.
    verbatim: bool,
}

impl Text {
    fn clear(&mut self) {
        self.chars.clear();
        self.source = None;
        self.runs = 0;
        self.mark = 0;
        self.verbatim = true;
    }

    /// A run of the document at `start..end` ended; its characters are
    /// those read since the last one.
    fn run(&mut self, xml: &str, start: usize, end: usize, cdata: bool) {
        let read = &self.chars[self.mark..];
        self.verbatim &= !cdata && read == &xml[start..end];
        self.mark = self.chars.len();
        self.runs += 1;
        let span = Span::new(start as u32, end as u32);
        self.source = Some(self.source.map_or(span, |s| Span::new(s.start, span.end)));
    }
}

struct ConstitHandler<'x> {
    xml: &'x str,
    parses: Vec<ConstitParse>,
    error: Option<FtbError>,
    inside_sentence: bool,
    buffer: Text,
    /// The sentence's text: its words, each followed by a space.
    text: String,
    offset: u32,
    stack: Vec<Open>,
    nodes: Vec<Node>,
    /// The source of each word of the sentence.
    tokens: Vec<Span>,
    /// Where the open `SENT` element starts in the document.
    sentence_start: u32,
    /// Whether the element just ended was the `SENT` whose parse was made
    /// last, so its end can be recorded.
    sentence_ended: bool,
    cat: Option<String>,
    subcat: Option<String>,
}

fn value<'v>(attributes: &'v [Attribute<'_>], name: &str) -> Option<&'v str> {
    attributes.iter().find(|a| a.name == name).map(|a| a.value.as_ref())
}

/// `String.trim`: every leading and trailing character up to U+0020
/// removed. The byte offset of what is left, and what is left.
fn trim(s: &str) -> (usize, &str) {
    let start = s.len() - s.trim_start_matches(|c: char| c <= ' ').len();
    (start, s[start..].trim_end_matches(|c: char| c <= ' '))
}

impl ConstitHandler<'_> {
    /// The type of a `w` element, from its attributes and those of the
    /// words before it.
    fn word_type(&mut self, attributes: &[Attribute<'_>]) -> String {
        if let Some(cat) = value(attributes, "cat").filter(|c| !c.is_empty()) {
            self.cat = Some(cat.to_owned());
        }
        if let Some(subcat) = value(attributes, "subcat").filter(|s| !s.is_empty()) {
            self.subcat = Some(subcat.to_owned());
        }
        match (&self.cat, value(attributes, "catint")) {
            (Some(cat), _) => [cat.as_str(), self.subcat.as_deref().unwrap_or("")].concat(),
            (None, Some(catint)) => ["null", catint].concat(),
            (None, None) => ["null", self.subcat.as_deref().unwrap_or("null")].concat(),
        }
    }

    /// The parse of the sentence just ended.
    fn finish_sentence(&mut self, end: usize) {
        let text: Arc<str> = std::mem::take(&mut self.text).into();
        // The nodes come as the elements close them, which the builder
        // inserts in time linear in their number.
        let mut parse =
            TreeBuilder::new(Parse::new(Arc::clone(&text), Span::new(0, text.len() as u32), TOP_NODE, 1.0, 0));
        let source = Span::new(self.sentence_start, end as u32);
        let mut token_index = -1i32;
        for node in self.nodes.drain(..) {
            if node.label == TOP_NODE {
                continue;
            }
            if node.label == TOK_NODE {
                token_index += 1;
            }
            let child = Parse::new(Arc::clone(&text), node.span, node.label, 1.0, token_index);
            if let Err(error) = parse.insert(child) {
                self.error.get_or_insert(FtbError::Parse { sentence: source, error });
            }
        }
        self.parses.push(ConstitParse { parse: parse.finish(), source, tokens: std::mem::take(&mut self.tokens) });
    }
}

impl Handler for ConstitHandler<'_> {
    fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], offset: usize) -> turbo_maxent::Result<()> {
        let label = if name == SENT_ELEMENT {
            self.text.clear();
            self.offset = 0;
            self.stack.clear();
            self.nodes.clear();
            self.tokens.clear();
            self.sentence_start = offset as u32;
            self.inside_sentence = true;
            SENT_TYPE.to_owned()
        } else if name == WORD_ELEMENT {
            // The categories carry over from word to word, so they are read
            // inside a sentence or not.
            self.word_type(attributes)
        } else {
            name.to_owned()
        };
        // Only an element opened inside a sentence is ever taken off the
        // stack: a SENT clears it, and nothing is taken off outside one.
        if self.inside_sentence {
            self.stack.push(Open { label, start: self.offset });
        }
        self.buffer.clear();
        Ok(())
    }

    fn characters(&mut self, text: &str) {
        self.buffer.chars.push_str(text);
    }

    fn text_run(&mut self, start: usize, end: usize) {
        self.buffer.run(self.xml, start, end, false);
    }

    fn cdata_section(&mut self, start: usize, end: usize) {
        self.buffer.run(self.xml, start, end, true);
    }

    fn end_element(&mut self, name: &str, _offset: usize) -> turbo_maxent::Result<()> {
        self.sentence_ended = false;
        if !self.inside_sentence {
            return Ok(());
        }
        let mut create = true;
        if name == WORD_ELEMENT {
            let (lead, token) = trim(&self.buffer.chars);
            if token.is_empty() {
                create = false;
            } else {
                let len = token.len() as u32;
                self.nodes.push(Node { label: TOK_NODE.to_owned(), span: Span::new(self.offset, self.offset + len) });
                let source = match self.buffer.source {
                    Some(s) if self.buffer.runs == 1 && self.buffer.verbatim => {
                        Span::new(s.start + lead as u32, s.start + lead as u32 + len)
                    }
                    Some(s) => s,
                    None => Span::new(self.sentence_start, self.sentence_start),
                };
                self.tokens.push(source);
                self.text.push_str(token);
                self.text.push(' ');
                self.offset += len + 1;
            }
        }
        if let Some(open) = self.stack.pop()
            && create
            && open.start < self.offset
        {
            self.nodes.push(Node { label: open.label, span: Span::new(open.start, self.offset - 1) });
        }
        if name == SENT_ELEMENT {
            self.inside_sentence = false;
            self.sentence_ended = true;
        }
        self.buffer.clear();
        Ok(())
    }

    fn element_closed(&mut self, end: usize) {
        if self.sentence_ended {
            self.sentence_ended = false;
            self.finish_sentence(end);
        }
    }
}

/// Reads the parses of `xml` (no longer than `u32::MAX` bytes).
pub(super) fn read(xml: &str) -> Result<Vec<ConstitParse>, FtbError> {
    let mut handler = ConstitHandler {
        xml,
        parses: Vec::new(),
        error: None,
        inside_sentence: false,
        buffer: Text { verbatim: true, ..Text::default() },
        text: String::new(),
        offset: 0,
        stack: Vec::new(),
        nodes: Vec::new(),
        tokens: Vec::new(),
        sentence_start: 0,
        sentence_ended: false,
        cat: None,
        subcat: None,
    };
    parse_with(xml, Names::Qualified, &mut handler).map_err(FtbError::Xml)?;
    match handler.error {
        Some(e) => Err(e),
        None => Ok(handler.parses),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trees(xml: &str) -> Vec<String> {
        read(xml).unwrap().iter().map(|p| p.parse().to_string_penn_treebank()).collect()
    }

    #[test]
    fn reads_sentences_into_trees() {
        let xml = "<text><SENT nb=\"1\"><NP><w cat=\"D\" subcat=\"def\">le</w> <w cat=\"N\" subcat=\"C\">chat</w></NP>\
                   <VN><w cat=\"V\">dort</w></VN><w cat=\"PONCT\">.</w></SENT></text>";
        assert_eq!(trees(xml), ["(TOP (S (NP (Ddef le) (NC chat)) (VN (VC dort)) (PONCTC .)) )"]);
    }

    #[test]
    fn categories_carry_over_and_start_as_null() {
        let xml = "<t><SENT><w catint=\"P\">a</w><w subcat=\"X\">b</w><w>c</w></SENT></t>";
        assert_eq!(trees(xml), ["(TOP (S (nullP a) (nullX b) (nullX c)) )"]);
        let xml = "<t><SENT><w>a</w></SENT></t>";
        assert_eq!(trees(xml), ["(TOP (S (nullnull a)) )"]);
    }

    #[test]
    fn compounds_are_nodes_only_with_text_after_their_words() {
        let xml = "<t><SENT><w cat=\"P\"><w catint=\"P\">au</w> <w catint=\"N\">cours</w></w>\
                   <w cat=\"A\"><w>tout</w> fait</w></SENT></t>";
        assert_eq!(trees(xml), ["(TOP (S (P au) (P cours) (A (A tout) fait)) )"]);
    }

    #[test]
    fn token_sources_point_into_the_document() {
        let xml =
            "<t><SENT><w cat=\"N\">\n  chat </w><w cat=\"N\">a&amp;b</w><w cat=\"N\"><![CDATA[x]]></w></SENT></t>";
        let p = &read(xml).unwrap()[0];
        let sources: Vec<&str> = p.tokens().iter().map(|s| &xml[s.range()]).collect();
        assert_eq!(sources, ["chat", "a&amp;b", "<![CDATA[x]]>"]);
        assert_eq!(&xml[p.source().range()], &xml[3..xml.len() - 4]);
        assert_eq!(p.parse().text(), "chat a&b x ");
    }

    #[test]
    fn a_nested_sentence_starts_again() {
        let xml = "<t><SENT><w cat=\"N\">a</w><SENT><w>b</w></SENT><w>c</w></SENT><SENT></SENT></t>";
        assert_eq!(trees(xml), ["(TOP (S (N b)) )", "(TOP )"]);
    }

    #[test]
    fn malformed_documents_give_no_parse() {
        assert!(matches!(read("<t><SENT><w>a</w></SENT><SENT></t>"), Err(FtbError::Xml(_))));
    }
}
