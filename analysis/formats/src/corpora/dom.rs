//! The part of `org.w3c.dom` the NKJP and Irish Sentence Bank readers
//! walk: a tree of the document as a non-validating, namespace-unaware
//! `DocumentBuilder` builds it, made with the XML reader of
//! `turbo-maxent`.
//!
//! Each run of text between two pieces of markup is one text node, with
//! its references resolved, as the JDK's parser builds it (it does not
//! split a text node at `&amp;`). CDATA sections, comments and processing
//! instructions are nodes of their own. Names are qualified names, prefix
//! and all. Every node keeps the bytes of the source it was read from.

use turbo_analysis::Span;
use turbo_maxent::xml::{self, Attribute, Handler, Names};

/// A node of the tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Node {
    Element(Element),
    Text(CharData),
    CData(CharData),
    /// A comment, with its text between `<!--` and `-->`.
    Comment(CharData),
    /// A processing instruction: its target, and its data with the
    /// spaces after the target removed.
    Pi {
        target: String,
        data: CharData,
    },
}

/// The characters of a text, CDATA, comment or processing instruction
/// node, and the bytes of the node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CharData {
    pub text: String,
    pub span: Span,
}

/// An element: its qualified name, its attributes in order (without the
/// `xmlns` declarations, which no reader asks for), its children and the
/// bytes from its start tag to the end of its end tag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Element {
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<Node>,
    pub span: Span,
}

impl Node {
    /// `Node.getNodeName`.
    pub(crate) fn name(&self) -> &str {
        match self {
            Node::Element(e) => &e.name,
            Node::Text(_) => "#text",
            Node::CData(_) => "#cdata-section",
            Node::Comment(_) => "#comment",
            Node::Pi { target, .. } => target,
        }
    }

    pub(crate) fn as_element(&self) -> Option<&Element> {
        match self {
            Node::Element(e) => Some(e),
            _ => None,
        }
    }

    /// `Node.getTextContent`: the text of a text, CDATA, comment or
    /// processing instruction node, and for an element the text and CDATA
    /// of all its descendants, in order.
    pub(crate) fn push_text_content(&self, out: &mut String) {
        match self {
            Node::Element(e) => e.push_text_content(out),
            Node::Text(d) | Node::CData(d) | Node::Comment(d) | Node::Pi { data: d, .. } => out.push_str(&d.text),
        }
    }

    /// The bytes of the node.
    pub(crate) fn span(&self) -> Span {
        match self {
            Node::Element(e) => e.span,
            Node::Text(d) | Node::CData(d) | Node::Comment(d) | Node::Pi { data: d, .. } => d.span,
        }
    }
}

impl Element {
    /// The value of the attribute named `name`, as written with its
    /// prefix.
    pub(crate) fn attribute(&self, name: &str) -> Option<&str> {
        self.attributes.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }

    /// The child elements named `name`, in order: the XPath step
    /// `./name`.
    pub(crate) fn children_named<'e>(&'e self, name: &'e str) -> impl Iterator<Item = &'e Element> + 'e {
        self.children.iter().filter_map(Node::as_element).filter(move |e| e.name == name)
    }

    /// `Element.getTextContent`.
    pub(crate) fn push_text_content(&self, out: &mut String) {
        for child in &self.children {
            match child {
                Node::Comment(_) | Node::Pi { .. } => {}
                other => other.push_text_content(out),
            }
        }
    }

    pub(crate) fn text_content(&self) -> String {
        let mut out = String::new();
        self.push_text_content(&mut out);
        out
    }
}

/// The root element of `text`, or what the XML reader found wrong.
pub(crate) fn parse(text: &str) -> Result<Element, turbo_maxent::Error> {
    let mut builder = Builder { source: text, stack: Vec::new(), pending: String::new(), root: None };
    xml::parse_with(text, Names::Qualified, &mut builder)?;
    Ok(builder.root.expect("a document parsed without error has a root element"))
}

struct Builder<'s> {
    source: &'s str,
    stack: Vec<Element>,
    /// The characters of the text or CDATA node being read.
    pending: String,
    root: Option<Element>,
}

impl Builder<'_> {
    fn push(&mut self, node: Node) {
        if let Some(parent) = self.stack.last_mut() {
            parent.children.push(node);
        }
    }

    fn chars(&mut self, start: usize, end: usize) -> CharData {
        CharData { text: std::mem::take(&mut self.pending), span: Span::new(start as u32, end as u32) }
    }
}

impl Handler for Builder<'_> {
    fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], offset: usize) -> turbo_maxent::Result<()> {
        self.stack.push(Element {
            name: name.to_owned(),
            attributes: attributes.iter().map(|a| (a.name.to_owned(), a.value.clone().into_owned())).collect(),
            children: Vec::new(),
            span: Span::new(offset as u32, offset as u32),
        });
        Ok(())
    }

    fn end_element(&mut self, _: &str, _: usize) -> turbo_maxent::Result<()> {
        Ok(())
    }

    fn element_closed(&mut self, end: usize) {
        let mut element = self.stack.pop().expect("an element closes after it opens");
        element.span = Span::new(element.span.start, end as u32);
        if self.stack.is_empty() {
            self.root = Some(element);
        } else {
            self.push(Node::Element(element));
        }
    }

    fn characters(&mut self, text: &str) {
        self.pending.push_str(text);
    }

    fn text_run(&mut self, start: usize, end: usize) {
        let data = self.chars(start, end);
        self.push(Node::Text(data));
    }

    fn cdata_section(&mut self, start: usize, end: usize) {
        let data = self.chars(start, end);
        self.push(Node::CData(data));
    }

    fn comment(&mut self, start: usize, end: usize) {
        if self.stack.is_empty() {
            return;
        }
        let text = self.source[start + 4..end - 3].to_owned();
        self.push(Node::Comment(CharData { text, span: Span::new(start as u32, end as u32) }));
    }

    fn processing_instruction(&mut self, target: &str, start: usize, end: usize) {
        if self.stack.is_empty() {
            return;
        }
        let body = &self.source[start + 2 + target.len()..end - 2];
        let data = body.trim_start_matches([' ', '\t', '\n', '\r']).to_owned();
        let data = CharData { text: data, span: Span::new(start as u32, end as u32) };
        self.push(Node::Pi { target: target.to_owned(), data });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_runs_are_nodes_with_their_bytes() {
        let src = "<r a='1'>x &amp; y<e/><![CDATA[c]]><!--k--><?p d?>z</r>";
        let root = parse(src).unwrap();
        assert_eq!(root.span, Span::new(0, src.len() as u32));
        assert_eq!(root.attribute("a"), Some("1"));
        let names: Vec<&str> = root.children.iter().map(Node::name).collect();
        assert_eq!(names, ["#text", "e", "#cdata-section", "#comment", "p", "#text"]);
        let Node::Text(t) = &root.children[0] else { panic!() };
        assert_eq!((t.text.as_str(), &src[t.span.range()]), ("x & y", "x &amp; y"));
        assert_eq!(&src[root.children[1].span().range()], "<e/>");
        let mut content = String::new();
        root.children[3].push_text_content(&mut content);
        root.children[4].push_text_content(&mut content);
        assert_eq!(content, "kd");
        assert_eq!(root.text_content(), "x & ycz");
    }
}
