//! The descriptor as a small element tree, the part of the DOM that
//! `GeneratorFactory` reads: elements with their attributes, and the text
//! between them.

use turbo_maxent::xml::{self, Attribute, Handler};

use crate::error::Result;

/// An element of the descriptor.
#[derive(Debug, Default)]
pub(super) struct Element {
    /// The local name (a namespace prefix is dropped by the reader).
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<Node>,
}

#[derive(Debug)]
pub(super) enum Node {
    Element(Element),
    /// A run of character data. Adjacent runs (text, references, CDATA)
    /// are one node, which is what `Text.getWholeText` returns.
    Text(String),
}

impl Element {
    /// `Element.getAttribute`: the value, or the empty string when the
    /// attribute is absent.
    pub fn attribute(&self, name: &str) -> &str {
        self.attributes.iter().rev().find(|(k, _)| k == name).map_or("", |(_, v)| v.as_str())
    }

    /// The child elements in document order.
    pub fn elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|n| match n {
            Node::Element(e) => Some(e),
            Node::Text(_) => None,
        })
    }
}

/// Reads the descriptor's XML into its root element.
pub(super) fn parse(bytes: &[u8]) -> Result<Element> {
    let text = xml::decode(bytes)?;
    let mut builder = Builder { stack: vec![Element::default()] };
    xml::parse(&text, &mut builder)?;
    let mut document = builder.stack.pop().expect("the document node");
    let root = document.children.drain(..).find_map(|n| match n {
        Node::Element(e) => Some(e),
        Node::Text(_) => None,
    });
    Ok(root.expect("a well-formed document has a root element"))
}

struct Builder {
    /// The open elements, the document node first.
    stack: Vec<Element>,
}

impl Handler for Builder {
    fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], _offset: usize) -> turbo_maxent::Result<()> {
        let attributes = attributes.iter().map(|a| (a.name.to_owned(), a.value.to_string())).collect();
        self.stack.push(Element { name: name.to_owned(), attributes, children: Vec::new() });
        Ok(())
    }

    fn end_element(&mut self, _name: &str, _offset: usize) -> turbo_maxent::Result<()> {
        let element = self.stack.pop().expect("an element is open");
        self.stack.last_mut().expect("the document node").children.push(Node::Element(element));
        Ok(())
    }

    fn characters(&mut self, text: &str) {
        let children = &mut self.stack.last_mut().expect("the document node").children;
        match children.last_mut() {
            Some(Node::Text(t)) => t.push_str(text),
            _ => children.push(Node::Text(text.to_owned())),
        }
    }
}
