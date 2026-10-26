//! A `DoclingDocument` (docling's JSON export) as a source text and the
//! `Structure` over it.
//!
//! The body is walked in reading order through its `$ref`s. Each text
//! item's `text` is laid down once, blocks apart by a blank line and list
//! items by one line break, and becomes a block: `title` and
//! `section_header` a heading opening a section (a title at depth 1, a
//! header at its `level` plus one), `text`, `paragraph` and `reference` a
//! paragraph, `list_item` an item of the enclosing `list` group, and
//! `caption`, `footnote` and `code` their own kinds. A section runs to the
//! next heading at its depth or above. Furniture (page headers and
//! footers) is not walked. Tables, pictures and any other label are
//! refused by name, never dropped.
//!
//! [`Item`] keeps every text item's reference and its span in the source,
//! so any span of any view maps back to the docling item and the offset
//! inside its `text` ([`Docling::locate`]).

use serde_json::Value;
use std::collections::BTreeSet;
use turbo_analysis::{Ix, Span, Spans, ViewId};
use turbo_analysis_layers::{Block, Structure};

/// One docling text item, where its `text` lies in the source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    /// The item's `self_ref`, `#/texts/N`.
    pub self_ref: String,
    pub label: String,
    pub span: Span,
}

/// The part of one item a source span covers, in bytes, UTF-16 code units
/// and code points of the item's own `text`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Located<'a> {
    pub item: &'a Item,
    pub bytes: (u32, u32),
    pub utf16: (u32, u32),
    pub chars: (u32, u32),
}

pub struct Docling {
    /// The document's `name`.
    pub name: String,
    pub text: String,
    pub items: Vec<Item>,
}

impl Docling {
    /// Every item `source` overlaps, with the part of it covered. The
    /// separators between items belong to none.
    pub fn locate(&self, source: Span) -> Vec<Located<'_>> {
        self.items
            .iter()
            .filter(|i| i.span.start < source.end && source.start < i.span.end)
            .map(|item| {
                let own = &self.text[item.span.range()];
                let a = source.start.max(item.span.start) - item.span.start;
                let b = source.end.min(item.span.end) - item.span.start;
                let units = |n: u32| own[..n as usize].encode_utf16().count() as u32;
                let chars = |n: u32| own[..n as usize].chars().count() as u32;
                Located { item, bytes: (a, b), utf16: (units(a), units(b)), chars: (chars(a), chars(b)) }
            })
            .collect()
    }
}

/// Reads a `DoclingDocument`: the items laid over the source text, and
/// the structure to add to it.
pub fn parse(json: &str) -> Result<(Docling, Structure), String> {
    let doc: Value = serde_json::from_str(json).map_err(|e| format!("not JSON: {e}"))?;
    if doc["schema_name"] != "DoclingDocument" {
        return Err("not a DoclingDocument (schema_name)".into());
    }
    let body = doc.get("body").filter(|b| b.is_object()).ok_or("missing or invalid body")?;
    if body.get("children").is_none() {
        return Err("body has no children array".into());
    }
    let mut b = Builder { doc: &doc, ..Builder::default() };
    for child in children(body)? {
        b.node(child, false)?;
    }
    b.close_sections(0);
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(s, e) in &b.spans {
        spans.push(Span::new(s, e));
    }
    let parent = b.parent.iter().map(|p| p.map(|i| Ix::new(i as u32))).collect();
    let structure = Structure { spans, kind: b.kind, parent, depth: b.depth };
    let name = doc["name"].as_str().unwrap_or_default().to_owned();
    Ok((Docling { name, text: b.text, items: b.items }, structure))
}

fn children(node: &Value) -> Result<Vec<&str>, String> {
    let Some(value) = node.get("children") else { return Ok(Vec::new()) };
    let list = value.as_array().ok_or("children must be an array")?;
    list.iter().map(|c| c["$ref"].as_str().ok_or_else(|| format!("a child without $ref: {c}"))).collect()
}

#[derive(Default)]
struct Builder<'a> {
    doc: &'a Value,
    text: String,
    items: Vec<Item>,
    spans: Vec<(u32, u32)>,
    kind: Vec<Block>,
    parent: Vec<Option<usize>>,
    depth: Vec<u8>,
    /// The open sections, outermost first.
    sections: Vec<usize>,
    /// The open list, while its items are laid down, and how many it has.
    list: Option<usize>,
    list_items: usize,
    /// Docling references form a tree, not a cyclic graph or a shared subtree.
    visited: BTreeSet<String>,
    nesting: usize,
}

impl<'a> Builder<'a> {
    fn resolve(&self, r: &str) -> Result<&'a Value, String> {
        let path = r.strip_prefix("#/").ok_or_else(|| format!("{r}: not a local reference"))?;
        let (table, index) = path.split_once('/').ok_or_else(|| format!("{r}: no index"))?;
        let i: usize = index.parse().map_err(|_| format!("{r}: no index"))?;
        let doc: &'a Value = self.doc;
        doc[table].get(i).ok_or_else(|| format!("{r}: no such item"))
    }

    fn end(&self) -> u32 {
        self.text.len() as u32
    }

    fn block(&mut self, start: u32, kind: Block, depth: u8, parent: Option<usize>) -> usize {
        self.spans.push((start, start));
        self.kind.push(kind);
        self.parent.push(parent);
        self.depth.push(depth);
        self.spans.len() - 1
    }

    /// Ends every open section at `depth` or deeper at the text laid so far.
    fn close_sections(&mut self, depth: u8) {
        while let Some(&top) = self.sections.last() {
            if self.depth[top] < depth {
                break;
            }
            self.spans[top].1 = self.last_end();
            self.sections.pop();
        }
    }

    /// The end of the last block, before any separator.
    fn last_end(&self) -> u32 {
        self.items.last().map_or(0, |i| i.span.end)
    }

    fn separate(&mut self, list_item: bool) {
        if !self.text.is_empty() {
            self.text.push_str(if list_item { "\n" } else { "\n\n" });
        }
    }

    fn node(&mut self, r: &str, in_list: bool) -> Result<(), String> {
        if self.nesting >= 128 {
            return Err(format!("{r}: reference nesting exceeds 128"));
        }
        if !self.visited.insert(r.to_owned()) {
            return Err(format!("{r}: repeated or cyclic reference"));
        }
        self.nesting += 1;
        let result = self.visit(r, in_list);
        self.nesting -= 1;
        result
    }

    fn visit(&mut self, r: &str, in_list: bool) -> Result<(), String> {
        let node = self.resolve(r)?;
        if r.starts_with("#/groups/") {
            return self.group(r, node, in_list);
        }
        if !r.starts_with("#/texts/") {
            return Err(format!("{r}: only text items and groups are converted"));
        }
        if !children(node)?.is_empty() {
            return Err(format!("{r}: text children are not converted"));
        }
        let label = node["label"].as_str().ok_or_else(|| format!("{r}: no label"))?;
        let text = node["text"].as_str().ok_or_else(|| format!("{r}: no text"))?;
        let section = self.sections.last().copied();
        let (kind, depth) = match label {
            "title" => (Block::Heading, 1),
            "section_header" => {
                let level = match node.get("level") {
                    None => 1,
                    Some(value) => value.as_u64().ok_or_else(|| format!("{r}: invalid heading level"))?,
                };
                let depth = u8::try_from(level).ok().and_then(|n| n.checked_add(1));
                (Block::Heading, depth.ok_or_else(|| format!("{r}: level {level}"))?)
            }
            "text" | "paragraph" | "reference" => (Block::Paragraph, 0),
            "caption" => (Block::Caption, 0),
            "footnote" => (Block::Footnote, 0),
            "code" => (Block::Code, 0),
            "list_item" if in_list => (Block::ListItem, 1),
            "list_item" => return Err(format!("{r}: a list item outside a list group")),
            other => return Err(format!("{r}: label {other} is not converted")),
        };
        if text.is_empty() {
            return Ok(());
        }
        if in_list && kind == Block::Heading {
            return Err(format!("{r}: a heading inside a list is not converted"));
        }
        self.separate(in_list && self.list_items > 0);
        let start = self.end();
        if let (true, Some(l)) = (in_list, self.list) {
            if self.list_items == 0 {
                self.spans[l].0 = start;
            }
            self.list_items += 1;
        }
        let parent = if kind == Block::Heading {
            self.close_sections(depth);
            let s = self.block(start, Block::Section, depth, self.sections.last().copied());
            self.sections.push(s);
            Some(s)
        } else if in_list {
            self.list
        } else {
            section
        };
        let b = self.block(start, kind, depth, parent);
        self.text.push_str(text);
        let end = self.end();
        self.spans[b].1 = end;
        self.items.push(Item { self_ref: r.to_owned(), label: label.to_owned(), span: Span::new(start, end) });
        Ok(())
    }

    fn group(&mut self, r: &str, node: &Value, in_list: bool) -> Result<(), String> {
        let label = node["label"].as_str().unwrap_or_default();
        if !matches!(label, "list" | "ordered_list") {
            // Chapters, sections and other wrappers add no block.
            for c in children(node)? {
                self.node(c, in_list)?;
            }
            return Ok(());
        }
        if in_list {
            return Err(format!("{r}: a list inside a list is not converted"));
        }
        // The list starts where its first item does, once one is laid.
        let list = self.block(0, Block::List, 1, self.sections.last().copied());
        (self.list, self.list_items) = (Some(list), 0);
        for c in children(node)? {
            self.node(c, true)?;
        }
        self.list = None;
        if self.list_items == 0 {
            // No item had text: no block. Nothing was pushed after it.
            self.spans.pop();
            self.kind.pop();
            self.parent.pop();
            self.depth.pop();
        } else {
            self.spans[list].1 = self.last_end();
        }
        Ok(())
    }
}
