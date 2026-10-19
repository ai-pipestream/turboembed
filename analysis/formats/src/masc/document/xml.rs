//! XML parsers for MASC stand-off annotation and header files.

use std::collections::HashMap;

use turbo_jvm::hash_map::put_order;
use turbo_maxent::xml::{self, Attribute, Handler, Names};

use super::super::error::{AnnotationFile, MascError};
use super::super::java::{equals_ignore_case_ascii, parse_int, split_on_spaces, split_space};
use super::{MascWarning, Quark};

pub(in crate::masc) fn bad(offset: usize, message: &'static str) -> turbo_maxent::Error {
    turbo_maxent::Error::Xml { offset, message }
}

pub(in crate::masc) fn attr<'x>(attributes: &'x [Attribute<'_>], name: &str) -> Option<&'x str> {
    attributes.iter().find(|a| a.name == name).map(|a| a.value.as_ref())
}

/// `MascIdentifiers.parseId`: `prefix` and one or more ASCII digits, in
/// the `int` range.
pub(in crate::masc) fn parse_id(id: Option<&str>, prefix: &str) -> Option<i32> {
    let digits = id?.strip_prefix(prefix)?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<i32>().ok()
}

/// `MascIdentifiers.parseIds`: one or more ids separated by spaces.
fn parse_ids(ids: Option<&str>, prefix: &str) -> Option<Vec<i32>> {
    let items = split_on_spaces(ids?);
    if items.is_empty() {
        return None;
    }
    items.into_iter().map(|i| parse_id(Some(i), prefix)).collect()
}

/// Whether `Span(start, end)` accepts the offsets.
pub(in crate::masc) fn valid_span(start: i32, end: i32) -> bool {
    start >= 0 && end >= 0 && start <= end
}

pub(in crate::masc) fn read_xml(
    bytes: &[u8],
    file: AnnotationFile,
    handler: &mut impl Handler,
) -> Result<(), MascError> {
    let fail = |cause| MascError::Annotation { file, cause };
    let text = xml::decode(bytes).map_err(fail)?;
    xml::parse_with(&text, Names::Qualified, handler).map_err(fail)
}

/// `MascWordParser`: every `region` element (any case) is a quark.
struct Segments(Vec<Quark>);

impl Handler for Segments {
    fn start_element(&mut self, name: &str, a: &[Attribute<'_>], at: usize) -> turbo_maxent::Result<()> {
        if !equals_ignore_case_ascii(name, "region") {
            return Ok(());
        }
        let id = parse_id(attr(a, "xml:id"), "seg-r").ok_or(bad(at, "a region id is not seg-r and digits"))?;
        let anchors = split_on_spaces(attr(a, "anchors").ok_or(bad(at, "a region has no anchors"))?);
        let [left, right] = anchors[..] else {
            return Err(bad(at, "a region's anchors are not two numbers"));
        };
        let (Some(start), Some(end)) = (parse_int(left), parse_int(right)) else {
            return Err(bad(at, "a region anchor is not an int"));
        };
        if !valid_span(start, end) {
            return Err(bad(at, "a region's anchors are not a span"));
        }
        self.0.push(Quark { id, start, end });
        Ok(())
    }

    fn end_element(&mut self, _: &str, _: usize) -> turbo_maxent::Result<()> {
        Ok(())
    }

    fn characters(&mut self, _: &str) {}
}

pub(super) fn parse_segments(bytes: &[u8]) -> Result<Vec<Quark>, MascError> {
    let mut h = Segments(Vec::new());
    read_xml(bytes, AnnotationFile::Segments, &mut h)?;
    Ok(h.0)
}

/// `MascSentenceParser`: every `region` element (any case) is a sentence
/// region, the first two fields of its anchors split at each space.
struct Sentences(Vec<(i32, i32)>);

impl Handler for Sentences {
    fn start_element(&mut self, name: &str, a: &[Attribute<'_>], at: usize) -> turbo_maxent::Result<()> {
        if !equals_ignore_case_ascii(name, "region") {
            return Ok(());
        }
        let fields = split_space(attr(a, "anchors").ok_or(bad(at, "a region has no anchors"))?);
        let (Some(start), Some(end)) =
            (fields.first().and_then(|f| parse_int(f)), fields.get(1).and_then(|f| parse_int(f)))
        else {
            return Err(bad(at, "a region's first two anchors are not ints"));
        };
        if !valid_span(start, end) {
            return Err(bad(at, "a region's anchors are not a span"));
        }
        self.0.push((start, end));
        Ok(())
    }

    fn end_element(&mut self, _: &str, _: usize) -> turbo_maxent::Result<()> {
        Ok(())
    }

    fn characters(&mut self, _: &str) {}
}

/// A region of the sentence file: its start and end, in UTF-16 units.
pub(super) type Region = (i32, i32);

/// The regions, without each one that does not end before the next
/// starts (the last is always kept), and the regions left out.
pub(super) fn parse_sentences(bytes: &[u8]) -> Result<(Vec<Region>, Vec<Region>), MascError> {
    let mut h = Sentences(Vec::new());
    read_xml(bytes, AnnotationFile::Sentences, &mut h)?;
    let anchors = h.0;
    let Some(&last) = anchors.last() else {
        return Err(MascError::IndexOutOfBounds { index: -1, len: 0 });
    };
    let (mut kept, mut dropped) = (Vec::new(), Vec::new());
    for w in anchors.windows(2) {
        if w[0].1 < w[1].0 { kept.push(w[0]) } else { dropped.push(w[0]) }
    }
    kept.push(last);
    Ok((kept, dropped))
}

/// One token of the Penn file and its quark ids.
pub(super) struct PennToken {
    pub(super) id: i32,
    pub(super) quarks: Vec<i32>,
}

/// What `MascPennTagParser` reads.
#[derive(Default)]
pub(super) struct Penn {
    /// Each token once, at its first link; a later link replaces its
    /// quarks.
    pub(super) tokens: Vec<PennToken>,
    pub(super) places: HashMap<i32, usize>,
    pub(super) tags: HashMap<i32, Option<String>>,
    pub(super) bases: HashMap<i32, Option<String>>,
    /// The ids of the `node` elements not linked yet.
    nodes: Vec<i32>,
    /// The ids of the open `a` elements.
    annotations: Vec<i32>,
}

impl Handler for Penn {
    fn start_element(&mut self, name: &str, a: &[Attribute<'_>], at: usize) -> turbo_maxent::Result<()> {
        match name {
            "node" => {
                let id = parse_id(attr(a, "xml:id"), "penn-n").ok_or(bad(at, "a node id is not penn-n and digits"))?;
                self.nodes.push(id);
            }
            "link" => {
                if self.nodes.is_empty() {
                    return Err(bad(at, "a link with no node to link"));
                }
                let quarks =
                    parse_ids(attr(a, "targets"), "seg-r").ok_or(bad(at, "a link's targets are not seg-r ids"))?;
                let id = self.nodes.pop().ok_or(bad(at, "a link with no node to link"))?;
                match self.places.get(&id) {
                    Some(&p) => self.tokens[p].quarks = quarks,
                    None => {
                        self.places.insert(id, self.tokens.len());
                        self.tokens.push(PennToken { id, quarks });
                    }
                }
            }
            "a" => {
                let id =
                    parse_id(attr(a, "ref"), "penn-n").ok_or(bad(at, "an annotation ref is not penn-n and digits"))?;
                self.annotations.push(id);
            }
            "f" => {
                let token = *self.annotations.last().ok_or(bad(at, "a feature outside an annotation"))?;
                let value = attr(a, "value").map(str::to_owned);
                match attr(a, "name").ok_or(bad(at, "a feature without a name"))? {
                    "msd" => {
                        self.tags.insert(token, value);
                    }
                    "base" => {
                        self.bases.insert(token, value);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn end_element(&mut self, name: &str, _: usize) -> turbo_maxent::Result<()> {
        if name == "a" {
            self.annotations.pop();
        }
        Ok(())
    }

    fn characters(&mut self, _: &str) {}
}

pub(super) fn parse_penn(bytes: &[u8]) -> Result<Penn, MascError> {
    let mut h = Penn::default();
    read_xml(bytes, AnnotationFile::Penn, &mut h)?;
    Ok(h)
}

/// One named entity and the ids of its tokens, in edge order.
pub(super) struct Entity {
    pub(super) id: i32,
    pub(super) tokens: Vec<i32>,
}

/// What `MascNamedEntityParser` reads.
#[derive(Default)]
pub(super) struct NamedEntities {
    /// The label of each entity that has an `a` element.
    pub(super) types: HashMap<i32, Option<String>>,
    /// Each entity once, at its first edge.
    pub(super) entities: Vec<Entity>,
    places: HashMap<i32, usize>,
    /// The iteration order of the entity map.
    pub(super) order: Vec<usize>,
    /// The type each token was last given.
    token_types: HashMap<i32, Option<String>>,
    pub(super) warnings: Vec<MascWarning>,
}

impl Handler for NamedEntities {
    fn start_element(&mut self, name: &str, a: &[Attribute<'_>], at: usize) -> turbo_maxent::Result<()> {
        match name {
            "a" => {
                let id = parse_id(attr(a, "ref"), "ne-n").ok_or(bad(at, "an annotation ref is not ne-n and digits"))?;
                if self.types.contains_key(&id) {
                    return Err(bad(at, "two labels for one named entity"));
                }
                self.types.insert(id, attr(a, "label").map(str::to_owned));
            }
            "edge" => {
                let id = parse_id(attr(a, "from"), "ne-n").ok_or(bad(at, "an edge's from is not ne-n and digits"))?;
                let token =
                    parse_id(attr(a, "to"), "penn-n").ok_or(bad(at, "an edge's to is not penn-n and digits"))?;
                match self.places.get(&id) {
                    Some(&p) => self.entities[p].tokens.push(token),
                    None => {
                        self.places.insert(id, self.entities.len());
                        self.entities.push(Entity { id, tokens: vec![token] });
                    }
                }
                let kind = self.types.get(&id).cloned().flatten();
                if let Some(before) = self.token_types.get(&token) {
                    match &kind {
                        // `type.equals` on a null type: the entity has no
                        // label, or no `a` element before this edge
                        None => return Err(bad(at, "an edge of an entity without a label to a named token")),
                        Some(k) if before.as_deref() != Some(k.as_str()) => {
                            self.warnings.push(MascWarning::TokenInSeveralTypes { token });
                        }
                        Some(_) => {}
                    }
                }
                self.token_types.insert(token, kind);
            }
            _ => {}
        }
        Ok(())
    }

    fn end_element(&mut self, _: &str, _: usize) -> turbo_maxent::Result<()> {
        Ok(())
    }

    fn characters(&mut self, _: &str) {}
}

pub(super) fn parse_named_entities(bytes: &[u8]) -> Result<NamedEntities, MascError> {
    let mut h = NamedEntities::default();
    read_xml(bytes, AnnotationFile::NamedEntities, &mut h)?;
    h.order = put_order(h.entities.iter().map(|e| e.id)).0;
    Ok(h)
}

/// A file a document header names: an `annotation` or `primaryData`
/// element's `f.id` and `loc`, either of which may be missing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct HeaderFile {
    pub(crate) id: Option<String>,
    pub(crate) loc: Option<String>,
}

/// The files of a header (`MascDocumentStream.HeaderHandler`), one per
/// `f.id` (a later element with the same `f.id` replaces the `loc`), in the
/// iteration order of its map, or `None` for a header without an
/// `annotation` or `primaryData` element.
pub(crate) fn parse_header(text: &str) -> Result<Option<Vec<HeaderFile>>, turbo_maxent::Error> {
    #[derive(Default)]
    struct Header {
        files: Option<Vec<HeaderFile>>,
        loc: Option<String>,
        id: Option<String>,
    }
    fn file_element(name: &str) -> bool {
        equals_ignore_case_ascii(name, "annotation") || equals_ignore_case_ascii(name, "primaryData")
    }
    impl Handler for Header {
        fn start_element(&mut self, name: &str, a: &[Attribute<'_>], _: usize) -> turbo_maxent::Result<()> {
            if file_element(name) {
                self.loc = attr(a, "loc").map(str::to_owned);
                self.id = attr(a, "f.id").map(str::to_owned);
                self.files.get_or_insert_with(Vec::new);
            }
            Ok(())
        }

        fn end_element(&mut self, name: &str, _: usize) -> turbo_maxent::Result<()> {
            if file_element(name) {
                let files = self.files.get_or_insert_with(Vec::new);
                match files.iter_mut().find(|f| f.id == self.id) {
                    Some(file) => file.loc = self.loc.clone(),
                    None => {
                        files.push(HeaderFile { id: self.id.clone(), loc: self.loc.clone() });
                    }
                }
            }
            Ok(())
        }

        fn characters(&mut self, _: &str) {}
    }
    let mut h = Header::default();
    xml::parse_with(text, Names::Qualified, &mut h)?;
    Ok(h.files.map(|mut files| {
        let (order, _) = put_order(files.iter().map(|f| f.id.as_deref()));
        order.into_iter().map(|i| std::mem::take(&mut files[i])).collect()
    }))
}
