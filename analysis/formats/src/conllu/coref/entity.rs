//! The `Entity` attribute of the CoNLL-U `MISC` column, in which CorefUD
//! 1.x and the Universal Anaphora releases mark the coreference mentions
//! that start and end at a word.
//!
//! The value is a run of brackets, each naming an entity by its id:
//!
//! - `(e3-person-1-` opens a mention of entity `e3` at the word;
//! - `e3)` closes the innermost open mention of `e3` at the word;
//! - `(e3-person-1-)` is a mention of the word alone.
//!
//! What follows the id after a `-` is the mention's attributes in the
//! order a `# global.Entity = eid-etype-head-other` comment declares
//! (entity type, the head's place in the mention, and so on); CorefUD
//! writes them on the opening bracket only. A mention in several parts
//! (a discontinuous mention) writes each part as a mention of its own
//! with `[k/n]` after the id: `(e5[1/2]-place-1-` ... `e5[1/2])`.
//!
//! [`parse_entity`] reads a value as `ConlluCorefDocumentStream.brackets`
//! does, so it refuses what Java refuses and accepts what it accepts:
//!
//! - An opening bracket runs from `(` to the next `(` or `)`; it is a
//!   one-word mention when that is a `)`.
//! - Anything else is a closing bracket, which runs to the next `)` and
//!   must have one ([`EntityError::MissingClose`]). Its text may hold a
//!   `(`, which then is part of the id.
//! - The id is the text up to the first `-` or `[`, and must not be empty
//!   ([`EntityError::EmptyId`]); so `e5[1/2]` is entity `e5`. The part
//!   number and the attributes are read here as well, which Java does not
//!   do: [`Bracket::part`] when `[k/n]` follows the id with two ASCII
//!   decimals, and [`Bracket::attributes`] when a `-` follows the id (and
//!   the part).

use std::fmt;

use turbo_analysis::Span;

use crate::source::{Field, Piece};

/// Whether a bracket opens a mention, closes one, or is a whole one-word
/// mention.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BracketKind {
    Open,
    Close,
    Single,
}

/// The part `k` of `n` of a discontinuous mention, from `[k/n]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Part {
    pub index: u32,
    pub total: u32,
}

/// One bracket of an `Entity` value, every piece of it a field of the
/// source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bracket<'a> {
    pub kind: BracketKind,
    /// The whole bracket, its parentheses included.
    pub span: Span,
    /// What is between the parentheses.
    pub content: Field<'a>,
    /// The entity id: the content up to its first `-` or `[`.
    pub entity: Field<'a>,
    /// The part of a discontinuous mention, when `[k/n]` follows the id.
    pub part: Option<Part>,
    /// The text after the `-` that follows the id (and the part), when
    /// there is one: the attributes, separated by `-`.
    pub attributes: Option<Field<'a>>,
}

impl<'a> Bracket<'a> {
    /// The attributes one by one, as they are written between `-`.
    pub fn attribute_values(&self) -> impl Iterator<Item = &str> + '_ {
        self.attributes.as_ref().map(|a| a.as_str().split('-')).into_iter().flatten()
    }
}

/// An `Entity` value `ConlluCorefDocumentStream` refuses; each is its
/// `InvalidFormatException`, with the message as `Display`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntityError {
    /// An empty value, or a bracket with an empty id.
    EmptyId,
    /// A closing bracket with no `)` after it; `value` is the whole value.
    MissingClose { value: String },
}

impl fmt::Display for EntityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityError::EmptyId => f.write_str("entity id must not be empty"),
            EntityError::MissingClose { value } => write!(f, "entity close is missing ')': {value}"),
        }
    }
}

impl std::error::Error for EntityError {}

/// The brackets of an `Entity` value, in order.
pub fn parse_entity(value: Piece<'_>) -> Result<Vec<Bracket<'_>>, EntityError> {
    let text = value.text.as_bytes();
    if text.is_empty() {
        return Err(EntityError::EmptyId);
    }
    let mut brackets = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if text[i] == b'(' {
            let j = i + 1 + text[i + 1..].iter().take_while(|&&b| b != b'(' && b != b')').count();
            let content = value.slice(i + 1, j);
            let (kind, end) =
                if text.get(j) == Some(&b')') { (BracketKind::Single, j + 1) } else { (BracketKind::Open, j) };
            brackets.push(bracket(kind, value.slice(i, end).span, content)?);
            i = end;
        } else {
            let j = i + text[i..].iter().take_while(|&&b| b != b')').count();
            if j == text.len() {
                return Err(EntityError::MissingClose { value: value.text.to_owned() });
            }
            brackets.push(bracket(BracketKind::Close, value.slice(i, j + 1).span, value.slice(i, j))?);
            i = j + 1;
        }
    }
    Ok(brackets)
}

/// `canonicalId` of `content`, with the part and the attributes.
fn bracket<'a>(kind: BracketKind, span: Span, content: Piece<'a>) -> Result<Bracket<'a>, EntityError> {
    let text = content.text;
    let cut = text.find(['-', '[']).unwrap_or(text.len());
    if cut == 0 {
        return Err(EntityError::EmptyId);
    }
    let mut rest = cut;
    let mut part = None;
    if text[cut..].starts_with('[')
        && let Some(close) = text[cut..].find(']').map(|c| cut + c)
        && let Some(p) = parse_part(&text[cut + 1..close])
    {
        part = Some(p);
        rest = close + 1;
    }
    let attributes = text[rest..].starts_with('-').then(|| content.slice(rest + 1, text.len()).field());
    Ok(Bracket { kind, span, content: content.field(), entity: content.slice(0, cut).field(), part, attributes })
}

/// `k/n`, two ASCII decimals without sign.
fn parse_part(s: &str) -> Option<Part> {
    let (k, n) = s.split_once('/')?;
    let number = |d: &str| {
        if d.is_empty() || !d.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        d.parse::<u32>().ok()
    };
    Some(Part { index: number(k)?, total: number(n)? })
}

/// The value of the first attribute of a `MISC` column named exactly
/// `key`: `ConlluCorefDocumentStream.misc`, which splits the column on
/// `|` and takes the text after `key=`.
pub fn misc_attribute<'a>(misc: Piece<'a>, key: &str) -> Option<Piece<'a>> {
    let mut start = 0;
    for end in misc.text.match_indices('|').map(|(i, _)| i).chain([misc.text.len()]) {
        let attribute = &misc.text[start..end];
        if attribute.len() > key.len() && attribute.starts_with(key) && attribute.as_bytes()[key.len()] == b'=' {
            return Some(misc.slice(start + key.len() + 1, end));
        }
        start = end + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(s: &str) -> Piece<'_> {
        Piece { text: s, span: Span::new(100, 100 + s.len() as u32) }
    }

    fn brackets(s: &str) -> Vec<(BracketKind, &str, Option<Part>, Option<String>)> {
        parse_entity(piece(s))
            .unwrap()
            .into_iter()
            .map(|b| {
                let entity = &s[(b.entity.span.start - 100) as usize..(b.entity.span.end - 100) as usize];
                (b.kind, entity, b.part, b.attributes.map(|a| a.as_str().to_owned()))
            })
            .collect()
    }

    use BracketKind::{Close, Open, Single};

    #[test]
    fn corefud_brackets() {
        assert_eq!(
            brackets("(e1-person-1-(e2-place-2-)e3)"),
            [
                (Open, "e1", None, Some("person-1-".into())),
                (Single, "e2", None, Some("place-2-".into())),
                (Close, "e3", None, None),
            ]
        );
        let p = Some(Part { index: 1, total: 2 });
        assert_eq!(brackets("(e5[1/2]-place"), [(Open, "e5", p, Some("place".into()))]);
        assert_eq!(brackets("e5[1/2])e1)"), [(Close, "e5", p, None), (Close, "e1", None, None)]);
        // A malformed part is not a part; the id still ends at `[`.
        assert_eq!(brackets("(e5[x/2]-a)"), [(Single, "e5", None, None)]);
        // A close may hold a `(`, which Java keeps in the id.
        assert_eq!(brackets("e1(e2)"), [(Close, "e1(e2", None, None)]);
        assert_eq!(brackets("(3"), [(Open, "3", None, None)]);
    }

    #[test]
    fn spans_point_into_the_source() {
        let s = "(é1-x)é2)";
        let b = parse_entity(piece(s)).unwrap();
        assert_eq!(b[0].span, Span::new(100, 107));
        assert_eq!(b[0].entity.as_str(), "é1");
        assert_eq!(b[0].entity.span, Span::new(101, 104));
        assert_eq!(b[1].span, Span::new(107, 111));
        assert_eq!(b[1].content.span, Span::new(107, 110));
        assert_eq!(b[0].attribute_values().collect::<Vec<_>>(), ["x"]);
    }

    #[test]
    fn refusals() {
        assert_eq!(parse_entity(piece("")), Err(EntityError::EmptyId));
        assert_eq!(parse_entity(piece("()")), Err(EntityError::EmptyId));
        assert_eq!(parse_entity(piece("(-x")), Err(EntityError::EmptyId));
        assert_eq!(parse_entity(piece("([1/2]")), Err(EntityError::EmptyId));
        assert_eq!(parse_entity(piece(")")), Err(EntityError::EmptyId));
        assert_eq!(parse_entity(piece("(e1)e2")), Err(EntityError::MissingClose { value: "(e1)e2".into() }));
        assert_eq!(EntityError::MissingClose { value: "x".into() }.to_string(), "entity close is missing ')': x");
    }

    #[test]
    fn misc_attributes_match_whole_keys() {
        let m = piece("Entityx=1|Entity=|Entity=(e1)|SpaceAfter=No");
        // `Entity=` is longer than the key, so its empty value is taken.
        assert_eq!(misc_attribute(m, "Entity").map(|p| p.text), Some(""));
        assert_eq!(misc_attribute(m, "SpaceAfter").map(|p| p.text), Some("No"));
        assert_eq!(misc_attribute(piece("_"), "Entity"), None);
        assert_eq!(misc_attribute(piece("Entity"), "Entity"), None);
        let v = misc_attribute(piece("a|Entity=(e1)"), "Entity").unwrap();
        assert_eq!(v.span, Span::new(109, 113));
    }
}
