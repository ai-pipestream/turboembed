//! Retrieval units at three levels over one text: the chunks, the
//! sections they sit in and the whole document, as one tree.

use turbo_analysis::{Ix, Key, KeyId, Layer, Spans, ViewId};

use crate::{Chunks, Structure, Vectors};

/// What one unit of `Levels` is, and the row of the layer it comes from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Unit {
    /// A chunk the chunker cut: a level 0 row of `Chunks`.
    Chunk(Ix<Chunks>),
    /// A block of kind `Block::Section` in `Structure`.
    Section(Ix<Structure>),
    /// The whole view.
    Document,
}

/// The level of a unit, finest first.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Chunk,
    Section,
    Document,
}

impl Unit {
    pub fn level(self) -> Level {
        match self {
            Unit::Chunk(_) => Level::Chunk,
            Unit::Section(_) => Level::Section,
            Unit::Document => Level::Document,
        }
    }
}

/// The chunk, section and document levels of a text as one tree:
/// `turbo:levels`. Row 0 is the document, over the whole view; the
/// sections and chunks follow in text order, a parent before its
/// children. A chunk's parent is a section or the document, a section's a
/// section around it or the document.
///
/// It is a layer of its own rather than more rows of `Chunks`: `Chunks`
/// is the chunker's output, one writer, with a cost, a reason and a
/// token count per row that a section or the document does not have, and
/// each unit here names the row of `Chunks` or `Structure` it comes from,
/// so nothing of those layers is copied but the spans.
pub struct Levels {
    pub spans: Spans<Levels>,
    pub unit: Vec<Unit>,
    pub parent: Vec<Option<Ix<Levels>>>,
}

impl Levels {
    pub fn level(&self, i: Ix<Levels>) -> Level {
        self.unit[i.get()].level()
    }

    /// The units of one level, in text order.
    pub fn of(&self, level: Level) -> impl Iterator<Item = Ix<Levels>> + '_ {
        (0..self.unit.len() as u32).map(Ix::new).filter(move |&i| self.level(i) == level)
    }

    pub fn view(&self) -> ViewId {
        self.spans.view()
    }
}

impl Layer for Levels {
    const ID: &'static str = "turbo:levels";
    fn len(&self) -> usize {
        self.spans.len()
    }
    fn text_spans(&self) -> Option<(ViewId, &[u32], &[u32])> {
        let (start, end) = self.spans.columns();
        Some((self.spans.view(), start, end))
    }
    /// Only the layers some unit points into are indexed: a text with no
    /// sections has no `Structure` to point into.
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let mut chunks = 0;
        let mut sections = None;
        for u in &self.unit {
            match *u {
                Unit::Chunk(c) => chunks = chunks.max(c.get() + 1),
                Unit::Section(s) => sections = Some(sections.unwrap_or(0).max(s.get() + 1)),
                Unit::Document => {}
            }
        }
        let mut out = vec![(Key::<Chunks>::DEFAULT.id(), chunks)];
        out.extend(sections.map(|n| (Key::<Structure>::DEFAULT.id(), n)));
        out
    }
    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.unit.len() != n || self.parent.len() != n {
            return Err("a unit and a parent for every span".into());
        }
        if n == 0 {
            return Err("the document unit at row 0".into());
        }
        if self.unit[0] != Unit::Document || self.parent[0].is_some() {
            return Err("the document unit at row 0, with no parent".into());
        }
        let (start, _) = self.spans.columns();
        for i in 1..n {
            if self.unit[i] == Unit::Document {
                return Err(format!("unit {i} is a second document"));
            }
            if start[i] < start[i - 1] {
                return Err(format!("unit {i} starts before the unit before it"));
            }
            let Some(p) = self.parent[i] else { return Err(format!("unit {i} has no parent")) };
            if p.get() >= i {
                return Err(format!("unit {i} has a parent at or after it"));
            }
            if self.unit[p.get()].level() == Level::Chunk {
                return Err(format!("unit {i} is inside a chunk"));
            }
            if !self.spans.get(p).contains(self.spans.get(Ix::new(i as u32))) {
                return Err(format!("unit {i} is outside its parent"));
            }
        }
        Ok(())
    }
}

/// Which vector of a unit: of its own text alone, or pooled with the text
/// around it. The two are instances of `Vectors<Levels>` under the names
/// `key` gives, so both can be on one document.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum VectorKind {
    /// The model reads the unit's text and nothing else.
    Local,
    /// The model reads the unit's parent (its section, or the document),
    /// and the unit's pieces are pooled from that.
    Contextual,
}

impl VectorKind {
    pub const ALL: [VectorKind; 2] = [VectorKind::Local, VectorKind::Contextual];

    /// The instance name: `local` or `contextual`.
    pub const fn name(self) -> &'static str {
        match self {
            VectorKind::Local => "local",
            VectorKind::Contextual => "contextual",
        }
    }

    pub const fn key(self) -> Key<Vectors<Levels>> {
        Key::named(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::Span;

    fn levels(rows: &[(u32, u32, Unit, Option<u32>)]) -> Levels {
        let mut spans = Spans::new(ViewId::SOURCE);
        for &(a, b, ..) in rows {
            spans.push(Span::new(a, b));
        }
        Levels {
            spans,
            unit: rows.iter().map(|r| r.2).collect(),
            parent: rows.iter().map(|r| r.3.map(Ix::new)).collect(),
        }
    }

    const D: Unit = Unit::Document;

    fn s(i: u32) -> Unit {
        Unit::Section(Ix::new(i))
    }

    fn c(i: u32) -> Unit {
        Unit::Chunk(Ix::new(i))
    }

    #[test]
    fn levels_are_a_tree_under_the_document() {
        let ok = levels(&[(0, 30, D, None), (0, 5, c(0), Some(0)), (6, 30, s(0), Some(0)), (6, 30, c(1), Some(2))]);
        assert_eq!(ok.check(), Ok(()));
        assert_eq!(ok.of(Level::Chunk).map(|i| i.get()).collect::<Vec<_>>(), [1, 3]);
        assert_eq!(ok.level(Ix::new(2)), Level::Section);
        let keys: Vec<usize> = ok.indexes().iter().map(|k| k.1).collect();
        assert_eq!(keys, [2, 1]);
        // A document alone, with no sections, indexes no structure.
        let alone = levels(&[(0, 9, D, None)]);
        assert_eq!(alone.check(), Ok(()));
        assert_eq!(alone.indexes().len(), 1);
    }

    #[test]
    fn the_check_refuses_what_is_not_a_tree() {
        let err = |rows: &[(u32, u32, Unit, Option<u32>)]| levels(rows).check().unwrap_err();
        assert!(levels(&[]).check().unwrap_err().contains("row 0"));
        assert!(err(&[(0, 9, c(0), None)]).contains("row 0"));
        assert!(err(&[(0, 9, D, None), (0, 9, D, Some(0))]).contains("second document"));
        assert!(err(&[(0, 9, D, None), (0, 4, c(0), None)]).contains("no parent"));
        assert!(err(&[(0, 9, D, None), (0, 4, c(0), Some(1))]).contains("at or after"));
        assert!(err(&[(0, 9, D, None), (0, 4, c(0), Some(0)), (1, 3, c(1), Some(1))]).contains("inside a chunk"));
        assert!(err(&[(0, 9, D, None), (2, 8, s(0), Some(0)), (1, 3, c(0), Some(1))]).contains("starts before"));
        assert!(err(&[(0, 9, D, None), (2, 5, s(0), Some(0)), (3, 7, c(0), Some(1))]).contains("outside its parent"));
        assert!(err(&[(0, 9, D, None), (0, 10, c(0), Some(0))]).contains("outside its parent"));
    }

    #[test]
    fn the_vector_kinds_are_named_instances() {
        assert_eq!(VectorKind::Local.key().name, "local");
        assert_eq!(VectorKind::Contextual.key().name, "contextual");
        assert_eq!(Vectors::<Levels>::NAME.to_string(), "opennlp:embeddings<turbo:levels>");
    }
}
