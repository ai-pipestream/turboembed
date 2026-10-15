//! Ordinals written in a text view: positions such as `third`, `21st`,
//! `one hundred twenty-fifth`, and relative ones such as `last`, `next`
//! or `second to last`.
//!
//! An ordinal is a position counted from somewhere: from the start
//! (`third` is the start plus 3), from the end (`last` is the end plus 0,
//! `penultimate` the end minus 1) or from the current item (`next` is the
//! current item plus 1). The layer stores that pair, the way
//! Recognizers-Text resolves an ordinal.

use turbo_analysis::{Ix, Layer, Span, Spans, ViewId};

/// What an ordinal counts from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OrdinalBase {
    /// `third`, `21st`: the offset is the position, 1 or more.
    Start,
    /// `last`, `penultimate`: the offset is 0 or below.
    End,
    /// `next`, `previous`, `current`.
    Current,
}

impl OrdinalBase {
    /// The name Recognizers-Text uses: `start`, `end` or `current`.
    pub fn name(self) -> &'static str {
        match self {
            OrdinalBase::Start => "start",
            OrdinalBase::End => "end",
            OrdinalBase::Current => "current",
        }
    }
}

/// Every ordinal mention in a view, one item per mention, in text order
/// and not overlapping.
pub struct Ordinals {
    pub spans: Spans<Ordinals>,
    pub base: Vec<OrdinalBase>,
    pub offset: Vec<i64>,
}

/// One mention, as handed to [`Ordinals::push`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OrdinalMention {
    pub span: Span,
    pub base: OrdinalBase,
    pub offset: i64,
}

impl Ordinals {
    /// An empty layer over `view`.
    pub fn new(view: ViewId) -> Ordinals {
        Ordinals { spans: Spans::new(view), base: Vec::new(), offset: Vec::new() }
    }

    /// Appends one mention to every column.
    pub fn push(&mut self, m: OrdinalMention) -> Ix<Ordinals> {
        self.base.push(m.base);
        self.offset.push(m.offset);
        self.spans.push(m.span)
    }

    /// Mention `i`, column by column.
    pub fn get(&self, i: Ix<Ordinals>) -> OrdinalMention {
        OrdinalMention { span: self.spans.get(i), base: self.base[i.get()], offset: self.offset[i.get()] }
    }
}

impl Layer for Ordinals {
    const ID: &'static str = "turbo:ordinals";
    positional!(Ordinals);

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.base.len() != n || self.offset.len() != n {
            return Err("every column as long as the spans".into());
        }
        self.spans.check(Self::OVERLAP).map_err(|_| "ordinal mentions in text order, not overlapping")?;
        for (i, span) in self.spans.iter() {
            let i = i.get();
            if span.is_empty() {
                return Err(format!("ordinal mention {i} is empty at {}", span.start));
            }
            let offset = self.offset[i];
            let fits = match self.base[i] {
                OrdinalBase::Start => offset >= 1,
                OrdinalBase::End => offset <= 0,
                OrdinalBase::Current => true,
            };
            if !fits {
                return Err(format!("ordinal mention {i} has the offset {offset} from {}", self.base[i].name()));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(start: u32, end: u32, base: OrdinalBase, offset: i64) -> OrdinalMention {
        OrdinalMention { span: Span::new(start, end), base, offset }
    }

    #[test]
    fn mentions_round_trip() {
        let mut o = Ordinals::new(ViewId::SOURCE);
        o.push(m(0, 5, OrdinalBase::Start, 3));
        o.push(m(6, 10, OrdinalBase::End, -1));
        o.push(m(11, 15, OrdinalBase::Current, 1));
        assert_eq!(o.check(), Ok(()));
        assert_eq!(o.len(), 3);
        assert_eq!(o.get(Ix::new(1)), m(6, 10, OrdinalBase::End, -1));
        assert_eq!(Ordinals::ID, "turbo:ordinals");
        assert_eq!(OrdinalBase::Current.name(), "current");
    }

    #[test]
    fn check_refuses_offsets_outside_the_base() {
        let mut o = Ordinals::new(ViewId::SOURCE);
        o.push(m(0, 5, OrdinalBase::Start, 0));
        assert!(o.check().unwrap_err().contains("offset 0 from start"));
        let mut o = Ordinals::new(ViewId::SOURCE);
        o.push(m(0, 5, OrdinalBase::End, 1));
        assert!(o.check().unwrap_err().contains("offset 1 from end"));
    }

    #[test]
    fn check_wants_order_columns_and_text() {
        let mut o = Ordinals::new(ViewId::SOURCE);
        o.push(m(4, 6, OrdinalBase::Start, 1));
        o.push(m(0, 2, OrdinalBase::Start, 1));
        assert!(o.check().unwrap_err().contains("text order"));
        let mut o = Ordinals::new(ViewId::SOURCE);
        o.push(m(4, 6, OrdinalBase::Start, 1));
        o.offset.pop();
        assert_eq!(o.check().unwrap_err(), "every column as long as the spans");
        let mut o = Ordinals::new(ViewId::SOURCE);
        o.push(m(4, 4, OrdinalBase::Start, 1));
        assert!(o.check().unwrap_err().contains("empty"));
    }
}
