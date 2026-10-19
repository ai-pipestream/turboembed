use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroU32;

/// A position in layer `L`: item `i` of that layer and of no other. An
/// `Ix<Tokens>` cannot index sentences, and a dependency head cannot point
/// at a sentence by mistake; the compiler refuses it.
///
/// ```compile_fail
/// use turbo_analysis::{Ix, Spans};
/// struct Sentences;
/// struct Tokens;
/// fn first_sentence(s: &Spans<Sentences>, t: Ix<Tokens>) {
///     let _ = s.get(t); // expected Ix<Sentences>, found Ix<Tokens>
/// }
/// ```
pub struct Ix<L>(NonZeroU32, PhantomData<fn() -> L>);

// Stored as i + 1, so an absent index costs nothing: a dependency head that
// may be the root, or a parent that may be none, is four bytes.
const _: () = assert!(std::mem::size_of::<Option<Ix<()>>>() == 4);

impl<L> Ix<L> {
    pub const fn new(i: u32) -> Ix<L> {
        match NonZeroU32::new(i.wrapping_add(1)) {
            Some(n) => Ix(n, PhantomData),
            None => panic!("an index below u32::MAX"),
        }
    }

    pub const fn get(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl<L> Clone for Ix<L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<L> Copy for Ix<L> {}
impl<L> PartialEq for Ix<L> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<L> Eq for Ix<L> {}
impl<L> PartialOrd for Ix<L> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<L> Ord for Ix<L> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
impl<L> std::hash::Hash for Ix<L> {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.0.hash(h)
    }
}
impl<L> fmt::Debug for Ix<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = std::any::type_name::<L>();
        write!(f, "{}#{}", name.rsplit("::").next().unwrap_or(name), self.get())
    }
}

/// Items `[start, end)` of layer `L`, such as the tokens of one sentence.
pub struct IxRange<L> {
    pub start: Ix<L>,
    pub end: Ix<L>,
}

impl<L> IxRange<L> {
    pub const fn new(start: u32, end: u32) -> IxRange<L> {
        IxRange { start: Ix::new(start), end: Ix::new(end) }
    }

    pub const fn len(&self) -> usize {
        self.end.get() - self.start.get()
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Ix<L>> + use<L> {
        (self.start.get() as u32..self.end.get() as u32).map(Ix::new)
    }
}

impl<L> Clone for IxRange<L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<L> Copy for IxRange<L> {}
impl<L> PartialEq for IxRange<L> {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}
impl<L> Eq for IxRange<L> {}
impl<L> fmt::Debug for IxRange<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}..{}", self.start, self.end.get())
    }
}
