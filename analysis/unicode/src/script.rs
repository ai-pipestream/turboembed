//! The Script property (Scripts.txt).

pub use crate::tables::script::Script;
use crate::tables::script::{ALL, BLOCKS, INDEX, NAMES};
use crate::trie::lookup;

impl Script {
    /// The value's long name in Scripts.txt, for example `"Old_Italic"`.
    pub fn name(self) -> &'static str {
        NAMES[self as usize]
    }
}

/// The script of `c`; unassigned code points are [`Script::Unknown`].
#[inline]
pub fn script(c: char) -> Script {
    ALL[lookup(&INDEX, &BLOCKS, c) as usize]
}
