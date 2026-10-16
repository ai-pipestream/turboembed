//! Affix rules: one line of a `PFX` or `SFX` block.

use std::sync::Arc;

use crate::condition::Condition;
use crate::flag::{Flag, FlagSet};

/// Morphological fields such as `st:walk` or `po:verb`, as the file lists
/// them; shared when an `AM` alias names them.
pub type Fields = Arc<[Box<str>]>;

/// Whether a rule adds a prefix or a suffix.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AffixKind {
    Prefix,
    Suffix,
}

/// One rule of a `PFX` or `SFX` block.
#[derive(Clone, Debug)]
pub struct Affix {
    /// The block's flag, which an entry carries to take the rule.
    pub(crate) flag: Flag,
    /// The flag as the file wrote it, which an analysis reports as `fl:`
    /// when the rule has no morphological fields.
    pub(crate) flag_text: Box<str>,
    /// `Y` in the block header: the rule combines with a rule of the other
    /// kind.
    pub(crate) cross_product: bool,
    /// The stem text the rule removes, restored when it is undone.
    pub(crate) strip: Box<str>,
    /// The text the rule adds.
    pub(crate) add: Box<str>,
    pub(crate) condition: Condition,
    /// The flags of the rules that may be added on top of this one, and the
    /// marker flags (`NEEDAFFIX`, `COMPOUNDPERMITFLAG`, ...) the rule
    /// carries.
    pub(crate) continuation: FlagSet,
    pub(crate) kind: AffixKind,
    pub(crate) morphology: Option<Fields>,
}

impl Affix {
    pub fn kind(&self) -> AffixKind {
        self.kind
    }

    pub fn flag(&self) -> Flag {
        self.flag
    }

    pub fn strip(&self) -> &str {
        &self.strip
    }

    /// The text the rule adds.
    pub fn affix(&self) -> &str {
        &self.add
    }

    pub(crate) fn is_suffix(&self) -> bool {
        self.kind == AffixKind::Suffix
    }

    /// Whether the rule's continuation classes hold `flag`.
    #[inline]
    pub(crate) fn allows(&self, flag: Flag) -> bool {
        self.continuation.contains(flag)
    }

    /// Whether the rule neither adds nor removes text.
    pub(crate) fn is_identity(&self) -> bool {
        self.add.is_empty() && self.strip.is_empty()
    }

    /// The morphological fields, empty without any.
    pub(crate) fn fields(&self) -> &[Box<str>] {
        self.morphology.as_deref().unwrap_or(&[])
    }
}
