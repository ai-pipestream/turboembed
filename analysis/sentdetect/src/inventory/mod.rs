//! Classified abbreviation inventories: which full stops end no sentence,
//! judged from the abbreviation before the stop and the word after it.
//!
//! An [`AbbreviationInventory`] holds entries of four classes
//! ([`AbbreviationClass`]), each with a [`ContextRule`]:
//!
//! - [`ContextRule::Prepositive`]: a title or connective (`Dr.`, `e.g.`,
//!   `v.`) ends no sentence when a word, an initial or a number follows.
//! - [`ContextRule::Numeric`]: a numbered reference (`fig.`, `vol.`,
//!   `No.`, `pp.`, `Sec.`) ends no sentence when a number or a Roman
//!   numeral follows.
//! - [`ContextRule::Trailing`]: an abbreviation that can close a sentence
//!   (`etc.`, `Inc.`, `et al.`) ends none when a lower case word or a
//!   number follows, and has no opinion before a capitalized word.
//!
//! Any entry followed directly by `,`, `;` or `:` ends no sentence, and a
//! full stop inside an entry (`e.` of `e.g.`, `op.` of `op. cit.`) never
//! ends one. A word the inventory knows is never an abbreviation (`no`,
//! `it`) allows a sentence end that a model's dictionary or the CLDR
//! suppressions would refuse. Everything else has no verdict, and the
//! detector decides as it would without an inventory.
//!
//! The English entries are static, sorted tables (see `english.rs`);
//! user additions and exclusions are kept sorted beside them. The active
//! entries are indexed when an inventory is built or changed (see
//! `index.rs`), so judging a full stop costs a few binary searches, and
//! most words are turned away by a bit filter before any search.
//! [`AbbreviationInventory::EMPTY`] judges nothing, and a detector with
//! it splits exactly as OpenNLP does.

mod english;
mod index;
mod judge;
#[cfg(test)]
mod tests;

use std::fmt;
use std::ops::BitOr;

use index::Index;

/// The four kinds of abbreviation the bundled inventories hold.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbbreviationClass {
    /// `etc.`, `vs.`, `e.g.`, `i.e.`, `fig.`, months, `a.m.`.
    General = 0,
    /// `Mr.`, `Mrs.`, `Dr.`, `Prof.`, `Sen.`, `Rep.`, `Gen.`.
    Titles = 1,
    /// `et al.`, `op. cit.`, `ib.`, `cf.`, `Ph.D.`, `vol.`, `no.`, `pp.`.
    Academic = 2,
    /// `Inc.`, `Corp.`, `Ltd.`, `Co.`, `v.`, `Sec.`, `Art.`, `No.`,
    /// `U.S.C.`, `Fed. Reg.`.
    LegalTechnical = 3,
}

impl AbbreviationClass {
    pub const ALL: [AbbreviationClass; 4] = [
        AbbreviationClass::General,
        AbbreviationClass::Titles,
        AbbreviationClass::Academic,
        AbbreviationClass::LegalTechnical,
    ];
}

/// A set of [`AbbreviationClass`]es, composed with `|`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AbbreviationClasses(u8);

impl AbbreviationClasses {
    pub const NONE: AbbreviationClasses = AbbreviationClasses(0);
    pub const ALL: AbbreviationClasses = AbbreviationClasses(0b1111);

    pub const fn contains(self, class: AbbreviationClass) -> bool {
        self.0 & (1 << class as u8) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn bits(self) -> u8 {
        self.0
    }
}

impl From<AbbreviationClass> for AbbreviationClasses {
    fn from(class: AbbreviationClass) -> AbbreviationClasses {
        AbbreviationClasses(1 << class as u8)
    }
}

impl<T: Into<AbbreviationClasses>> BitOr<T> for AbbreviationClasses {
    type Output = AbbreviationClasses;
    fn bitor(self, rhs: T) -> AbbreviationClasses {
        AbbreviationClasses(self.0 | rhs.into().0)
    }
}

impl<T: Into<AbbreviationClasses>> BitOr<T> for AbbreviationClass {
    type Output = AbbreviationClasses;
    fn bitor(self, rhs: T) -> AbbreviationClasses {
        AbbreviationClasses::from(self) | rhs
    }
}

/// Named class sets.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbbreviationPreset {
    /// [`AbbreviationClass::General`].
    Minimal,
    /// General and titles: news and web text.
    Standard,
    /// General, titles and academic: papers and books.
    Scholarly,
    /// All four classes.
    Full,
}

impl AbbreviationPreset {
    pub fn classes(self) -> AbbreviationClasses {
        use AbbreviationClass::*;
        match self {
            AbbreviationPreset::Minimal => General.into(),
            AbbreviationPreset::Standard => General | Titles,
            AbbreviationPreset::Scholarly => General | Titles | Academic,
            AbbreviationPreset::Full => AbbreviationClasses::ALL,
        }
    }
}

impl From<AbbreviationPreset> for AbbreviationClasses {
    fn from(preset: AbbreviationPreset) -> AbbreviationClasses {
        preset.classes()
    }
}

/// When an abbreviation's full stop ends no sentence; see the module
/// documentation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ContextRule {
    Prepositive = 1,
    Numeric = 2,
    Trailing = 3,
}

/// One entry: the abbreviation as written, with its final full stop and
/// a single space between the tokens of a multi-word entry.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Abbreviation {
    pub text: &'static str,
    pub rule: ContextRule,
}

/// The bundled data of one language.
#[derive(Debug, PartialEq, Eq)]
pub struct Bundle {
    pub code: &'static str,
    /// Per [`AbbreviationClass`], in its order.
    pub classes: [&'static [Abbreviation]; 4],
    /// Words that are never abbreviations, in lower case, sorted.
    pub penultimates: &'static [&'static str],
}

/// What an inventory says about one full stop.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The stop ends no sentence.
    Suppress,
    /// The word before it is no abbreviation: a split is allowed even
    /// where a dictionary would refuse it.
    Allow,
    /// No opinion.
    Neutral,
}

/// A user addition the inventory refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbbreviationError(pub String);

impl fmt::Display for AbbreviationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "abbreviation {:?}: must end with a full stop, hold no whitespace but single spaces, and be at most 64 bytes",
            self.0
        )
    }
}

impl std::error::Error for AbbreviationError {}

/// The abbreviations a sentence detector consults; see the module
/// documentation.
#[derive(Clone, Debug)]
pub struct AbbreviationInventory {
    bundle: Option<&'static Bundle>,
    classes: AbbreviationClasses,
    /// Sorted by text, unique.
    additions: Vec<(Box<str>, ContextRule)>,
    /// Sorted, unique.
    exclusions: Vec<Box<str>>,
    /// The active entries, derived from the rest.
    index: Index,
}

/// Equal when the language, classes, additions and exclusions are: the
/// index follows from them.
impl PartialEq for AbbreviationInventory {
    fn eq(&self, other: &AbbreviationInventory) -> bool {
        self.bundle == other.bundle
            && self.classes == other.classes
            && self.additions == other.additions
            && self.exclusions == other.exclusions
    }
}

impl Eq for AbbreviationInventory {}

/// The bundled inventories by language.
pub struct AbbreviationInventories;

impl AbbreviationInventories {
    /// The bundled inventory of a BCP 47 tag with every class, or
    /// [`AbbreviationInventory::EMPTY`] for a language without one. Only
    /// English (`en`) has one.
    pub fn for_language(tag: &str) -> AbbreviationInventory {
        let primary = tag.split(['-', '_']).next().unwrap_or("");
        if primary.eq_ignore_ascii_case("en") { AbbreviationInventory::english() } else { AbbreviationInventory::EMPTY }
    }
}

impl Default for AbbreviationInventory {
    fn default() -> AbbreviationInventory {
        AbbreviationInventory::EMPTY
    }
}

impl AbbreviationInventory {
    /// No entries: a detector with it behaves as without an inventory.
    pub const EMPTY: AbbreviationInventory = AbbreviationInventory {
        bundle: None,
        classes: AbbreviationClasses::NONE,
        additions: Vec::new(),
        exclusions: Vec::new(),
        index: Index::EMPTY,
    };

    /// The bundled English inventory with all four classes.
    pub fn english() -> AbbreviationInventory {
        AbbreviationInventory::with_classes(AbbreviationClasses::ALL)
    }

    /// The bundled English inventory with the given classes, a class, a
    /// set or an [`AbbreviationPreset`].
    pub fn with_classes(classes: impl Into<AbbreviationClasses>) -> AbbreviationInventory {
        let mut inventory = AbbreviationInventory {
            bundle: Some(&english::ENGLISH),
            classes: classes.into(),
            ..AbbreviationInventory::EMPTY
        };
        inventory.index = Index::build(&inventory);
        inventory
    }

    /// The bundled English inventory of a preset.
    pub fn preset(preset: AbbreviationPreset) -> AbbreviationInventory {
        AbbreviationInventory::with_classes(preset)
    }

    /// The same inventory with an entry of the user's, which takes
    /// precedence over a bundled entry of the same text.
    pub fn with_addition(mut self, text: &str, rule: ContextRule) -> Result<AbbreviationInventory, AbbreviationError> {
        let valid = text.ends_with('.')
            && text.len() <= judge::MAX_TOKEN
            && text.split(' ').all(|t| !t.is_empty() && !t.chars().any(char::is_whitespace));
        if !valid {
            return Err(AbbreviationError(text.into()));
        }
        match self.additions.binary_search_by(|(t, _)| (**t).cmp(text)) {
            Ok(i) => self.additions[i].1 = rule,
            Err(i) => self.additions.insert(i, (text.into(), rule)),
        }
        if text.contains(' ') {
            let mut index = std::mem::replace(&mut self.index, Index::EMPTY);
            index.build_multis(&self);
            self.index = index;
        } else if !self.excluded(text) {
            self.index.add_single(text, rule);
        }
        Ok(self)
    }

    /// The same inventory without an entry (matched as written, in any
    /// class) or a never-abbreviation word (in lower case, without a full
    /// stop).
    pub fn with_exclusion(mut self, text: &str) -> AbbreviationInventory {
        if let Err(i) = self.exclusions.binary_search_by(|t| (**t).cmp(text)) {
            self.exclusions.insert(i, text.into());
            if text.contains(' ') {
                let mut index = std::mem::replace(&mut self.index, Index::EMPTY);
                index.build_multis(&self);
                self.index = index;
            } else {
                self.index.remove_single(text);
            }
        }
        self
    }

    /// The active multi-token entries with a part `token`, as written:
    /// each entry and the position of the part in it, in the order the
    /// inventory tries them. A token that is part of none is found so by
    /// one binary search, whatever the size of the inventory.
    pub fn multi_token_entries<'s>(&'s self, token: &str) -> impl Iterator<Item = (&'s str, usize)> + 's {
        self.index.entries(token, None).map(|e| (e.text, e.k))
    }

    /// The number of active multi-token entries.
    pub fn multi_token_len(&self) -> usize {
        self.index.multi_len()
    }

    pub fn classes(&self) -> AbbreviationClasses {
        self.classes
    }

    /// The language code of the bundled data, when any.
    pub fn language(&self) -> Option<&'static str> {
        self.bundle.map(|b| b.code)
    }

    /// Whether it judges nothing.
    pub fn is_empty(&self) -> bool {
        (self.bundle.is_none() || self.classes.is_empty()) && self.additions.is_empty()
    }

    /// Whether it is [`EMPTY`](Self::EMPTY) in every respect, so that a
    /// detector's configuration hash stays what it was without one.
    pub(crate) fn is_unset(&self) -> bool {
        self.is_empty() && self.exclusions.is_empty()
    }

    /// The bundled entries of the active classes, with their class.
    pub fn bundled(&self) -> impl Iterator<Item = (AbbreviationClass, &'static Abbreviation)> + '_ {
        AbbreviationClass::ALL
            .into_iter()
            .filter(|c| self.classes.contains(*c))
            .flat_map(|c| self.bundle.map_or(&[][..], |b| b.classes[c as usize]).iter().map(move |a| (c, a)))
    }

    /// FNV-1a over everything that changes a verdict: the language, the
    /// classes, every active entry and its rule, the never-abbreviation
    /// words, the additions and the exclusions.
    pub fn config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        };
        eat(b"turbo-sentdetect:abbreviations:1\0");
        eat(self.language().unwrap_or("").as_bytes());
        eat(&[0, self.classes.bits()]);
        for (class, a) in self.bundled() {
            eat(&[class as u8, a.rule as u8]);
            eat(a.text.as_bytes());
            eat(&[0]);
        }
        eat(&[0xFE]);
        if !self.classes.is_empty() {
            for w in self.bundle.map_or(&[][..], |b| b.penultimates) {
                eat(w.as_bytes());
                eat(&[0]);
            }
        }
        eat(&[0xFD]);
        for (text, rule) in &self.additions {
            eat(&[*rule as u8]);
            eat(text.as_bytes());
            eat(&[0]);
        }
        eat(&[0xFC]);
        for text in &self.exclusions {
            eat(text.as_bytes());
            eat(&[0]);
        }
        h
    }
}
