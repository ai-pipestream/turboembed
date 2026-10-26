//! What a grammar is made of: categories, a lexicon, rules, and the
//! [`Grammar`] trait a language module implements.

use crate::value::Value;

use super::interned::Interned;
use super::lexer::{Kind, Lexed, Lexeme, PlainWords, WordForms};
use super::numeral::NumeralStyle;

/// A category of chart nodes: a class of lexicon words (`UNIT`, `SCALE`)
/// or what a rule builds (`INT`, `ORD`). A grammar numbers its own from 0.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Dim(pub u16);

impl Dim {
    /// Marks a node that a better one replaced; no rule refers to it.
    pub const DEAD: Dim = Dim(u16::MAX);
}

/// A test on a child's value.
pub type Test = fn(&Value) -> bool;

/// Makes a rule's value from its children's values, in item order (word
/// items have none), or refuses the match with `None`.
pub type Build = fn(&[Value], &Ctx<'_>) -> Option<Value>;

/// One position of a rule.
#[derive(Copy, Clone, Debug)]
pub enum Item {
    /// A node of `dim` whose value passes `test`. With `glued`, its first
    /// lexeme must follow the previous one without whitespace.
    Node { dim: Dim, test: Option<Test>, glued: bool },
    /// One lexeme whose lowercase form is `text`, or, with `exact_case`,
    /// whose text as written is `text`. An optional word may be absent.
    Word { text: &'static str, glued: bool, optional: bool, exact_case: bool },
}

pub const fn node(dim: Dim) -> Item {
    Item::Node { dim, test: None, glued: false }
}

pub const fn node_if(dim: Dim, test: Test) -> Item {
    Item::Node { dim, test: Some(test), glued: false }
}

pub const fn glued(dim: Dim) -> Item {
    Item::Node { dim, test: None, glued: true }
}

pub const fn glued_if(dim: Dim, test: Test) -> Item {
    Item::Node { dim, test: Some(test), glued: true }
}

pub const fn word(text: &'static str) -> Item {
    Item::Word { text, glued: false, optional: false, exact_case: false }
}

pub const fn glued_word(text: &'static str) -> Item {
    Item::Word { text, glued: true, optional: false, exact_case: false }
}

pub const fn opt_word(text: &'static str) -> Item {
    Item::Word { text, glued: false, optional: true, exact_case: false }
}

/// A word matched as written, case and all: `M` but not `m`.
pub const fn exact(text: &'static str) -> Item {
    Item::Word { text, glued: false, optional: false, exact_case: true }
}

/// A rule: when its items match in a row, `build` makes a node of `dim`.
#[derive(Copy, Clone, Debug)]
pub struct Rule {
    pub name: &'static str,
    pub dim: Dim,
    pub items: &'static [Item],
    /// Added to the node's score; a higher score wins a tie between two
    /// readings of the same text.
    pub weight: i16,
    pub build: Build,
}

/// A word of a grammar's lexicon and the node it seeds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub word: &'static str,
    pub dim: Dim,
    pub value: i64,
    pub flags: u8,
}

/// The words a grammar knows, sorted for binary search. One word may have
/// several entries (`second` is an ordinal and a denominator).
#[derive(Clone, Debug)]
pub struct Lexicon {
    entries: Vec<Entry>,
    /// Where the words starting with each byte begin; the last element is
    /// the number of entries.
    first: Vec<u32>,
}

impl Lexicon {
    /// Sorts `entries` by word, keeping the order of a word's entries.
    pub fn new(entries: &[Entry]) -> Lexicon {
        let mut entries = entries.to_vec();
        entries.sort_by(|a, b| a.word.cmp(b.word));
        let first = (0..=256u32)
            .map(|k| entries.partition_point(|e| e.word.as_bytes().first().is_some_and(|&b| u32::from(b) < k)) as u32)
            .collect();
        Lexicon { entries, first }
    }

    /// The entries of `word`, which is in lowercase.
    pub fn lookup(&self, word: &str) -> &[Entry] {
        let Some(&k) = word.as_bytes().first() else { return &[] };
        let bucket = &self.entries[self.first[usize::from(k)] as usize..self.first[usize::from(k) + 1] as usize];
        let lo = bucket.partition_point(|e| e.word < word);
        let hi = lo + bucket[lo..].partition_point(|e| e.word == word);
        &bucket[lo..hi]
    }

    /// Whether some word starts with `prefix`, which is in lowercase: a
    /// compound splitter stops extending a piece when none does.
    pub fn has_prefix(&self, prefix: &str) -> bool {
        let Some(&k) = prefix.as_bytes().first() else { return !self.entries.is_empty() };
        let bucket = &self.entries[self.first[usize::from(k)] as usize..self.first[usize::from(k) + 1] as usize];
        let lo = bucket.partition_point(|e| e.word < prefix);
        bucket.get(lo).is_some_and(|e| e.word.starts_with(prefix))
    }

    pub(super) fn categories_below(&self, dims: u16) -> bool {
        self.entries.iter().all(|entry| entry.dim.0 < dims)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The layer a node can be written to.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Output {
    Number,
    Ordinal,
    Percent,
    /// Amounts with a unit, money included, and units named alone.
    Quantity,
    Time,
}

/// Where a match lies, for a rule's `build` or a grammar's filter that
/// needs to look at the text around it.
#[derive(Copy, Clone, Debug)]
pub struct Ctx<'a> {
    pub text: &'a str,
    pub lexed: &'a Lexed,
    /// The first lexeme of the match.
    pub start: usize,
    /// One past its last lexeme.
    pub end: usize,
    /// The parse's time values.
    pub times: &'a Interned,
}

impl<'a> Ctx<'a> {
    pub fn lexeme(&self, i: usize) -> &'a Lexeme {
        &self.lexed.lexemes[i]
    }

    /// The lowercase form of lexeme `i`.
    pub fn word(&self, i: usize) -> &'a str {
        self.lexed.word(i)
    }

    /// The lexeme before the match.
    pub fn before(&self) -> Option<usize> {
        self.start.checked_sub(1)
    }

    /// The lexeme after the match.
    pub fn after(&self) -> Option<usize> {
        (self.end < self.lexed.len()).then_some(self.end)
    }

    /// Whether lexeme `i` is the single punctuation character `c`.
    pub fn is_punct(&self, i: usize, c: char) -> bool {
        let l = self.lexeme(i);
        l.kind == Kind::Punct && self.text[l.start as usize..].starts_with(c) && l.end - l.start == c.len_utf8() as u32
    }

    /// The text of lexeme `i` as written.
    pub fn raw(&self, i: usize) -> &'a str {
        self.lexed.raw(self.text, i)
    }

    /// A node value for the time value `v`.
    pub fn time(&self, v: crate::time::TimeValue) -> Option<Value> {
        Some(self.times.intern(v))
    }

    /// The time value of a child.
    pub fn time_of(&self, v: &Value) -> Option<crate::time::TimeValue> {
        self.times.get(v)
    }

    /// The character of punctuation lexeme `i`.
    pub fn punct(&self, i: usize) -> Option<char> {
        let l = self.lexeme(i);
        (l.kind == Kind::Punct).then(|| self.text[l.start as usize..].chars().next()).flatten()
    }
}

/// A language: its lexicon, its rules, how it writes numerals and which
/// categories become output.
pub trait Grammar: Send + Sync + 'static {
    /// A short name for provenance and messages: `en`.
    fn name(&self) -> &'static str;

    fn lexicon(&self) -> &Lexicon;

    fn rules(&self) -> &'static [Rule];

    /// How many categories the grammar has: every `Dim` it uses is below.
    fn dims(&self) -> u16;

    /// The category every numeral lexeme seeds a node of, with a
    /// [`crate::Num`] value carrying [`crate::flag::DIGITS`].
    fn numeral(&self) -> Dim;

    fn style(&self) -> NumeralStyle;

    /// How the lexer forms the grammar's words. The default leaves them
    /// whole and keeps their accents.
    fn words(&self) -> &dyn WordForms {
        &PlainWords
    }

    /// How the grammar's numbers are written, for the numbers layer.
    fn notation(&self) -> turbo_analysis_layers::NumberNotation {
        turbo_analysis_layers::NumberNotation::LatinUs
    }

    /// Which output a node of `dim` goes to, if any.
    fn output(&self, dim: Dim) -> Option<Output>;

    /// A latent clock category that becomes output only with a recognized
    /// zone suffix. Normal selection still excludes this category.
    fn time_zone_context(&self, _dim: Dim) -> bool {
        false
    }

    /// Whether a candidate for `output` is kept. The default keeps all.
    fn keep(&self, _output: Output, _node: &super::chart::Node, _ctx: &Ctx<'_>) -> bool {
        true
    }

    /// Whether a chosen ordinal hides the numbers that lie inside it. The
    /// default hides them.
    fn shadows(&self, _ordinal: &super::chart::Node) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Dim = Dim(0);
    const B: Dim = Dim(1);

    #[test]
    fn lookup_finds_every_entry_of_a_word() {
        let lex = Lexicon::new(&[
            Entry { word: "two", dim: A, value: 2, flags: 0 },
            Entry { word: "second", dim: A, value: 2, flags: 0 },
            Entry { word: "second", dim: B, value: 2, flags: 1 },
            Entry { word: "a", dim: B, value: 1, flags: 0 },
        ]);
        assert_eq!(lex.len(), 4);
        let second: Vec<Dim> = lex.lookup("second").iter().map(|e| e.dim).collect();
        assert_eq!(second, [A, B]);
        assert_eq!(lex.lookup("a").len(), 1);
        assert!(lex.lookup("seconds").is_empty());
        assert!(lex.lookup("").is_empty());
        assert!(lex.lookup("zzz").is_empty());
        assert!(lex.has_prefix("sec") && lex.has_prefix("second") && lex.has_prefix("t"));
        assert!(!lex.has_prefix("seconds") && !lex.has_prefix("x") && !lex.has_prefix("tx"));
    }

    #[test]
    fn context_reads_the_neighbours() {
        let style =
            NumeralStyle { mixed_separators: true, space_groups: false, indian_groups: false, decimal_comma: false };
        let mut lexed = Lexed::default();
        let text = "$2000 over 3";
        lexed.lex(text, &style);
        let times = Interned::default();
        let ctx = Ctx { text, lexed: &lexed, start: 1, end: 4, times: &times };
        assert_eq!(ctx.before(), Some(0));
        assert_eq!(ctx.after(), None);
        assert!(ctx.is_punct(0, '$'));
        assert_eq!(ctx.punct(0), Some('$'));
        assert_eq!(ctx.punct(1), None);
        assert_eq!(ctx.word(2), "over");
    }
}
