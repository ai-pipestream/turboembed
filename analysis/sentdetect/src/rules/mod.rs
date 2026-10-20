//! The rule sentence splitter: UAX #29 sentence boundaries, tailored per
//! language in the manner of Wikimedia's sentencex.
//!
//! It needs no model, so it covers the languages OpenNLP has no sentence
//! model for. The steps, each of which [`Rules`] can turn off:
//!
//! 1. Every character gets its UAX #29 `Sentence_Break` class (Unicode
//!    18.0). A language can make more characters end a sentence (the Greek
//!    question mark `;`) or fewer (the full stop in Armenian), and U+2026
//!    HORIZONTAL ELLIPSIS ends one as a full stop does.
//! 2. With [`LineBreaks::Paragraph`], the default, a line break inside a
//!    paragraph is a space and only a blank line or U+2029 separates
//!    paragraphs; with [`LineBreaks::Sentence`] every line break ends a
//!    sentence, as UAX #29 has it.
//! 3. The UAX #29 rules SB1 to SB998 place the boundaries.
//! 4. Numbered references after a terminator (`end.[1] Next`) stay with
//!    their sentence.
//! 5. A lower case word after `!` or `?` continues the sentence, as one
//!    after a full stop already does in UAX #29.
//! 6. A full stop after one of the language's abbreviations (the CLDR
//!    suppressions or the sentencex lists) or after an initial (`J. Smith`)
//!    ends no sentence, unless one of the language's sentence starters
//!    follows (`and co. They`). The language's bundled
//!    [`AbbreviationInventory`] (English has one) is asked first, and its
//!    verdict stands when it has one.
//! 7. List item markers start sentences, and a full stop inside one ends
//!    none.
//! 8. No boundary falls between a bracket or quotation mark and the one
//!    that closes it.
//!
//! The spans are then trimmed of Unicode `White_Space` and empty ones
//! dropped, as the other detectors' spans are. docs/segmentation.md
//! describes the rules and the data in full.

mod brackets;
pub mod languages;
mod lists;
pub mod profile;
mod refine;
#[rustfmt::skip]
mod tables;
mod uax29;
mod units;

use turbo_analysis::{Annotator, Error, Refs, Span, Spans, View};
use turbo_analysis_layers::Sentences;
use turbo_unicode::WhitespaceMode;

use crate::inventory::{AbbreviationInventories, AbbreviationInventory};
use crate::scan::trim;
use crate::{check_len, fold_hash};
pub use languages::{LANGUAGES, Language, Resolved, resolve};
pub use profile::LanguageRules;
use refine::Text;
use uax29::Break;
use units::Unit;

/// How line breaks are read.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum LineBreaks {
    /// Every line break ends a sentence (UAX #29 rule SB4).
    Sentence,
    /// A line break is a space unless it ends a paragraph: when only
    /// spaces separate it from the next line break. U+2029 always ends a
    /// paragraph. A line that starts with a list item marker starts a
    /// sentence.
    #[default]
    Paragraph,
}

/// Which tailorings of UAX #29 run. [`Rules::ALL`] is the default;
/// [`Rules::NONE`] with [`LineBreaks::Sentence`] is plain UAX #29.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Rules {
    /// The language's own terminators and U+2026 as a full stop.
    pub terminators: bool,
    /// Numbered references stay with the sentence before them.
    pub references: bool,
    /// A lower case word after `!` or `?` continues the sentence.
    pub continuation: bool,
    /// The language's abbreviations end no sentence.
    pub abbreviations: bool,
    /// An initial (one upper case letter and a full stop before a capital
    /// letter) ends no sentence.
    pub initials: bool,
    /// List item markers start sentences.
    pub lists: bool,
    /// No boundary inside brackets and quotations.
    pub brackets: bool,
}

impl Rules {
    pub const ALL: Rules = Rules {
        terminators: true,
        references: true,
        continuation: true,
        abbreviations: true,
        initials: true,
        lists: true,
        brackets: true,
    };
    pub const NONE: Rules = Rules {
        terminators: false,
        references: false,
        continuation: false,
        abbreviations: false,
        initials: false,
        lists: false,
        brackets: false,
    };

    fn bits(self) -> u64 {
        [
            self.terminators,
            self.references,
            self.continuation,
            self.abbreviations,
            self.initials,
            self.lists,
            self.brackets,
        ]
        .iter()
        .enumerate()
        .fold(0, |acc, (i, &on)| acc | (u64::from(on) << i))
    }
}

impl Default for Rules {
    fn default() -> Rules {
        Rules::ALL
    }
}

/// The rule splitter for one language.
#[derive(Clone, Debug)]
pub struct RuleSplitter {
    language: Resolved,
    rules: Rules,
    line_breaks: LineBreaks,
    inventory: AbbreviationInventory,
}

/// Buffers a split reuses, so repeated calls allocate nothing once they
/// have grown.
#[derive(Default)]
pub struct RuleScratch {
    units: Vec<Unit>,
    breaks: Vec<Break>,
    depth: Vec<i32>,
    stack: Vec<(char, u32)>,
    found: Vec<lists::Marker>,
    index: Vec<lists::Key>,
    markers: Vec<lists::Marker>,
    lower: String,
}

impl RuleSplitter {
    /// The splitter for a BCP 47 language tag, with every tailoring,
    /// [`LineBreaks::Paragraph`] and the bundled abbreviation inventory of
    /// the tag ([`AbbreviationInventories::for_language`]): every class for
    /// English, none for a language without one. A tag the table does not
    /// know gets the root rules.
    pub fn new(tag: &str) -> RuleSplitter {
        RuleSplitter {
            language: resolve(tag),
            rules: Rules::ALL,
            line_breaks: LineBreaks::Paragraph,
            inventory: AbbreviationInventories::for_language(tag),
        }
    }

    /// Plain UAX #29: no tailoring, and every line break ends a sentence.
    pub fn uax29() -> RuleSplitter {
        RuleSplitter {
            language: resolve("und"),
            rules: Rules::NONE,
            line_breaks: LineBreaks::Sentence,
            inventory: AbbreviationInventory::EMPTY,
        }
    }

    pub fn with_rules(self, rules: Rules) -> RuleSplitter {
        RuleSplitter { rules, ..self }
    }

    pub fn with_line_breaks(self, line_breaks: LineBreaks) -> RuleSplitter {
        RuleSplitter { line_breaks, ..self }
    }

    /// The same splitter with no abbreviation inventory: the language's
    /// own data alone, as the splitter split before inventories.
    pub fn without_inventory(self) -> RuleSplitter {
        self.with_abbreviations(AbbreviationInventory::EMPTY)
    }

    /// The same splitter with an abbreviation inventory, consulted at a
    /// full stop while [`Rules::abbreviations`] is on, before the
    /// language's own abbreviations and initials: a
    /// [`Verdict::Suppress`](crate::Verdict::Suppress) removes the
    /// boundary, a [`Verdict::Allow`](crate::Verdict::Allow) keeps it.
    pub fn with_abbreviations(self, inventory: AbbreviationInventory) -> RuleSplitter {
        RuleSplitter { inventory, ..self }
    }

    pub fn abbreviations(&self) -> &AbbreviationInventory {
        &self.inventory
    }

    /// How the tag was resolved.
    pub fn language(&self) -> Resolved {
        self.language
    }

    pub fn rules(&self) -> Rules {
        self.rules
    }

    pub fn line_breaks(&self) -> LineBreaks {
        self.line_breaks
    }

    /// Appends to `out` the span of every sentence in `text`, in text order,
    /// as UTF-8 byte offsets.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`.
    pub fn split(&self, text: &str, out: &mut Vec<Span>) {
        self.split_with(text, &mut RuleScratch::default(), out);
    }

    /// [`split`](Self::split) with buffers from `scratch`.
    pub fn split_with(&self, text: &str, scratch: &mut RuleScratch, out: &mut Vec<Span>) {
        assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
        self.boundaries_with(text, scratch);
        let t = Text { text, units: &scratch.units };
        let mut start = 0;
        for b in &scratch.breaks {
            let at = t.offset(b.at as usize);
            push_trimmed(text, start, at, out);
            start = at;
        }
        push_trimmed(text, start, text.len(), out);
    }

    /// The byte offsets where sentences start, after the first, before
    /// trimming: the boundaries inside the text.
    pub fn boundaries(&self, text: &str, out: &mut Vec<u32>) {
        let mut scratch = RuleScratch::default();
        self.boundaries_with(text, &mut scratch);
        out.extend(
            scratch.breaks.iter().map(|b| scratch.units.get(b.at as usize).map_or(text.len() as u32, |u| u.start)),
        );
    }

    /// Leaves the sorted boundaries in `scratch.breaks`.
    fn boundaries_with(&self, text: &str, scratch: &mut RuleScratch) {
        let rules = self.rules;
        let language = self.language.rules();
        units::classify(text, rules, language, self.line_breaks, &mut scratch.units);
        uax29::boundaries(&scratch.units, &mut scratch.breaks);
        let t = Text { text, units: &scratch.units };
        let options = refine::Options {
            abbreviations: rules.abbreviations,
            initials: rules.initials,
            inventory: &self.inventory,
        };
        let lower = &mut scratch.lower;
        let mut kept = 0;
        for k in 0..scratch.breaks.len() {
            let mut b = scratch.breaks[k];
            if rules.references {
                match refine::after_references(&t, b) {
                    Some(moved) => b = moved,
                    None => continue,
                }
            }
            if rules.continuation && refine::continues_lower(&t, b) {
                continue;
            }
            if (options.abbreviations || options.initials)
                && refine::after_abbreviation(&t, b, language, &options, lower)
            {
                continue;
            }
            scratch.breaks[kept] = b;
            kept += 1;
        }
        scratch.breaks.truncate(kept);
        if rules.brackets {
            brackets::protect(text, &scratch.units, &mut scratch.stack, &mut scratch.depth);
            let depth = &scratch.depth;
            scratch.breaks.retain(|b| b.term == uax29::NO_TERM || depth[b.at as usize] <= 0);
        }
        if rules.lists {
            lists::markers(text, &scratch.units, &mut scratch.found, &mut scratch.index, &mut scratch.markers);
            // No boundary inside a marker. The markers are in text order and
            // do not overlap, as are the breaks, so one pass over both does.
            let markers = &scratch.markers;
            let mut next = 0;
            scratch.breaks.retain(|b| {
                while next < markers.len() && markers[next].content < b.at {
                    next += 1;
                }
                !markers.get(next).is_some_and(|m| b.at > m.start && b.at <= m.content)
            });
            for m in &scratch.markers {
                let paragraph_start = m.start == 0 || units::is_para_sep(scratch.units[m.start as usize - 1].class);
                let protected = rules.brackets && scratch.depth[m.start as usize] > 0;
                if !(m.after_bullet || paragraph_start || protected) {
                    scratch.breaks.push(Break { at: m.start, term: uax29::NO_TERM });
                }
            }
            scratch.breaks.sort_unstable_by_key(|b| b.at);
            scratch.breaks.dedup_by_key(|b| b.at);
        }
    }
}

/// Appends the span `start..end` trimmed of Unicode `White_Space`, when
/// anything is left.
fn push_trimmed(text: &str, start: usize, end: usize, out: &mut Vec<Span>) {
    if let Some((s, e)) = trim(text, start, end, WhitespaceMode::Unicode) {
        out.push(Span::new(s as u32, e as u32));
    }
}

/// Writes the sentences of a view with the rule splitter.
#[derive(Clone, Debug)]
pub struct RuleSentenceDetector {
    pub splitter: RuleSplitter,
}

impl RuleSentenceDetector {
    /// The detector for a BCP 47 language tag, with the default rules.
    pub fn new(tag: &str) -> RuleSentenceDetector {
        RuleSentenceDetector { splitter: RuleSplitter::new(tag) }
    }
}

thread_local! {
    /// The buffers of the rule annotator, one set per thread, so that a
    /// pipeline running many documents allocates only while they grow.
    static RULE_SCRATCH: std::cell::RefCell<RuleScratch> = std::cell::RefCell::new(RuleScratch::default());
}

impl Annotator for RuleSentenceDetector {
    const ID: &'static str = "turbo-sentdetect:rules";
    /// Changes whenever the rules or the data change what is split.
    const VERSION: &'static str = "3";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences,), Error> {
        let text = view.text();
        check_len(Self::ID, text)?;
        let mut out = Vec::new();
        RULE_SCRATCH.with_borrow_mut(|scratch| self.splitter.split_with(text, scratch, &mut out));
        let mut spans = Spans::with_capacity(view.id(), out.len());
        for s in out {
            spans.push(s);
        }
        Ok((Sentences { spans },))
    }

    /// FNV-1a over the code of the language whose rules apply, the
    /// tailorings and the line break mode, then the abbreviation
    /// inventory's hash when there is one.
    fn config_hash(&self) -> u64 {
        let s = &self.splitter;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |b: u8| h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        for b in s.language.rules_code().bytes() {
            eat(b);
        }
        eat(0);
        for b in s.rules.bits().to_le_bytes() {
            eat(b);
        }
        eat(match s.line_breaks {
            LineBreaks::Sentence => 1,
            LineBreaks::Paragraph => 2,
        });
        if s.inventory.is_unset() { h } else { fold_hash(h, s.inventory.config_hash()) }
    }
}

#[cfg(test)]
mod tests;
