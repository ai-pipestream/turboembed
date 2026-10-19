//! Folds accents: removes the non-spacing marks of letters in chosen
//! scripts and maps letters with a stroke and ligatures to ASCII.

use turbo_analysis::ViewBuilder;
use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::normalize::{Form, QuickCheck, canonical_combining_class, normalize, quick_check};
use turbo_unicode::script::{Script, script};

use crate::fold_runs::{RunFold, Scratch, fold_runs, nfd_first};
use crate::{Normalizer, hash_config};

/// Which atomic letters (letters that NFD does not decompose, so removing
/// marks cannot reach them) are mapped to ASCII.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum StrokeLetters {
    /// None: the letters stay as they are.
    Off,
    /// The 19 letters of OpenNLP's list: ø Ø æ Æ œ Œ ß ẞ þ Þ ð Ð đ Đ ł Ł ħ Ħ ı.
    #[default]
    OpenNlp,
    /// OpenNLP's list and the letters of [`EXTENDED_MODIFIED`] and
    /// [`EXTENDED_OTHER`].
    Extended,
}

/// Folds accents the way `AccentFoldCharSequenceNormalizer` does: the text
/// is decomposed (NFD), every non-spacing mark (general category Mn)
/// whose base character belongs to one of the fold scripts is removed,
/// the stroke letters and ligatures are mapped to ASCII, and the result is
/// composed again (NFC). "café" becomes "cafe", Greek ά becomes α,
/// Cyrillic й becomes и, "Straße" becomes "Strasse". Marks on letters of
/// other scripts (Devanagari vowel signs, Arabic and Hebrew points) are
/// kept, and so is a mark with no base character before it.
///
/// The base of a mark is the last character before it in the decomposed
/// text that is not a non-spacing mark; a mapped stroke letter counts as
/// Latin. The default fold scripts are Latin, Greek and Cyrillic.
///
/// Each source character whose fold differs from it is one edit. Where
/// normalization reorders or composes characters across source
/// characters (marks out of canonical order, a base and a separate
/// combining character that compose once a mark between them is gone) the
/// base and its marks form one edit; see `fold_runs`.
///
/// OpenNLP counterpart: `AccentFoldCharSequenceNormalizer`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AccentFold {
    scripts: ScriptSet,
    strokes: StrokeLetters,
}

impl Default for AccentFold {
    fn default() -> AccentFold {
        AccentFold::new()
    }
}

impl AccentFold {
    /// Latin, Greek and Cyrillic, with OpenNLP's stroke letter list.
    pub fn new() -> AccentFold {
        AccentFold::for_scripts(&[Script::Latin, Script::Greek, Script::Cyrillic])
    }

    /// Folds the marks of `scripts` only, with OpenNLP's stroke letter list.
    pub fn for_scripts(scripts: &[Script]) -> AccentFold {
        let mut set = ScriptSet::default();
        for &s in scripts {
            set.insert(s);
        }
        AccentFold { scripts: set, strokes: StrokeLetters::OpenNlp }
    }

    /// Latin, Greek and Cyrillic, with the extended stroke letter list.
    pub fn extended() -> AccentFold {
        AccentFold::new().with_stroke_letters(StrokeLetters::Extended)
    }

    /// The same fold with another stroke letter list.
    pub fn with_stroke_letters(self, strokes: StrokeLetters) -> AccentFold {
        AccentFold { strokes, ..self }
    }

    /// Whether marks on letters of `s` are removed.
    pub fn folds(&self, s: Script) -> bool {
        self.scripts.contains(s)
    }

    /// The ASCII replacement of an atomic letter under this configuration.
    pub fn stroke_letter(&self, c: char) -> Option<&'static str> {
        match self.strokes {
            StrokeLetters::Off => None,
            StrokeLetters::OpenNlp => opennlp_stroke_letter(c),
            StrokeLetters::Extended => opennlp_stroke_letter(c).or_else(|| extended_stroke_letter(c)),
        }
    }

    /// Appends the fold of one character `c` of the decomposed text, given
    /// and updating the script of the last base character.
    fn push_folded(&self, c: char, base: &mut Option<Script>, out: &mut String) {
        if !c.is_ascii() && general_category(c) == GeneralCategory::Mn {
            if !base.is_some_and(|s| self.folds(s)) {
                out.push(c);
            }
        } else if let Some(ascii) = self.stroke_letter(c) {
            out.push_str(ascii);
            *base = Some(Script::Latin);
        } else {
            out.push(c);
            *base = Some(script(c));
        }
    }

    /// NFD of `text`, marks removed and stroke letters mapped, NFC of that.
    fn fold_into(&self, text: &str, base: &mut Option<Script>, out: &mut String, s: &mut Scratch) {
        s.decomposed.clear();
        normalize(Form::Nfd, text, &mut s.decomposed);
        s.mapped.clear();
        for c in s.decomposed.chars() {
            self.push_folded(c, base, &mut s.mapped);
        }
        normalize(Form::Nfc, &s.mapped, out);
    }
}

impl RunFold for AccentFold {
    type State = Option<Script>;

    /// A character whose decomposition starts with a character that is not
    /// a non-spacing mark, has combining class 0 and cannot be the second
    /// half of a composition. The character is never removed, so it is the
    /// base of the marks after it whatever came before, and NFC cannot
    /// combine it with anything before it.
    fn is_boundary(&self, c: char) -> bool {
        if c.is_ascii() {
            return true;
        }
        let first = nfd_first(c);
        canonical_combining_class(first) == 0
            && quick_check(Form::Nfc, first) == QuickCheck::Yes
            && general_category(first) != GeneralCategory::Mn
    }

    /// A boundary that neither decomposes nor maps. Hangul syllables
    /// decompose to jamo that compose back to the same syllable.
    fn is_inert(&self, c: char) -> bool {
        if c.is_ascii() || ('\u{AC00}'..='\u{D7A3}').contains(&c) {
            return true;
        }
        canonical_combining_class(c) == 0
            && quick_check(Form::Nfd, c) == QuickCheck::Yes
            && quick_check(Form::Nfc, c) == QuickCheck::Yes
            && general_category(c) != GeneralCategory::Mn
            && self.stroke_letter(c).is_none()
    }

    fn fold_run(&self, run: &str, out: &mut String, s: &mut Scratch) {
        self.fold_into(run, &mut None, out, s);
    }

    fn fold_char(&self, c: char, base: &mut Option<Script>, out: &mut String, s: &mut Scratch) {
        self.fold_into(c.encode_utf8(&mut [0; 4]), base, out, s);
    }
}

impl Normalizer for AccentFold {
    const ID: &'static str = "turbo:normalize:accent-fold";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        fold_runs(self, text, out);
    }

    fn config_hash(&self) -> u64 {
        hash_config(self)
    }
}

/// A set of scripts, one bit per `Script` value (room for 256 values;
/// Unicode 18.0 has 178).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
struct ScriptSet([u64; 4]);

impl ScriptSet {
    fn insert(&mut self, s: Script) {
        let i = s as usize;
        self.0[i / 64] |= 1 << (i % 64);
    }

    fn contains(&self, s: Script) -> bool {
        let i = s as usize;
        self.0[i / 64] & (1 << (i % 64)) != 0
    }
}

/// OpenNLP's list: atomic Latin letters that NFD does not decompose.
fn opennlp_stroke_letter(c: char) -> Option<&'static str> {
    Some(match c {
        '\u{00F8}' => "o",  // o with stroke
        '\u{00D8}' => "O",  // O with stroke
        '\u{00E6}' => "ae", // ae ligature
        '\u{00C6}' => "AE", // AE ligature
        '\u{0153}' => "oe", // oe ligature
        '\u{0152}' => "OE", // OE ligature
        '\u{00DF}' => "ss", // sharp s
        '\u{1E9E}' => "SS", // capital sharp s
        '\u{00FE}' => "th", // thorn
        '\u{00DE}' => "TH", // capital thorn
        '\u{00F0}' => "d",  // eth
        '\u{00D0}' => "D",  // capital eth
        '\u{0111}' => "d",  // d with stroke
        '\u{0110}' => "D",  // D with stroke
        '\u{0142}' => "l",  // l with stroke
        '\u{0141}' => "L",  // L with stroke
        '\u{0127}' => "h",  // h with stroke
        '\u{0126}' => "H",  // H with stroke
        '\u{0131}' => "i",  // dotless i
        _ => return None,
    })
}

/// The extended list, for characters OpenNLP's list does not have.
fn extended_stroke_letter(c: char) -> Option<&'static str> {
    lookup(EXTENDED_MODIFIED, c).or_else(|| lookup(EXTENDED_OTHER, c))
}

fn lookup(table: &[(char, &'static str)], c: char) -> Option<&'static str> {
    table.binary_search_by_key(&c, |&(k, _)| k).ok().map(|i| table[i].1)
}

mod tables;

pub use tables::{EXTENDED_MODIFIED, EXTENDED_OTHER};
