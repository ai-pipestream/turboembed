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

/// Every Latin letter whose name in UnicodeData.txt is "LATIN CAPITAL
/// LETTER x WITH ..." or "LATIN SMALL LETTER x WITH ...", where x is one
/// of the letters A to Z, that has no decomposition and is not in
/// OpenNLP's list, mapped to x in the case its name gives: the letters
/// with a stroke, bar, hook, tail, curl, belt, loop, descender or other
/// modification of an ASCII letter that NFD leaves alone. Sorted by code
/// point; `analysis/normalize/tests/accent_folds.rs` derives the list
/// from the Unicode 18.0 data and compares.
pub const EXTENDED_MODIFIED: &[(char, &str)] = &[
    ('\u{0166}', "T"),  // LATIN CAPITAL LETTER T WITH STROKE
    ('\u{0167}', "t"),  // LATIN SMALL LETTER T WITH STROKE
    ('\u{0180}', "b"),  // LATIN SMALL LETTER B WITH STROKE
    ('\u{0181}', "B"),  // LATIN CAPITAL LETTER B WITH HOOK
    ('\u{0182}', "B"),  // LATIN CAPITAL LETTER B WITH TOPBAR
    ('\u{0183}', "b"),  // LATIN SMALL LETTER B WITH TOPBAR
    ('\u{0187}', "C"),  // LATIN CAPITAL LETTER C WITH HOOK
    ('\u{0188}', "c"),  // LATIN SMALL LETTER C WITH HOOK
    ('\u{018A}', "D"),  // LATIN CAPITAL LETTER D WITH HOOK
    ('\u{018B}', "D"),  // LATIN CAPITAL LETTER D WITH TOPBAR
    ('\u{018C}', "d"),  // LATIN SMALL LETTER D WITH TOPBAR
    ('\u{0191}', "F"),  // LATIN CAPITAL LETTER F WITH HOOK
    ('\u{0192}', "f"),  // LATIN SMALL LETTER F WITH HOOK
    ('\u{0193}', "G"),  // LATIN CAPITAL LETTER G WITH HOOK
    ('\u{0197}', "I"),  // LATIN CAPITAL LETTER I WITH STROKE
    ('\u{0198}', "K"),  // LATIN CAPITAL LETTER K WITH HOOK
    ('\u{0199}', "k"),  // LATIN SMALL LETTER K WITH HOOK
    ('\u{019A}', "l"),  // LATIN SMALL LETTER L WITH BAR
    ('\u{019D}', "N"),  // LATIN CAPITAL LETTER N WITH LEFT HOOK
    ('\u{019E}', "n"),  // LATIN SMALL LETTER N WITH LONG RIGHT LEG
    ('\u{019F}', "O"),  // LATIN CAPITAL LETTER O WITH MIDDLE TILDE
    ('\u{01A4}', "P"),  // LATIN CAPITAL LETTER P WITH HOOK
    ('\u{01A5}', "p"),  // LATIN SMALL LETTER P WITH HOOK
    ('\u{01AB}', "t"),  // LATIN SMALL LETTER T WITH PALATAL HOOK
    ('\u{01AC}', "T"),  // LATIN CAPITAL LETTER T WITH HOOK
    ('\u{01AD}', "t"),  // LATIN SMALL LETTER T WITH HOOK
    ('\u{01AE}', "T"),  // LATIN CAPITAL LETTER T WITH RETROFLEX HOOK
    ('\u{01B2}', "V"),  // LATIN CAPITAL LETTER V WITH HOOK
    ('\u{01B3}', "Y"),  // LATIN CAPITAL LETTER Y WITH HOOK
    ('\u{01B4}', "y"),  // LATIN SMALL LETTER Y WITH HOOK
    ('\u{01B5}', "Z"),  // LATIN CAPITAL LETTER Z WITH STROKE
    ('\u{01B6}', "z"),  // LATIN SMALL LETTER Z WITH STROKE
    ('\u{01E4}', "G"),  // LATIN CAPITAL LETTER G WITH STROKE
    ('\u{01E5}', "g"),  // LATIN SMALL LETTER G WITH STROKE
    ('\u{0220}', "N"),  // LATIN CAPITAL LETTER N WITH LONG RIGHT LEG
    ('\u{0221}', "d"),  // LATIN SMALL LETTER D WITH CURL
    ('\u{0224}', "Z"),  // LATIN CAPITAL LETTER Z WITH HOOK
    ('\u{0225}', "z"),  // LATIN SMALL LETTER Z WITH HOOK
    ('\u{0234}', "l"),  // LATIN SMALL LETTER L WITH CURL
    ('\u{0235}', "n"),  // LATIN SMALL LETTER N WITH CURL
    ('\u{0236}', "t"),  // LATIN SMALL LETTER T WITH CURL
    ('\u{023A}', "A"),  // LATIN CAPITAL LETTER A WITH STROKE
    ('\u{023B}', "C"),  // LATIN CAPITAL LETTER C WITH STROKE
    ('\u{023C}', "c"),  // LATIN SMALL LETTER C WITH STROKE
    ('\u{023D}', "L"),  // LATIN CAPITAL LETTER L WITH BAR
    ('\u{023E}', "T"),  // LATIN CAPITAL LETTER T WITH DIAGONAL STROKE
    ('\u{023F}', "s"),  // LATIN SMALL LETTER S WITH SWASH TAIL
    ('\u{0240}', "z"),  // LATIN SMALL LETTER Z WITH SWASH TAIL
    ('\u{0243}', "B"),  // LATIN CAPITAL LETTER B WITH STROKE
    ('\u{0246}', "E"),  // LATIN CAPITAL LETTER E WITH STROKE
    ('\u{0247}', "e"),  // LATIN SMALL LETTER E WITH STROKE
    ('\u{0248}', "J"),  // LATIN CAPITAL LETTER J WITH STROKE
    ('\u{0249}', "j"),  // LATIN SMALL LETTER J WITH STROKE
    ('\u{024B}', "q"),  // LATIN SMALL LETTER Q WITH HOOK TAIL
    ('\u{024C}', "R"),  // LATIN CAPITAL LETTER R WITH STROKE
    ('\u{024D}', "r"),  // LATIN SMALL LETTER R WITH STROKE
    ('\u{024E}', "Y"),  // LATIN CAPITAL LETTER Y WITH STROKE
    ('\u{024F}', "y"),  // LATIN SMALL LETTER Y WITH STROKE
    ('\u{0253}', "b"),  // LATIN SMALL LETTER B WITH HOOK
    ('\u{0255}', "c"),  // LATIN SMALL LETTER C WITH CURL
    ('\u{0256}', "d"),  // LATIN SMALL LETTER D WITH TAIL
    ('\u{0257}', "d"),  // LATIN SMALL LETTER D WITH HOOK
    ('\u{0260}', "g"),  // LATIN SMALL LETTER G WITH HOOK
    ('\u{0266}', "h"),  // LATIN SMALL LETTER H WITH HOOK
    ('\u{0268}', "i"),  // LATIN SMALL LETTER I WITH STROKE
    ('\u{026B}', "l"),  // LATIN SMALL LETTER L WITH MIDDLE TILDE
    ('\u{026C}', "l"),  // LATIN SMALL LETTER L WITH BELT
    ('\u{026D}', "l"),  // LATIN SMALL LETTER L WITH RETROFLEX HOOK
    ('\u{0271}', "m"),  // LATIN SMALL LETTER M WITH HOOK
    ('\u{0272}', "n"),  // LATIN SMALL LETTER N WITH LEFT HOOK
    ('\u{0273}', "n"),  // LATIN SMALL LETTER N WITH RETROFLEX HOOK
    ('\u{027C}', "r"),  // LATIN SMALL LETTER R WITH LONG LEG
    ('\u{027D}', "r"),  // LATIN SMALL LETTER R WITH TAIL
    ('\u{027E}', "r"),  // LATIN SMALL LETTER R WITH FISHHOOK
    ('\u{0282}', "s"),  // LATIN SMALL LETTER S WITH HOOK
    ('\u{0288}', "t"),  // LATIN SMALL LETTER T WITH RETROFLEX HOOK
    ('\u{028B}', "v"),  // LATIN SMALL LETTER V WITH HOOK
    ('\u{0290}', "z"),  // LATIN SMALL LETTER Z WITH RETROFLEX HOOK
    ('\u{0291}', "z"),  // LATIN SMALL LETTER Z WITH CURL
    ('\u{029D}', "j"),  // LATIN SMALL LETTER J WITH CROSSED-TAIL
    ('\u{02A0}', "q"),  // LATIN SMALL LETTER Q WITH HOOK
    ('\u{1D6C}', "b"),  // LATIN SMALL LETTER B WITH MIDDLE TILDE
    ('\u{1D6D}', "d"),  // LATIN SMALL LETTER D WITH MIDDLE TILDE
    ('\u{1D6E}', "f"),  // LATIN SMALL LETTER F WITH MIDDLE TILDE
    ('\u{1D6F}', "m"),  // LATIN SMALL LETTER M WITH MIDDLE TILDE
    ('\u{1D70}', "n"),  // LATIN SMALL LETTER N WITH MIDDLE TILDE
    ('\u{1D71}', "p"),  // LATIN SMALL LETTER P WITH MIDDLE TILDE
    ('\u{1D72}', "r"),  // LATIN SMALL LETTER R WITH MIDDLE TILDE
    ('\u{1D73}', "r"),  // LATIN SMALL LETTER R WITH FISHHOOK AND MIDDLE TILDE
    ('\u{1D74}', "s"),  // LATIN SMALL LETTER S WITH MIDDLE TILDE
    ('\u{1D75}', "t"),  // LATIN SMALL LETTER T WITH MIDDLE TILDE
    ('\u{1D76}', "z"),  // LATIN SMALL LETTER Z WITH MIDDLE TILDE
    ('\u{1D7D}', "p"),  // LATIN SMALL LETTER P WITH STROKE
    ('\u{1D80}', "b"),  // LATIN SMALL LETTER B WITH PALATAL HOOK
    ('\u{1D81}', "d"),  // LATIN SMALL LETTER D WITH PALATAL HOOK
    ('\u{1D82}', "f"),  // LATIN SMALL LETTER F WITH PALATAL HOOK
    ('\u{1D83}', "g"),  // LATIN SMALL LETTER G WITH PALATAL HOOK
    ('\u{1D84}', "k"),  // LATIN SMALL LETTER K WITH PALATAL HOOK
    ('\u{1D85}', "l"),  // LATIN SMALL LETTER L WITH PALATAL HOOK
    ('\u{1D86}', "m"),  // LATIN SMALL LETTER M WITH PALATAL HOOK
    ('\u{1D87}', "n"),  // LATIN SMALL LETTER N WITH PALATAL HOOK
    ('\u{1D88}', "p"),  // LATIN SMALL LETTER P WITH PALATAL HOOK
    ('\u{1D89}', "r"),  // LATIN SMALL LETTER R WITH PALATAL HOOK
    ('\u{1D8A}', "s"),  // LATIN SMALL LETTER S WITH PALATAL HOOK
    ('\u{1D8C}', "v"),  // LATIN SMALL LETTER V WITH PALATAL HOOK
    ('\u{1D8D}', "x"),  // LATIN SMALL LETTER X WITH PALATAL HOOK
    ('\u{1D8E}', "z"),  // LATIN SMALL LETTER Z WITH PALATAL HOOK
    ('\u{1D8F}', "a"),  // LATIN SMALL LETTER A WITH RETROFLEX HOOK
    ('\u{1D91}', "d"),  // LATIN SMALL LETTER D WITH HOOK AND TAIL
    ('\u{1D92}', "e"),  // LATIN SMALL LETTER E WITH RETROFLEX HOOK
    ('\u{1D96}', "i"),  // LATIN SMALL LETTER I WITH RETROFLEX HOOK
    ('\u{1D99}', "u"),  // LATIN SMALL LETTER U WITH RETROFLEX HOOK
    ('\u{1EFE}', "Y"),  // LATIN CAPITAL LETTER Y WITH LOOP
    ('\u{1EFF}', "y"),  // LATIN SMALL LETTER Y WITH LOOP
    ('\u{2C60}', "L"),  // LATIN CAPITAL LETTER L WITH DOUBLE BAR
    ('\u{2C61}', "l"),  // LATIN SMALL LETTER L WITH DOUBLE BAR
    ('\u{2C62}', "L"),  // LATIN CAPITAL LETTER L WITH MIDDLE TILDE
    ('\u{2C63}', "P"),  // LATIN CAPITAL LETTER P WITH STROKE
    ('\u{2C64}', "R"),  // LATIN CAPITAL LETTER R WITH TAIL
    ('\u{2C65}', "a"),  // LATIN SMALL LETTER A WITH STROKE
    ('\u{2C66}', "t"),  // LATIN SMALL LETTER T WITH DIAGONAL STROKE
    ('\u{2C67}', "H"),  // LATIN CAPITAL LETTER H WITH DESCENDER
    ('\u{2C68}', "h"),  // LATIN SMALL LETTER H WITH DESCENDER
    ('\u{2C69}', "K"),  // LATIN CAPITAL LETTER K WITH DESCENDER
    ('\u{2C6A}', "k"),  // LATIN SMALL LETTER K WITH DESCENDER
    ('\u{2C6B}', "Z"),  // LATIN CAPITAL LETTER Z WITH DESCENDER
    ('\u{2C6C}', "z"),  // LATIN SMALL LETTER Z WITH DESCENDER
    ('\u{2C6E}', "M"),  // LATIN CAPITAL LETTER M WITH HOOK
    ('\u{2C71}', "v"),  // LATIN SMALL LETTER V WITH RIGHT HOOK
    ('\u{2C72}', "W"),  // LATIN CAPITAL LETTER W WITH HOOK
    ('\u{2C73}', "w"),  // LATIN SMALL LETTER W WITH HOOK
    ('\u{2C74}', "v"),  // LATIN SMALL LETTER V WITH CURL
    ('\u{2C78}', "e"),  // LATIN SMALL LETTER E WITH NOTCH
    ('\u{2C7A}', "o"),  // LATIN SMALL LETTER O WITH LOW RING INSIDE
    ('\u{2C7E}', "S"),  // LATIN CAPITAL LETTER S WITH SWASH TAIL
    ('\u{2C7F}', "Z"),  // LATIN CAPITAL LETTER Z WITH SWASH TAIL
    ('\u{A740}', "K"),  // LATIN CAPITAL LETTER K WITH STROKE
    ('\u{A741}', "k"),  // LATIN SMALL LETTER K WITH STROKE
    ('\u{A742}', "K"),  // LATIN CAPITAL LETTER K WITH DIAGONAL STROKE
    ('\u{A743}', "k"),  // LATIN SMALL LETTER K WITH DIAGONAL STROKE
    ('\u{A744}', "K"),  // LATIN CAPITAL LETTER K WITH STROKE AND DIAGONAL STROKE
    ('\u{A745}', "k"),  // LATIN SMALL LETTER K WITH STROKE AND DIAGONAL STROKE
    ('\u{A748}', "L"),  // LATIN CAPITAL LETTER L WITH HIGH STROKE
    ('\u{A749}', "l"),  // LATIN SMALL LETTER L WITH HIGH STROKE
    ('\u{A74A}', "O"),  // LATIN CAPITAL LETTER O WITH LONG STROKE OVERLAY
    ('\u{A74B}', "o"),  // LATIN SMALL LETTER O WITH LONG STROKE OVERLAY
    ('\u{A74C}', "O"),  // LATIN CAPITAL LETTER O WITH LOOP
    ('\u{A74D}', "o"),  // LATIN SMALL LETTER O WITH LOOP
    ('\u{A750}', "P"),  // LATIN CAPITAL LETTER P WITH STROKE THROUGH DESCENDER
    ('\u{A751}', "p"),  // LATIN SMALL LETTER P WITH STROKE THROUGH DESCENDER
    ('\u{A752}', "P"),  // LATIN CAPITAL LETTER P WITH FLOURISH
    ('\u{A753}', "p"),  // LATIN SMALL LETTER P WITH FLOURISH
    ('\u{A754}', "P"),  // LATIN CAPITAL LETTER P WITH SQUIRREL TAIL
    ('\u{A755}', "p"),  // LATIN SMALL LETTER P WITH SQUIRREL TAIL
    ('\u{A756}', "Q"),  // LATIN CAPITAL LETTER Q WITH STROKE THROUGH DESCENDER
    ('\u{A757}', "q"),  // LATIN SMALL LETTER Q WITH STROKE THROUGH DESCENDER
    ('\u{A758}', "Q"),  // LATIN CAPITAL LETTER Q WITH DIAGONAL STROKE
    ('\u{A759}', "q"),  // LATIN SMALL LETTER Q WITH DIAGONAL STROKE
    ('\u{A75E}', "V"),  // LATIN CAPITAL LETTER V WITH DIAGONAL STROKE
    ('\u{A75F}', "v"),  // LATIN SMALL LETTER V WITH DIAGONAL STROKE
    ('\u{A78E}', "l"),  // LATIN SMALL LETTER L WITH RETROFLEX HOOK AND BELT
    ('\u{A790}', "N"),  // LATIN CAPITAL LETTER N WITH DESCENDER
    ('\u{A791}', "n"),  // LATIN SMALL LETTER N WITH DESCENDER
    ('\u{A792}', "C"),  // LATIN CAPITAL LETTER C WITH BAR
    ('\u{A793}', "c"),  // LATIN SMALL LETTER C WITH BAR
    ('\u{A794}', "c"),  // LATIN SMALL LETTER C WITH PALATAL HOOK
    ('\u{A795}', "h"),  // LATIN SMALL LETTER H WITH PALATAL HOOK
    ('\u{A796}', "B"),  // LATIN CAPITAL LETTER B WITH FLOURISH
    ('\u{A797}', "b"),  // LATIN SMALL LETTER B WITH FLOURISH
    ('\u{A798}', "F"),  // LATIN CAPITAL LETTER F WITH STROKE
    ('\u{A799}', "f"),  // LATIN SMALL LETTER F WITH STROKE
    ('\u{A7A0}', "G"),  // LATIN CAPITAL LETTER G WITH OBLIQUE STROKE
    ('\u{A7A1}', "g"),  // LATIN SMALL LETTER G WITH OBLIQUE STROKE
    ('\u{A7A2}', "K"),  // LATIN CAPITAL LETTER K WITH OBLIQUE STROKE
    ('\u{A7A3}', "k"),  // LATIN SMALL LETTER K WITH OBLIQUE STROKE
    ('\u{A7A4}', "N"),  // LATIN CAPITAL LETTER N WITH OBLIQUE STROKE
    ('\u{A7A5}', "n"),  // LATIN SMALL LETTER N WITH OBLIQUE STROKE
    ('\u{A7A6}', "R"),  // LATIN CAPITAL LETTER R WITH OBLIQUE STROKE
    ('\u{A7A7}', "r"),  // LATIN SMALL LETTER R WITH OBLIQUE STROKE
    ('\u{A7A8}', "S"),  // LATIN CAPITAL LETTER S WITH OBLIQUE STROKE
    ('\u{A7A9}', "s"),  // LATIN SMALL LETTER S WITH OBLIQUE STROKE
    ('\u{A7AA}', "H"),  // LATIN CAPITAL LETTER H WITH HOOK
    ('\u{A7AD}', "L"),  // LATIN CAPITAL LETTER L WITH BELT
    ('\u{A7B2}', "J"),  // LATIN CAPITAL LETTER J WITH CROSSED-TAIL
    ('\u{A7B8}', "U"),  // LATIN CAPITAL LETTER U WITH STROKE
    ('\u{A7B9}', "u"),  // LATIN SMALL LETTER U WITH STROKE
    ('\u{A7C4}', "C"),  // LATIN CAPITAL LETTER C WITH PALATAL HOOK
    ('\u{A7C5}', "S"),  // LATIN CAPITAL LETTER S WITH HOOK
    ('\u{A7C6}', "Z"),  // LATIN CAPITAL LETTER Z WITH PALATAL HOOK
    ('\u{A7C7}', "D"),  // LATIN CAPITAL LETTER D WITH SHORT STROKE OVERLAY
    ('\u{A7C8}', "d"),  // LATIN SMALL LETTER D WITH SHORT STROKE OVERLAY
    ('\u{A7C9}', "S"),  // LATIN CAPITAL LETTER S WITH SHORT STROKE OVERLAY
    ('\u{A7CA}', "s"),  // LATIN SMALL LETTER S WITH SHORT STROKE OVERLAY
    ('\u{A7CC}', "S"),  // LATIN CAPITAL LETTER S WITH DIAGONAL STROKE
    ('\u{A7CD}', "s"),  // LATIN SMALL LETTER S WITH DIAGONAL STROKE
    ('\u{A7E2}', "R"),  // LATIN CAPITAL LETTER R WITH LONG LEG
    ('\u{AB34}', "e"),  // LATIN SMALL LETTER E WITH FLOURISH
    ('\u{AB37}', "l"),  // LATIN SMALL LETTER L WITH INVERTED LAZY S
    ('\u{AB38}', "l"),  // LATIN SMALL LETTER L WITH DOUBLE MIDDLE TILDE
    ('\u{AB39}', "l"),  // LATIN SMALL LETTER L WITH MIDDLE RING
    ('\u{AB3A}', "m"),  // LATIN SMALL LETTER M WITH CROSSED-TAIL
    ('\u{AB3B}', "n"),  // LATIN SMALL LETTER N WITH CROSSED-TAIL
    ('\u{AB49}', "r"),  // LATIN SMALL LETTER R WITH CROSSED-TAIL
    ('\u{AB4E}', "u"),  // LATIN SMALL LETTER U WITH SHORT RIGHT LEG
    ('\u{AB52}', "u"),  // LATIN SMALL LETTER U WITH LEFT HOOK
    ('\u{AB56}', "x"),  // LATIN SMALL LETTER X WITH LOW RIGHT RING
    ('\u{AB57}', "x"),  // LATIN SMALL LETTER X WITH LONG LEFT LEG
    ('\u{AB58}', "x"),  // LATIN SMALL LETTER X WITH LONG LEFT LEG AND LOW RIGHT RING
    ('\u{AB59}', "x"),  // LATIN SMALL LETTER X WITH LONG LEFT LEG WITH SERIF
    ('\u{AB5A}', "y"),  // LATIN SMALL LETTER Y WITH SHORT RIGHT LEG
    ('\u{1DF09}', "t"), // LATIN SMALL LETTER T WITH HOOK AND RETROFLEX HOOK
    ('\u{1DF11}', "l"), // LATIN SMALL LETTER L WITH FISHHOOK
    ('\u{1DF13}', "l"), // LATIN SMALL LETTER L WITH BELT AND PALATAL HOOK
    ('\u{1DF16}', "r"), // LATIN SMALL LETTER R WITH FISHHOOK AND PALATAL HOOK
    ('\u{1DF1A}', "i"), // LATIN SMALL LETTER I WITH STROKE AND RETROFLEX HOOK
    ('\u{1DF1B}', "o"), // LATIN SMALL LETTER O WITH RETROFLEX HOOK
    ('\u{1DF1D}', "c"), // LATIN SMALL LETTER C WITH RETROFLEX HOOK
    ('\u{1DF1E}', "s"), // LATIN SMALL LETTER S WITH CURL
    ('\u{1DF25}', "d"), // LATIN SMALL LETTER D WITH MID-HEIGHT LEFT HOOK
    ('\u{1DF26}', "l"), // LATIN SMALL LETTER L WITH MID-HEIGHT LEFT HOOK
    ('\u{1DF27}', "n"), // LATIN SMALL LETTER N WITH MID-HEIGHT LEFT HOOK
    ('\u{1DF28}', "r"), // LATIN SMALL LETTER R WITH MID-HEIGHT LEFT HOOK
    ('\u{1DF29}', "s"), // LATIN SMALL LETTER S WITH MID-HEIGHT LEFT HOOK
    ('\u{1DF2A}', "t"), // LATIN SMALL LETTER T WITH MID-HEIGHT LEFT HOOK
    ('\u{1DF2D}', "d"), // LATIN SMALL LETTER D WITH HOOK AND PALATAL HOOK
    ('\u{1DF32}', "h"), // LATIN SMALL LETTER H WITH STROKE AND PALATAL HOOK
    ('\u{1DF34}', "q"), // LATIN SMALL LETTER Q WITH PALATAL HOOK
    ('\u{1DF37}', "r"), // LATIN SMALL LETTER R WITH TAIL AND PALATAL HOOK
    ('\u{1DF39}', "v"), // LATIN SMALL LETTER V WITH HOOK AND PALATAL HOOK
    ('\u{1DF45}', "g"), // LATIN SMALL LETTER G WITH STROKE AND PALATAL HOOK
    ('\u{1DF55}', "y"), // LATIN SMALL LETTER Y WITH LOW STROKE
    ('\u{1DF61}', "r"), // LATIN SMALL LETTER R WITH LEFT TIE
    ('\u{1DF70}', "i"), // LATIN SMALL LETTER I WITH PIGTAIL AT BOTTOM
    ('\u{1DF72}', "O"), // LATIN CAPITAL LETTER O WITH CURL
    ('\u{1DF73}', "o"), // LATIN SMALL LETTER O WITH CURL
    ('\u{1DF7A}', "U"), // LATIN CAPITAL LETTER U WITH HOOK TAIL
    ('\u{1DF7B}', "u"), // LATIN SMALL LETTER U WITH HOOK TAIL
    ('\u{1DF7C}', "U"), // LATIN CAPITAL LETTER U WITH NOTCH AT BOTTOM
    ('\u{1DF7D}', "u"), // LATIN SMALL LETTER U WITH NOTCH AT BOTTOM
    ('\u{1DF80}', "A"), // LATIN CAPITAL LETTER A WITH TOPBAR
    ('\u{1DF81}', "E"), // LATIN CAPITAL LETTER E WITH BENT TOPBAR
    ('\u{1DF90}', "c"), // LATIN SMALL LETTER C WITH LOW SLASH
    ('\u{1DF91}', "c"), // LATIN SMALL LETTER C WITH LOOPED CEDILLA
    ('\u{1DF92}', "c"), // LATIN SMALL LETTER C WITH RIGHT LOOP
];

/// Atomic Latin letters that do not fit the naming rule of
/// [`EXTENDED_MODIFIED`], with the reason for each mapping: the ligature
/// IJ and the digraphs DZ, LJ and NJ (one letter each in Unicode, two
/// letters in ASCII; their compatibility decompositions are those letters
/// with any caron), the L with middle dot of Catalan (its compatibility
/// decomposition is L and U+00B7), the long s (its compatibility
/// decomposition is s), eng (n with a tail, written ng in ASCII
/// transcriptions but mapped to its base letter as the hooked letters
/// are) and kra (written q in the Greenlandic orthography of 1973 that
/// replaced it). Sorted by code point.
pub const EXTENDED_OTHER: &[(char, &str)] = &[
    ('\u{0132}', "IJ"), // LATIN CAPITAL LIGATURE IJ
    ('\u{0133}', "ij"), // LATIN SMALL LIGATURE IJ
    ('\u{0138}', "q"),  // LATIN SMALL LETTER KRA
    ('\u{013F}', "L"),  // LATIN CAPITAL LETTER L WITH MIDDLE DOT
    ('\u{0140}', "l"),  // LATIN SMALL LETTER L WITH MIDDLE DOT
    ('\u{014A}', "N"),  // LATIN CAPITAL LETTER ENG
    ('\u{014B}', "n"),  // LATIN SMALL LETTER ENG
    ('\u{017F}', "s"),  // LATIN SMALL LETTER LONG S
    ('\u{01C4}', "DZ"), // LATIN CAPITAL LETTER DZ WITH CARON
    ('\u{01C5}', "Dz"), // LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON
    ('\u{01C6}', "dz"), // LATIN SMALL LETTER DZ WITH CARON
    ('\u{01C7}', "LJ"), // LATIN CAPITAL LETTER LJ
    ('\u{01C8}', "Lj"), // LATIN CAPITAL LETTER L WITH SMALL LETTER J
    ('\u{01C9}', "lj"), // LATIN SMALL LETTER LJ
    ('\u{01CA}', "NJ"), // LATIN CAPITAL LETTER NJ
    ('\u{01CB}', "Nj"), // LATIN CAPITAL LETTER N WITH SMALL LETTER J
    ('\u{01CC}', "nj"), // LATIN SMALL LETTER NJ
    ('\u{01F1}', "DZ"), // LATIN CAPITAL LETTER DZ
    ('\u{01F2}', "Dz"), // LATIN CAPITAL LETTER D WITH SMALL LETTER Z
    ('\u{01F3}', "dz"), // LATIN SMALL LETTER DZ
];
