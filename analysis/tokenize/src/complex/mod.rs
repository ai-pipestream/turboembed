//! Word boundaries for Thai, Lao, Khmer and Burmese, which are written
//! without spaces between words, from the dictionaries of ICU4X's
//! segmenter data.
//!
//! The text is cut into runs: a complex script run is a maximal run of
//! characters of one of the four scripts ([`script`]), and everything else
//! is left to the UAX #29 word rules. Inside a run the words come from the
//! script's dictionary ([`dictionary`]); a run always starts and ends on
//! a boundary. This is how ICU4X's word segmenter (the `neo` state machine
//! with dictionaries loaded) handles complex scripts, except that here a
//! run and the words inside it always start and end on grapheme cluster
//! boundaries (docs/segmentation.md lists the differences).
//!
//! - [`zerotrie`]: the trie the dictionaries are stored in.
//! - [`dictionary`]: the dictionary file and the dictionary segmenter.
//! - [`script`]: which characters form complex script runs.
//! - [`tokenizer`]: the annotators.
//!
//! The dictionaries are in `analysis/tokenize/data/icu4x-segmenter/` with
//! their notices; that directory's README says where they come from.

pub mod dictionary;
pub mod script;
pub mod tokenizer;
pub(crate) mod zerotrie;

use std::sync::OnceLock;

use turbo_unicode::grapheme_boundaries;

pub use dictionary::{Dictionary, DictionaryError};
pub use script::{ComplexScript, complex_script};
pub use tokenizer::{DictionaryWordTokenizer, UntypedDictionaryWordTokenizer, dictionary_word_tokens};

use crate::uax29::for_each_segment as uax29_segments;
use crate::utf8::decode;
use script::may_start_complex;

/// The dictionary files, as ICU4X's segmenter data holds them.
static THAI: &[u8] = include_bytes!("../../data/icu4x-segmenter/thai.dict");
static LAO: &[u8] = include_bytes!("../../data/icu4x-segmenter/lao.dict");
static KHMER: &[u8] = include_bytes!("../../data/icu4x-segmenter/khmer.dict");
static BURMESE: &[u8] = include_bytes!("../../data/icu4x-segmenter/burmese.dict");

/// The bundled dictionary file of `script`.
pub fn bundled_dictionary(script: ComplexScript) -> &'static [u8] {
    match script {
        ComplexScript::Thai => THAI,
        ComplexScript::Lao => LAO,
        ComplexScript::Khmer => KHMER,
        ComplexScript::Myanmar => BURMESE,
    }
}

/// The word segmenter: UAX #29 rules, with a dictionary for each complex
/// script that has one.
#[derive(Clone, Debug)]
pub struct ComplexSegmenter<'a> {
    /// The dictionary of each script, in the order of [`ComplexScript::ALL`].
    dictionaries: [Option<Dictionary<'a>>; 4],
}

impl<'a> ComplexSegmenter<'a> {
    /// A segmenter with the given dictionary for each script; a script
    /// without one is segmented by the UAX #29 rules.
    pub fn new(
        thai: Option<Dictionary<'a>>,
        lao: Option<Dictionary<'a>>,
        khmer: Option<Dictionary<'a>>,
        myanmar: Option<Dictionary<'a>>,
    ) -> ComplexSegmenter<'a> {
        ComplexSegmenter { dictionaries: [thai, lao, khmer, myanmar] }
    }

    /// The dictionary used for `script`, if any.
    pub fn dictionary(&self, script: ComplexScript) -> Option<&Dictionary<'a>> {
        self.dictionaries[script as usize].as_ref()
    }

    /// Calls `segment` with the start and end of every word segment of
    /// `text`, as byte offsets, in order. The segments cover the text:
    /// spaces and punctuation are segments too, as in UAX #29.
    pub fn for_each_segment(&self, text: &str, mut segment: impl FnMut(u32, u32)) {
        let bytes = text.as_bytes();
        // The grapheme cluster boundaries, walked only as far as the last
        // run needs.
        let mut clusters = grapheme_boundaries(text);
        let mut cluster = 0usize;
        let mut stretch = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            let lead = bytes[i];
            if lead < 0x80 {
                i += 1;
                continue;
            }
            let (cp, len) = decode(bytes, i);
            let found = if may_start_complex(lead) {
                char::from_u32(cp).and_then(complex_script).and_then(|s| self.dictionary(s).map(|d| (s, d)))
            } else {
                None
            };
            let Some((run_script, dictionary)) = found else {
                i += len;
                continue;
            };
            let mut end = i + len;
            while let Some(c) = text[end..].chars().next() {
                if complex_script(c) != Some(run_script) {
                    break;
                }
                end += c.len_utf8();
            }
            // Deviation from ICU4X: no boundary falls inside a grapheme
            // cluster at the edges of the run. Marks at its start that
            // belong to the cluster before it join the segment before, and
            // marks or joiners after its end that belong to the cluster of
            // its last letter join its last word.
            let run_end = end;
            while cluster < i {
                cluster = clusters.next().unwrap_or(bytes.len());
            }
            let start = cluster;
            while cluster < run_end {
                cluster = clusters.next().unwrap_or(bytes.len());
            }
            let end = cluster;
            if stretch < i {
                // The UAX #29 segments before the run; the last one takes
                // the marks between `i` and `start`.
                let mut last: Option<(u32, u32)> = None;
                uax29_stretch(text, stretch, i, &mut |s, e| {
                    if let Some((s0, e0)) = last.replace((s, e)) {
                        segment(s0, e0);
                    }
                });
                if let Some((s0, _)) = last {
                    segment(s0, start as u32);
                }
            } else if i < start {
                segment(i as u32, start as u32);
            }
            i = end;
            stretch = end;
            if start >= run_end {
                // The run lies inside one cluster, which the segment before it took.
                continue;
            }
            let mut from = start;
            let mut last: Option<(u32, u32)> = None;
            dictionary.segment(&text[start..run_end], |after| {
                let after = start + after;
                if let Some((s0, e0)) = last.replace((from as u32, after as u32)) {
                    segment(s0, e0);
                }
                from = after;
            });
            if let Some((s0, _)) = last {
                segment(s0, end as u32);
            }
        }
        if stretch < bytes.len() {
            uax29_stretch(text, stretch, bytes.len(), &mut segment);
        }
    }

    /// Appends the word boundaries of `text` to `out`: 0, then the end of
    /// every segment.
    pub fn boundaries(&self, text: &str, out: &mut Vec<u32>) {
        out.push(0);
        self.for_each_segment(text, |_, end| out.push(end));
    }
}

/// Runs the UAX #29 rules over `text[from..to]` on its own.
fn uax29_stretch(text: &str, from: usize, to: usize, segment: &mut impl FnMut(u32, u32)) {
    let base = from as u32;
    uax29_segments(&text[from..to], |s, e| segment(base + s, base + e));
}

/// The segmenter with the four bundled dictionaries, read once.
pub fn bundled() -> &'static ComplexSegmenter<'static> {
    static SEGMENTER: OnceLock<ComplexSegmenter<'static>> = OnceLock::new();
    SEGMENTER.get_or_init(|| {
        let read = |script| {
            Dictionary::from_bytes(bundled_dictionary(script)).expect("the bundled dictionaries are well formed")
        };
        ComplexSegmenter::new(
            Some(read(ComplexScript::Thai)),
            Some(read(ComplexScript::Lao)),
            Some(read(ComplexScript::Khmer)),
            Some(read(ComplexScript::Myanmar)),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        bundled().for_each_segment(text, |s, e| out.push(&text[s as usize..e as usize]));
        out
    }

    #[test]
    fn runs_are_segmented_by_their_dictionary() {
        assert_eq!(segments("ภาษาไทยภาษาไทย"), ["ภาษา", "ไทย", "ภาษา", "ไทย"]);
        assert_eq!(segments("ພາສາລາວພາສາລາວພາສາລາວ"), ["ພາສາ", "ລາວ", "ພາສາ", "ລາວ", "ພາສາ", "ລາວ"]);
        assert_eq!(segments("ភាសាខ្មែរភាសាខ្មែរភាសាខ្មែរ"), ["ភាសាខ្មែរ", "ភាសាខ្មែរ", "ភាសាខ្មែរ"]);
        assert_eq!(segments("မြန်မာစာမြန်မာစာမြန်မာစာ"), ["မြန်မာ", "စာ", "မြန်မာ", "စာ", "မြန်မာ", "စာ"]);
    }

    #[test]
    fn other_text_keeps_the_uax29_rules() {
        assert_eq!(segments("Hello, world"), ["Hello", ",", " ", "world"]);
        assert_eq!(segments("a ภาษาไทย b"), ["a", " ", "ภาษา", "ไทย", " ", "b"]);
        assert_eq!(segments(""), Vec::<&str>::new());
    }

    #[test]
    fn runs_are_trimmed_to_grapheme_clusters() {
        // A Thai mark after a Latin letter is in the letter's cluster: it
        // joins the segment before (ICU4X starts the run at the mark).
        assert_eq!(segments("a\u{0E31}ภาษา"), ["a\u{0E31}", "ภาษา"]);
        // A combining mark or joiner of no script after a run is in the
        // cluster of the run's last letter: it joins the last word (ICU4X
        // ends the run before it, and the mark is a segment of its own).
        assert_eq!(segments("ภาษา\u{0300} b"), ["ภาษา\u{0300}", " ", "b"]);
        assert_eq!(segments("ไทย\u{200D}"), ["ไทย\u{200D}"]);
        // A run that lies inside one cluster has no word of its own.
        assert_eq!(segments(" \u{0E31} a"), [" \u{0E31}", " ", "a"]);
        // A Thai SARA AM after a Latin letter: UAX #29 alone would break
        // before it, inside the cluster.
        assert_eq!(segments("a\u{0E4B}\u{0E33}"), ["a\u{0E4B}\u{0E33}"]);
    }

    #[test]
    fn a_script_without_a_dictionary_keeps_the_uax29_rules() {
        let segmenter = ComplexSegmenter::new(None, None, None, None);
        let text = "ภาษา";
        let mut out = Vec::new();
        segmenter.for_each_segment(text, |s, e| out.push(&text[s as usize..e as usize]));
        let mut uax = Vec::new();
        uax29_segments(text, |s, e| uax.push(&text[s as usize..e as usize]));
        assert_eq!(out, uax);
    }
}
