//! Word boundaries for Thai, Lao, Khmer and Burmese, which are written
//! without spaces between words, from the dictionaries and LSTM models of
//! ICU4X's segmenter data.
//!
//! The text is cut into runs: a complex script run is a maximal run of
//! characters of one of the four scripts ([`script`]), and everything else
//! is left to the UAX #29 word rules. Inside a run the words come from the
//! script's segmenter ([`Segmenter`]: either [`dictionary`] or [`lstm`]),
//! chosen per script by accuracy on gold word segmentations; a run always
//! starts and ends on a boundary. This is how ICU4X's word segmenter
//! handles complex scripts, except that here a run and the words inside it
//! always start and end on grapheme cluster boundaries (docs/segmentation.md
//! lists the differences).
//!
//! - [`zerotrie`]: the trie the dictionaries are stored in.
//! - [`dictionary`]: the dictionary file and the dictionary segmenter.
//! - [`lstm`]: the `.lstm` file and the bidirectional LSTM segmenter.
//! - [`script`]: which characters form complex script runs.
//! - [`tokenizer`]: the annotators.
//!
//! The data files are in `analysis/tokenize/data/icu4x-segmenter/` with
//! their notices; that directory's README says where they come from.

pub mod dictionary;
pub mod lstm;
pub mod script;
pub mod tokenizer;
pub(crate) mod zerotrie;

use std::sync::OnceLock;

use turbo_unicode::grapheme_boundaries;

pub use dictionary::{Dictionary, DictionaryError};
pub use lstm::{Lstm, LstmError, LstmScratch};
pub use script::{ComplexScript, complex_script};
pub use tokenizer::{DictionaryWordTokenizer, UntypedDictionaryWordTokenizer, dictionary_word_tokens};

use crate::uax29::for_each_segment as uax29_segments;
use crate::utf8::decode;
use script::may_start_complex;

/// The dictionary files, as ICU4X's segmenter data holds them.
static THAI_DICT: &[u8] = include_bytes!("../../data/icu4x-segmenter/thai.dict");
static LAO_DICT: &[u8] = include_bytes!("../../data/icu4x-segmenter/lao.dict");
static KHMER_DICT: &[u8] = include_bytes!("../../data/icu4x-segmenter/khmer.dict");
static BURMESE_DICT: &[u8] = include_bytes!("../../data/icu4x-segmenter/burmese.dict");

/// The LSTM model files, as ICU4X's segmenter data holds them.
static THAI_LSTM: &[u8] = include_bytes!("../../data/icu4x-segmenter/thai.lstm");
static LAO_LSTM: &[u8] = include_bytes!("../../data/icu4x-segmenter/lao.lstm");
static KHMER_LSTM: &[u8] = include_bytes!("../../data/icu4x-segmenter/khmer.lstm");
static BURMESE_LSTM: &[u8] = include_bytes!("../../data/icu4x-segmenter/burmese.lstm");

/// The bundled dictionary file of `script`.
pub fn bundled_dictionary(script: ComplexScript) -> &'static [u8] {
    match script {
        ComplexScript::Thai => THAI_DICT,
        ComplexScript::Lao => LAO_DICT,
        ComplexScript::Khmer => KHMER_DICT,
        ComplexScript::Myanmar => BURMESE_DICT,
    }
}

/// The bundled LSTM model file of `script`.
pub fn bundled_lstm(script: ComplexScript) -> &'static [u8] {
    match script {
        ComplexScript::Thai => THAI_LSTM,
        ComplexScript::Lao => LAO_LSTM,
        ComplexScript::Khmer => KHMER_LSTM,
        ComplexScript::Myanmar => BURMESE_LSTM,
    }
}

/// A word segmenter for one complex script: either a greedy longest-match
/// dictionary or a bidirectional LSTM.
#[derive(Clone, Debug)]
pub enum Segmenter<'a> {
    /// ICU4X's greedy longest-match dictionary segmenter.
    Dictionary(Dictionary<'a>),
    /// ICU4X's bidirectional LSTM code-point segmenter.
    Lstm(Lstm<'a>),
}

impl<'a> Segmenter<'a> {
    /// The word boundaries of `text`, as byte offsets after each word, in
    /// increasing order; the last is the end of the text. Nothing is
    /// reported for an empty text.
    pub fn segment(&self, text: &str, boundary: impl FnMut(usize)) {
        match self {
            Segmenter::Dictionary(d) => d.segment(text, boundary),
            Segmenter::Lstm(l) => l.segment(text, boundary),
        }
    }
}

impl<'a> From<Dictionary<'a>> for Segmenter<'a> {
    fn from(d: Dictionary<'a>) -> Self {
        Segmenter::Dictionary(d)
    }
}

impl<'a> From<Lstm<'a>> for Segmenter<'a> {
    fn from(l: Lstm<'a>) -> Self {
        Segmenter::Lstm(l)
    }
}

/// The word segmenter: UAX #29 rules, with a [`Segmenter`] for each complex
/// script that has one.
#[derive(Clone, Debug)]
pub struct ComplexSegmenter<'a> {
    /// The segmenter of each script, in the order of [`ComplexScript::ALL`].
    segmenters: [Option<Segmenter<'a>>; 4],
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
        ComplexSegmenter::with_segmenters(
            thai.map(Segmenter::Dictionary),
            lao.map(Segmenter::Dictionary),
            khmer.map(Segmenter::Dictionary),
            myanmar.map(Segmenter::Dictionary),
        )
    }

    /// A segmenter with the given [`Segmenter`] (dictionary or LSTM) for
    /// each script; a script without one is segmented by the UAX #29 rules.
    pub fn with_segmenters(
        thai: Option<Segmenter<'a>>,
        lao: Option<Segmenter<'a>>,
        khmer: Option<Segmenter<'a>>,
        myanmar: Option<Segmenter<'a>>,
    ) -> ComplexSegmenter<'a> {
        ComplexSegmenter { segmenters: [thai, lao, khmer, myanmar] }
    }

    /// The segmenter used for `script`, if any.
    pub fn segmenter(&self, script: ComplexScript) -> Option<&Segmenter<'a>> {
        self.segmenters[script as usize].as_ref()
    }

    /// The dictionary used for `script`, when its segmenter is a dictionary.
    pub fn dictionary(&self, script: ComplexScript) -> Option<&Dictionary<'a>> {
        match self.segmenter(script)? {
            Segmenter::Dictionary(d) => Some(d),
            Segmenter::Lstm(_) => None,
        }
    }

    /// The LSTM used for `script`, when its segmenter is an LSTM.
    pub fn lstm(&self, script: ComplexScript) -> Option<&Lstm<'a>> {
        match self.segmenter(script)? {
            Segmenter::Lstm(l) => Some(l),
            Segmenter::Dictionary(_) => None,
        }
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
        // Where the run before ended inside a grapheme cluster, because the
        // cluster went on into a run of another script.
        let mut cut = usize::MAX;
        let mut i = 0usize;
        while i < bytes.len() {
            let lead = bytes[i];
            if lead < 0x80 {
                i += 1;
                continue;
            }
            let (cp, len) = decode(bytes, i);
            let found = if may_start_complex(lead) {
                char::from_u32(cp).and_then(complex_script).and_then(|s| self.segmenter(s).map(|seg| (s, seg)))
            } else {
                None
            };
            let Some((run_script, seg)) = found else {
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
            // its last letter join its last word, with any letters of the
            // same script among them. The one exception: a cluster that
            // goes on into a run of another script (GB9c joins a Khmer
            // COENG to a Myanmar consonant) is cut where that run starts,
            // so each segmenter sees its own run.
            let run_end = end;
            while cluster < i {
                cluster = clusters.next().unwrap_or(bytes.len());
            }
            let start = if i == cut { i } else { cluster };
            while cluster < run_end {
                cluster = clusters.next().unwrap_or(bytes.len());
            }
            let end = match text[run_end..cluster].char_indices().find(|&(_, c)| self.starts_other_run(c, run_script)) {
                Some((k, _)) => {
                    cut = run_end + k;
                    cut
                }
                None => cluster,
            };
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
            seg.segment(&text[start..run_end], |after| {
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

    /// Whether `c` starts a run of a script other than `script` that a
    /// segmenter handles.
    fn starts_other_run(&self, c: char, script: ComplexScript) -> bool {
        complex_script(c).is_some_and(|s| s != script && self.segmenter(s).is_some())
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
pub fn bundled_dictionaries() -> &'static ComplexSegmenter<'static> {
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

/// The segmenter with the four bundled LSTMs, read once.
pub fn bundled_lstms() -> &'static ComplexSegmenter<'static> {
    static SEGMENTER: OnceLock<ComplexSegmenter<'static>> = OnceLock::new();
    SEGMENTER.get_or_init(|| {
        let read = |script| {
            Segmenter::Lstm(Lstm::from_bytes(bundled_lstm(script)).expect("the bundled LSTMs are well formed"))
        };
        ComplexSegmenter::with_segmenters(
            Some(read(ComplexScript::Thai)),
            Some(read(ComplexScript::Lao)),
            Some(read(ComplexScript::Khmer)),
            Some(read(ComplexScript::Myanmar)),
        )
    })
}

/// The default bundled segmenter, choosing the better model per script on
/// gold word-segmentation corpora: LSTM for Thai, dictionary for Burmese,
/// Khmer and Lao.
pub fn bundled() -> &'static ComplexSegmenter<'static> {
    static SEGMENTER: OnceLock<ComplexSegmenter<'static>> = OnceLock::new();
    SEGMENTER.get_or_init(|| {
        let dict = |script| {
            Segmenter::Dictionary(
                Dictionary::from_bytes(bundled_dictionary(script)).expect("the bundled dictionaries are well formed"),
            )
        };
        let lstm = |script| {
            Segmenter::Lstm(Lstm::from_bytes(bundled_lstm(script)).expect("the bundled LSTMs are well formed"))
        };
        ComplexSegmenter::with_segmenters(
            Some(lstm(ComplexScript::Thai)),
            Some(dict(ComplexScript::Lao)),
            Some(dict(ComplexScript::Khmer)),
            Some(dict(ComplexScript::Myanmar)),
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
    fn runs_are_segmented_by_their_segmenter() {
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
    fn a_cluster_across_two_runs_is_cut_where_the_second_starts() {
        // GB9c makes KHMER SIGN COENG and MYANMAR LETTER KA one cluster.
        // Taking the cluster's end as the end of the Khmer run would give
        // the Khmer word the Myanmar letter; the Burmese dictionary sees
        // its whole run instead.
        let text = "\u{1780}\u{17D2}\u{1000}\u{1001}";
        assert_eq!(turbo_unicode::grapheme_boundaries(text).collect::<Vec<_>>(), [0, 9, 12]);
        let burmese = bundled().dictionary(ComplexScript::Myanmar).unwrap();
        let mut words = vec!["\u{1780}\u{17D2}"];
        let mut from = 0;
        burmese.segment("\u{1000}\u{1001}", |end| {
            words.push(&"\u{1000}\u{1001}"[from..end]);
            from = end;
        });
        assert_eq!(segments(text), words);
        assert_eq!(segments(text), ["\u{1780}\u{17D2}", "\u{1000}", "\u{1001}"]);
    }

    #[test]
    fn a_script_without_a_segmenter_keeps_the_uax29_rules() {
        let segmenter = ComplexSegmenter::new(None, None, None, None);
        let text = "ภาษา";
        let mut out = Vec::new();
        segmenter.for_each_segment(text, |s, e| out.push(&text[s as usize..e as usize]));
        let mut uax = Vec::new();
        uax29_segments(text, |s, e| uax.push(&text[s as usize..e as usize]));
        assert_eq!(out, uax);
    }
}
