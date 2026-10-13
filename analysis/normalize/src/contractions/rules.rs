//! Which apostrophe forms are contractions, and what each expands to.

/// The apostrophes a contraction may use: ASCII U+0027, the typographic
/// U+2019, the modifier letter U+02BC and the fullwidth U+FF07.
pub(crate) fn apostrophe_at(bytes: &[u8], i: usize) -> Option<usize> {
    match bytes[i] {
        b'\'' => Some(1),
        0xE2 if bytes[i + 1..].starts_with(&[0x80, 0x99]) => Some(3),
        0xCA if bytes.get(i + 1) == Some(&0xBC) => Some(2),
        0xEF if bytes[i + 1..].starts_with(&[0xBC, 0x87]) => Some(3),
        _ => None,
    }
}

/// Auxiliaries that form an unambiguous negative with `n't`.
const NEGATIVE_STEMS: &[&str] = &[
    "are", "could", "dare", "did", "do", "does", "had", "has", "have", "is", "may", "might", "must", "need", "ought",
    "should", "was", "were", "would",
];

/// Words whose `'re` is `are`. OpenNLP's list, plus `here`, `these` and
/// `those`.
const ARE_SUBJECTS: &[&str] =
    &["here", "how", "there", "these", "they", "those", "we", "what", "when", "where", "who", "why", "you"];

/// Words whose `'ve` is `have`. OpenNLP's list, plus `may`, `there`,
/// `these`, `those`, `what` and `where`.
const HAVE_SUBJECTS: &[&str] = &[
    "could", "i", "may", "might", "must", "should", "there", "these", "they", "those", "we", "what", "where", "who",
    "would", "you",
];

/// Words whose `'ll` is `will`. OpenNLP's list, plus `here`, `these`,
/// `this`, `those` and `which`.
const WILL_SUBJECTS: &[&str] = &[
    "he", "here", "how", "i", "it", "she", "that", "there", "these", "they", "this", "those", "we", "what", "when",
    "where", "which", "who", "why", "you",
];

/// A suffix the POS-aware normalizer can resolve: `'s` (is or has) and
/// `'d` (would or had).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Ambiguous {
    S,
    D,
}

/// One contraction found in the text, as byte offsets: `start..end` is the
/// whole word. The prefix `start..prefix_end` is kept unless `prefix`
/// replaces it, `prefix_end..suffix_end` is replaced by the expansion, and
/// `suffix_end..end` is kept (it is empty except for `y'all`, whose `all`
/// stays as written).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Contraction {
    pub start: usize,
    pub prefix_end: usize,
    pub suffix_end: usize,
    pub end: usize,
    pub prefix: Option<&'static str>,
    pub suffix: Expansion,
}

/// What the replaced part becomes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Expansion {
    /// A fixed expansion, already in the word's case.
    Fixed(&'static str),
    /// `'s` or `'d`, which only a part-of-speech tag can resolve; `upper`
    /// says whether the word is all uppercase.
    Ambiguous { which: Ambiguous, upper: bool },
}

/// A lowercase expansion and its uppercase form.
type Cased = (&'static str, &'static str);

const NOT: Cased = (" not", " NOT");
const US: Cased = (" us", " US");
const ARE: Cased = (" are", " ARE");
const HAVE: Cased = (" have", " HAVE");
const WILL: Cased = (" will", " WILL");
const AM: Cased = (" am", " AM");

fn pick((lower, upper): Cased, is_upper: bool) -> &'static str {
    if is_upper { upper } else { lower }
}

/// An irregular stem written like the stem it replaces: all uppercase,
/// capitalized, or lowercase.
fn styled(stem: &[u8], forms: [&'static str; 3]) -> &'static str {
    let [lower, capital, upper] = forms;
    if !stem.iter().any(u8::is_ascii_lowercase) {
        upper
    } else if stem[0].is_ascii_uppercase() {
        capital
    } else {
        lower
    }
}

fn is(slice: &[u8], word: &str) -> bool {
    slice.eq_ignore_ascii_case(word.as_bytes())
}

fn listed(list: &[&str], slice: &[u8]) -> bool {
    list.iter().any(|w| is(slice, w))
}

/// The contraction whose apostrophe is the `width` bytes at `apostrophe`,
/// if the ASCII letters around it form one: the word is the run of ASCII
/// letters before the apostrophe, the apostrophe and the run after it.
/// With `ambiguous` false, `'s` and `'d` words are not contractions.
pub(crate) fn contraction_at(bytes: &[u8], apostrophe: usize, width: usize, ambiguous: bool) -> Option<Contraction> {
    let start = apostrophe - bytes[..apostrophe].iter().rev().take_while(|b| b.is_ascii_alphabetic()).count();
    let after = apostrophe + width;
    let end = after + bytes[after..].iter().take_while(|b| b.is_ascii_alphabetic()).count();
    if start == apostrophe || end == after {
        return None;
    }
    let (stem, rest) = (&bytes[start..apostrophe], &bytes[after..end]);
    let upper = !bytes[start..end].iter().any(u8::is_ascii_lowercase);
    let whole = |prefix_end: usize, prefix: Option<&'static str>, cased: Cased| {
        Some(Contraction {
            start,
            prefix_end,
            suffix_end: end,
            end,
            prefix,
            suffix: Expansion::Fixed(pick(cased, upper)),
        })
    };
    let short = apostrophe - 1;
    if is(rest, "t") {
        if is(stem, "can") {
            return whole(apostrophe, None, NOT);
        }
        if is(stem, "won") {
            return whole(short, Some(styled(&bytes[start..short], ["will", "Will", "WILL"])), NOT);
        }
        if is(stem, "shan") {
            return whole(short, Some(styled(&bytes[start..short], ["shall", "Shall", "SHALL"])), NOT);
        }
        if stem.last().is_some_and(|b| b.eq_ignore_ascii_case(&b'n')) && listed(NEGATIVE_STEMS, &stem[..stem.len() - 1])
        {
            return whole(short, None, NOT);
        }
        return None;
    }
    if is(stem, "let") && is(rest, "s") {
        return whole(apostrophe, None, US);
    }
    if is(stem, "y") && is(rest, "all") {
        // Styled by the whole word: the one-letter stem of "Y'all" says
        // nothing. The apostrophe becomes a space and "all" stays.
        let you = styled(&bytes[start..end], ["you", "You", "YOU"]);
        return Some(Contraction {
            start,
            prefix_end: apostrophe,
            suffix_end: after,
            end,
            prefix: Some(you),
            suffix: Expansion::Fixed(" "),
        });
    }
    let cased = if is(rest, "re") && listed(ARE_SUBJECTS, stem) {
        ARE
    } else if is(rest, "ve") && listed(HAVE_SUBJECTS, stem) {
        HAVE
    } else if is(rest, "ll") && listed(WILL_SUBJECTS, stem) {
        WILL
    } else if is(stem, "i") && is(rest, "m") {
        AM
    } else {
        let which = if is(rest, "s") {
            Ambiguous::S
        } else if is(rest, "d") {
            Ambiguous::D
        } else {
            return None;
        };
        return ambiguous.then_some(Contraction {
            start,
            prefix_end: apostrophe,
            suffix_end: end,
            end,
            prefix: None,
            suffix: Expansion::Ambiguous { which, upper },
        });
    };
    whole(apostrophe, None, cased)
}

/// What an ambiguous `'s` or `'d` stands for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Reading {
    Is,
    Has,
    Would,
    Had,
}

impl Reading {
    /// The text that replaces the apostrophe and suffix.
    pub(crate) fn expansion(self, upper: bool) -> &'static str {
        let cased = match self {
            Reading::Is => (" is", " IS"),
            Reading::Has => (" has", " HAS"),
            Reading::Would => (" would", " WOULD"),
            Reading::Had => (" had", " HAD"),
        };
        pick(cased, upper)
    }
}
