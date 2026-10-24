//! Rule-based multi-word token (MWT) expander for Universal Dependencies
//! contractions in Spanish, Catalan, French, Portuguese, Italian, German and
//! English.

use turbo_analysis::{Annotator, Error, Ix, IxRange, Refs, Span, View};

use super::Words;
use crate::Tokens;

/// The language [`RuleMwtExpander`] splits. The tables are the closed
/// class of contractions each language's grammar has, written here from
/// that grammar and checked against how the Universal Dependencies
/// treebanks segment them; no treebank data is in them. A table splits a
/// form wherever it appears, so a homograph is split too (Portuguese
/// `nos` the clitic pronoun as well as `em` + `os`, `pelo` the noun as
/// well as `por` + `o`, Italian `dai` the verb as well as `da` + `i`); a
/// tagger or a model over the words decides between the readings.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MwtLanguage {
    /// Spanish (`del` -> `de` + `el`, `al` -> `a` + `el`).
    Spanish,
    /// Catalan (`del`/`dels`, `al`/`als`, `pel`/`pels`; `cal` is left
    /// whole, since it is the verb far more often than `ca` + `el`).
    Catalan,
    /// French (`au`/`aux`, `du`, `auquel`/`auxquels`/`auxquelles`,
    /// `duquel`/`desquels`/`desquelles`; `des` is left whole, as the
    /// French treebanks keep the plural article `des` one token).
    French,
    /// Portuguese (`do`/`da`/`dos`/`das`, `no`/`na`/`nos`/`nas`,
    /// `ao`/`à`/`aos`/`às`, `pelo`/`pela`/`pelos`/`pelas`, `dum`/`num`,
    /// demonstrative and pronoun contractions `deste`, `nesse`, `daquele`,
    /// `dele`, `nele`, etc.).
    Portuguese,
    /// Italian articulated prepositions (`del`/`dello`/`della`/`dei`/`degli`/`delle`,
    /// `al`/`allo`/`alla`/`ai`/`agli`/`alle`, `dal`/`dallo`/`dalla`/`dai`/`dagli`/`dalle`,
    /// `nel`/`nello`/`nella`/`nei`/`negli`/`nelle`, `col`/`coi`,
    /// `sul`/`sullo`/`sulla`/`sui`/`sugli`/`sulle`).
    Italian,
    /// German preposition + article contractions (`am`, `ans`, `beim`, `im`,
    /// `ins`, `vom`, `zum`, `zur`, `aufs`, `fürs`, `durchs`, `ums`, `übers`,
    /// `unters`, `hinters`, `vors`).
    German,
    /// English apostrophe clitics (`can't` -> `ca` + `n't`, `won't` -> `wo` +
    /// `n't`, `<aux>n't` -> `<aux>` + `n't`, `'re`, `'ve`, `'ll`, `'m`, `'s`,
    /// `'d`), whose parts are contiguous substrings of the surface token and
    /// therefore populate `Words.subspan`.
    English,
}

impl MwtLanguage {
    /// Parses an ISO 639-1 or 639-3 language code (`"es"`, `"spa"`, `"ca"`,
    /// `"cat"`, `"fr"`, `"fra"`, `"pt"`, `"por"`, `"it"`, `"ita"`, `"de"`,
    /// `"deu"`, `"en"`, `"eng"`).
    pub fn from_code(code: &str) -> Option<MwtLanguage> {
        match code.trim().to_ascii_lowercase().as_str() {
            "es" | "spa" => Some(MwtLanguage::Spanish),
            "ca" | "cat" => Some(MwtLanguage::Catalan),
            "fr" | "fra" | "fre" => Some(MwtLanguage::French),
            "pt" | "por" => Some(MwtLanguage::Portuguese),
            "it" | "ita" => Some(MwtLanguage::Italian),
            "de" | "deu" | "ger" => Some(MwtLanguage::German),
            "en" | "eng" => Some(MwtLanguage::English),
            _ => None,
        }
    }
}

/// Deterministic rule-based multi-word token (MWT) expander that reads `(View,
/// Tokens)` and writes [`Words`] (`turbo:words`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RuleMwtExpander {
    pub language: MwtLanguage,
}

impl RuleMwtExpander {
    pub const fn new(language: MwtLanguage) -> RuleMwtExpander {
        RuleMwtExpander { language }
    }

    /// Expands a single surface token `surface` at `token_span` into either:
    /// - `Some(TokenExpansion::Forms([w1, w2]))` for fusional contractions
    ///   whose syntactic words are not substrings of the token (adjusted to the
    ///   surface casing), or
    /// - `Some(TokenExpansion::Subspans([s1, s2]))` for substring clitic splits
    ///   (`"can't"` -> `"ca"` + `"n't"`).
    pub fn expand_token(&self, surface: &str, token_span: Span) -> Option<TokenExpansion> {
        match self.language {
            MwtLanguage::English => expand_english_subspans(surface, token_span).map(TokenExpansion::Subspans),
            lang => {
                // The longest form in any table is ten bytes.
                if surface.len() > 10 {
                    return None;
                }
                // All table forms are Latin. Per-character Unicode lowercase
                // preserves their matching; contextual Greek sigma cannot match.
                // If folding exceeds ten bytes it cannot be a table form.
                let mut lower = [0u8; 10];
                let mut used = 0;
                for c in surface.chars().flat_map(char::to_lowercase) {
                    let end = used + c.len_utf8();
                    c.encode_utf8(lower.get_mut(used..end)?);
                    used = end;
                }
                let lower = std::str::from_utf8(&lower[..used]).expect("encoded UTF-8");
                let (w1, w2) = lookup_fusional(lang, lower)?;
                Some(TokenExpansion::Forms(apply_case(surface, w1, w2)))
            }
        }
    }

    /// Builds the [`Words`] layer for `tokens` over `text`. When no token in
    /// the document triggers an expansion rule, `form` and `subspan` remain
    /// `None`.
    pub fn expand(&self, text: &str, tokens: &Tokens) -> Words {
        let mut words = Words::new();
        for i in 0..tokens.spans.len() {
            let t = Ix::<Tokens>::new(i as u32);
            let span = tokens.spans.get(t);
            let surface = &text[span.range()];
            let range = IxRange::new(i as u32, i as u32 + 1);
            match self.expand_token(surface, span) {
                None => {
                    words.push_token(t);
                }
                Some(TokenExpansion::Forms([w1, w2])) => {
                    words.push_word(range, Some(&w1), None);
                    words.push_word(range, Some(&w2), None);
                }
                Some(TokenExpansion::Subspans([s1, s2])) => {
                    words.push_word(range, None, Some(s1));
                    words.push_word(range, None, Some(s2));
                }
            }
        }
        words
    }
}

/// The result of expanding a single surface token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenExpansion {
    /// Two syntactic words whose forms are not substrings of the surface token
    /// (`"del"` -> `["de", "el"]`).
    Forms([String; 2]),
    /// Two contiguous byte sub-spans of the surface token (`"can't"` ->
    /// `["ca", "n't"]`).
    Subspans([Span; 2]),
}

impl Annotator for RuleMwtExpander {
    const ID: &'static str = "turbo:mwt-expander";
    const VERSION: &'static str = "2";
    type Reads = (View, Tokens);
    type Writes = (Words,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        tokens.check_contiguous_bounds_in(view).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        let words = self.expand(view.text(), tokens);
        words.check_against(view.text(), tokens).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((words,))
    }

    /// FNV-1a over the language, the one setting the expander has.
    fn config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in (self.language as u64).to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

fn lookup_fusional(lang: MwtLanguage, lower: &str) -> Option<(&'static str, &'static str)> {
    match lang {
        MwtLanguage::Spanish => match lower {
            "del" => Some(("de", "el")),
            "al" => Some(("a", "el")),
            _ => None,
        },
        MwtLanguage::Catalan => match lower {
            "al" => Some(("a", "el")),
            "als" => Some(("a", "els")),
            "del" => Some(("de", "el")),
            "dels" => Some(("de", "els")),
            "pel" => Some(("per", "el")),
            "pels" => Some(("per", "els")),
            _ => None,
        },
        MwtLanguage::French => match lower {
            "au" => Some(("à", "le")),
            "aux" => Some(("à", "les")),
            "du" => Some(("de", "le")),
            "auquel" => Some(("à", "lequel")),
            "auxquels" => Some(("à", "lesquels")),
            "auxquelles" => Some(("à", "lesquelles")),
            "duquel" => Some(("de", "lequel")),
            "desquels" => Some(("de", "lesquels")),
            "desquelles" => Some(("de", "lesquelles")),
            _ => None,
        },
        MwtLanguage::Portuguese => match lower {
            "do" => Some(("de", "o")),
            "da" => Some(("de", "a")),
            "dos" => Some(("de", "os")),
            "das" => Some(("de", "as")),
            "no" => Some(("em", "o")),
            "na" => Some(("em", "a")),
            "nos" => Some(("em", "os")),
            "nas" => Some(("em", "as")),
            "ao" => Some(("a", "o")),
            "à" => Some(("a", "a")),
            "aos" => Some(("a", "os")),
            "às" => Some(("a", "as")),
            "pelo" => Some(("por", "o")),
            "pela" => Some(("por", "a")),
            "pelos" => Some(("por", "os")),
            "pelas" => Some(("por", "as")),
            "dum" => Some(("de", "um")),
            "duma" => Some(("de", "uma")),
            "duns" => Some(("de", "uns")),
            "dumas" => Some(("de", "umas")),
            "num" => Some(("em", "um")),
            "numa" => Some(("em", "uma")),
            "nuns" => Some(("em", "uns")),
            "numas" => Some(("em", "umas")),
            "deste" => Some(("de", "este")),
            "desta" => Some(("de", "esta")),
            "destes" => Some(("de", "estes")),
            "destas" => Some(("de", "estas")),
            "disto" => Some(("de", "isto")),
            "desse" => Some(("de", "esse")),
            "dessa" => Some(("de", "essa")),
            "desses" => Some(("de", "esses")),
            "dessas" => Some(("de", "essas")),
            "disso" => Some(("de", "isso")),
            "daquele" => Some(("de", "aquele")),
            "daquela" => Some(("de", "aquela")),
            "daqueles" => Some(("de", "aqueles")),
            "daquelas" => Some(("de", "aquelas")),
            "daquilo" => Some(("de", "aquilo")),
            "neste" => Some(("em", "este")),
            "nesta" => Some(("em", "esta")),
            "nestes" => Some(("em", "estes")),
            "nestas" => Some(("em", "estas")),
            "nisto" => Some(("em", "isto")),
            "nesse" => Some(("em", "esse")),
            "nessa" => Some(("em", "essa")),
            "nesses" => Some(("em", "esses")),
            "nessas" => Some(("em", "essas")),
            "nisso" => Some(("em", "isso")),
            "naquele" => Some(("em", "aquele")),
            "naquela" => Some(("em", "aquela")),
            "naqueles" => Some(("em", "aqueles")),
            "naquelas" => Some(("em", "aquelas")),
            "naquilo" => Some(("em", "aquilo")),
            "àquele" => Some(("a", "aquele")),
            "àquela" => Some(("a", "aquela")),
            "àqueles" => Some(("a", "aqueles")),
            "àquelas" => Some(("a", "aquelas")),
            "àquilo" => Some(("a", "aquilo")),
            "dele" => Some(("de", "ele")),
            "dela" => Some(("de", "ela")),
            "deles" => Some(("de", "eles")),
            "delas" => Some(("de", "elas")),
            "nele" => Some(("em", "ele")),
            "nela" => Some(("em", "ela")),
            "neles" => Some(("em", "eles")),
            "nelas" => Some(("em", "elas")),
            _ => None,
        },
        MwtLanguage::Italian => match lower {
            "del" => Some(("di", "il")),
            "dello" => Some(("di", "lo")),
            "della" => Some(("di", "la")),
            "dei" => Some(("di", "i")),
            "degli" => Some(("di", "gli")),
            "delle" => Some(("di", "le")),
            "al" => Some(("a", "il")),
            "allo" => Some(("a", "lo")),
            "alla" => Some(("a", "la")),
            "ai" => Some(("a", "i")),
            "agli" => Some(("a", "gli")),
            "alle" => Some(("a", "le")),
            "dal" => Some(("da", "il")),
            "dallo" => Some(("da", "lo")),
            "dalla" => Some(("da", "la")),
            "dai" => Some(("da", "i")),
            "dagli" => Some(("da", "gli")),
            "dalle" => Some(("da", "le")),
            "nel" => Some(("in", "il")),
            "nello" => Some(("in", "lo")),
            "nella" => Some(("in", "la")),
            "nei" => Some(("in", "i")),
            "negli" => Some(("in", "gli")),
            "nelle" => Some(("in", "le")),
            "col" => Some(("con", "il")),
            "coi" => Some(("con", "i")),
            "sul" => Some(("su", "il")),
            "sullo" => Some(("su", "lo")),
            "sulla" => Some(("su", "la")),
            "sui" => Some(("su", "i")),
            "sugli" => Some(("su", "gli")),
            "sulle" => Some(("su", "le")),
            _ => None,
        },
        MwtLanguage::German => match lower {
            "am" => Some(("an", "dem")),
            "ans" => Some(("an", "das")),
            "beim" => Some(("bei", "dem")),
            "im" => Some(("in", "dem")),
            "ins" => Some(("in", "das")),
            "vom" => Some(("von", "dem")),
            "zum" => Some(("zu", "dem")),
            "zur" => Some(("zu", "der")),
            "aufs" => Some(("auf", "das")),
            "fürs" => Some(("für", "das")),
            "durchs" => Some(("durch", "das")),
            "ums" => Some(("um", "das")),
            "übers" => Some(("über", "das")),
            "unters" => Some(("unter", "das")),
            "hinters" => Some(("hinter", "das")),
            "vors" => Some(("vor", "das")),
            _ => None,
        },
        MwtLanguage::English => None,
    }
}

fn apply_case(surface: &str, w1: &'static str, w2: &'static str) -> [String; 2] {
    let mut chars = surface.chars();
    let first_upper = chars.next().is_some_and(char::is_uppercase);
    let all_upper = first_upper && surface.chars().all(|c| !c.is_lowercase());
    if all_upper && surface.chars().count() > 1 {
        [w1.to_uppercase(), w2.to_uppercase()]
    } else if first_upper {
        let mut c1 = w1.chars();
        let cap1 = match c1.next() {
            Some(first) => first.to_uppercase().collect::<String>() + c1.as_str(),
            None => String::new(),
        };
        [cap1, w2.to_owned()]
    } else {
        [w1.to_owned(), w2.to_owned()]
    }
}

/// Splits English clitic contractions whose pieces are contiguous substrings
/// of the surface token (`"can't"` -> `"ca"` + `"n't"`, `"we're"` -> `"we"` +
/// `"'re"`, `"it's"` -> `"it"` + `"'s"`).
fn expand_english_subspans(surface: &str, span: Span) -> Option<[Span; 2]> {
    let bytes = surface.as_bytes();
    let (apos_idx, apos_len) = find_apostrophe(bytes)?;
    if apos_idx == 0 || apos_idx + apos_len >= bytes.len() {
        return None;
    }
    let before = &surface[..apos_idx];
    let after = &surface[apos_idx + apos_len..];
    // The host is letters of any script (`José's`); the clitic is ASCII.
    if !before.chars().all(char::is_alphabetic) || !after.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    if after.eq_ignore_ascii_case("t") && before.len() >= 2 {
        let prev = before.as_bytes()[before.len() - 1];
        if prev.eq_ignore_ascii_case(&b'n') {
            let split_at = span.start + (apos_idx as u32 - 1);
            return Some([Span::new(span.start, split_at), Span::new(split_at, span.end)]);
        }
    }
    if after.eq_ignore_ascii_case("re")
        || after.eq_ignore_ascii_case("ve")
        || after.eq_ignore_ascii_case("ll")
        || after.eq_ignore_ascii_case("m")
        || after.eq_ignore_ascii_case("s")
        || after.eq_ignore_ascii_case("d")
    {
        let split_at = span.start + apos_idx as u32;
        return Some([Span::new(span.start, split_at), Span::new(split_at, span.end)]);
    }
    None
}

fn find_apostrophe(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' => return Some((i, 1)),
            0xE2 if bytes[i + 1..].starts_with(&[0x80, 0x99]) => return Some((i, 3)),
            0xCA if bytes.get(i + 1) == Some(&0xBC) => return Some((i, 2)),
            _ => i += 1,
        }
    }
    None
}
