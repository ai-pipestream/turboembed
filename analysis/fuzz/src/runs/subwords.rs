//! The WordPiece run: the port's encoder with the tiny BERT bundle, whose
//! vocabulary is the one the JVM's `WordpieceEncoder` ran with.
//!
//! The port follows upstream `tokenizers`; `WordpieceEncoder` follows the
//! reference BERT, and keeps source offsets its own way. The differences
//! are documented and tested in `analysis/embed/tests/goldens.rs`. To tell
//! a documented difference from a divergence on any text, this module
//! models how `WordpieceEncoder` maps its normalized text back to the
//! source (control characters dropped, whitespace made spaces, ideographs
//! and punctuation isolated, simple lowercase with final sigma, accents
//! stripped with the mark's end given to the character before it, and
//! every piece given the whole run when canonical reordering crossed
//! characters) and lays the port's pieces over that mapping. A golden is
//! a documented difference when its spans are exactly what the model
//! gives for the port's pieces.

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_embed::SubwordAnnotator;
use turbo_analysis_layers::Subwords;
use turbo_goldens::{Golden, Tolerance, matches_golden};
use turbo_unicode::GeneralCategory;
use turbo_unicode::category::general_category;
use turbo_unicode::java::to_lower_char;
use turbo_unicode::normalize::{Form, is_normalized, normalize};

use super::{Outcome, Run, RustError, analyze};
use crate::resources::{Resources, Unavailable};

/// The run name of the vocabulary the goldens were made with.
pub(crate) const VOCABULARY: &str = "all-minilm-l6-v2-vocab";

/// `WordpieceEncoder`'s longest word, in normalized code points, that is
/// segmented rather than made unknown; the bundle's `max_input_chars_per_word`.
const MAX_WORD_CODE_POINTS: usize = 100;

pub(crate) struct WordPiece {
    analyzer: Analyzer,
    /// The length in characters of each piece of the vocabulary, the
    /// continuation prefix left out, by id.
    lengths: Vec<usize>,
    unknown: u32,
}

impl WordPiece {
    pub(crate) fn open(res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let dir = res.bert_bundle()?;
        let annotator =
            SubwordAnnotator::open(&dir).map_err(|e| Unavailable::Broken(format!("{}: {e}", dir.display())))?;
        let analyzer = Analyzer::builder().add(annotator).build().map_err(|e| Unavailable::Broken(e.to_string()))?;
        let vocabulary = res.fixture(&format!("{VOCABULARY}.txt"))?;
        let text = std::fs::read_to_string(&vocabulary)
            .map_err(|e| Unavailable::Broken(format!("{}: {e}", vocabulary.display())))?;
        let pieces: Vec<&str> = text.strip_suffix('\n').unwrap_or(&text).split('\n').collect();
        let unknown = pieces
            .iter()
            .position(|p| *p == "[UNK]")
            .ok_or_else(|| Unavailable::Broken(format!("{}: no [UNK] piece", vocabulary.display())))?
            as u32;
        let lengths = pieces.iter().map(|p| p.strip_prefix("##").unwrap_or(p).chars().count()).collect();
        Ok(Box::new(WordPiece { analyzer, lengths, unknown }))
    }
}

/// A piece: its id and byte span.
type Piece = (u32, [u32; 2]);

/// The golden's pieces.
fn jvm_pieces(golden: &Golden) -> Option<Vec<Piece>> {
    let l = golden.layer("opennlp:subwords")?;
    let (start, end, id) = (l.u32s("start")?, l.u32s("end")?, l.u32s("id")?);
    Some((0..l.len()).map(|i| (id[i], [start[i], end[i]])).collect())
}

/// The pieces of `doc`.
fn rust_pieces(doc: &Document) -> Vec<Piece> {
    doc.get::<Subwords>().map_or_else(Vec::new, |sub| {
        sub.ids.iter().zip(sub.spans.iter()).map(|(&id, (_, s))| (id as u32, [s.start, s.end])).collect()
    })
}

/// The strings upstream `tokenizers` matches as special tokens before
/// normalizing, which `WordpieceEncoder` reads as text.
const SPECIAL_TOKENS: [&str; 5] = ["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"];

/// The ideographs the reference BERT counts as CJK and upstream does not.
fn jvm_only_cjk(c: char) -> bool {
    ('\u{2B820}'..='\u{2B91F}').contains(&c)
}

/// One code point of the JVM's mapped text with its source byte span.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Unit {
    c: char,
    start: u32,
    end: u32,
}

/// Which of the documented differences the mapping went through.
#[derive(Default)]
struct Used {
    /// A stripped mark gave its end to the character before it.
    stripped: bool,
    /// Canonical reordering made every character of a run span the run.
    reordered: bool,
    /// A capital sigma was lowercased to final sigma.
    sigma: bool,
}

/// `BertNormalization.isControl`: a `C*` category but for tab, line feed
/// and carriage return.
fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r')
        && matches!(
            general_category(c),
            GeneralCategory::Cc | GeneralCategory::Cf | GeneralCategory::Cs | GeneralCategory::Co | GeneralCategory::Cn
        )
}

/// `BertNormalization.isWhitespace`: space, tab, line feed, carriage
/// return and the space separators, the no-break spaces among them.
fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r') || general_category(c) == GeneralCategory::Zs
}

fn is_line_or_paragraph_separator(c: char) -> bool {
    matches!(general_category(c), GeneralCategory::Zl | GeneralCategory::Zp)
}

/// `BertNormalization.isPunctuation`: ASCII symbols and every punctuation
/// category.
fn is_punctuation(c: char) -> bool {
    matches!(c, '!'..='/' | ':'..='@' | '['..='`' | '{'..='~')
        || matches!(
            general_category(c),
            GeneralCategory::Pc
                | GeneralCategory::Pd
                | GeneralCategory::Ps
                | GeneralCategory::Pe
                | GeneralCategory::Pi
                | GeneralCategory::Pf
                | GeneralCategory::Po
        )
}

/// `BertNormalization.isCjk`: the reference BERT's ideograph ranges.
fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// `WordpieceEncoder.isCaseIgnorable`: the categories and the Unicode
/// 15.0 word break values the `Case_Ignorable` property is made of.
fn case_ignorable(c: char) -> bool {
    matches!(
        general_category(c),
        GeneralCategory::Mn | GeneralCategory::Me | GeneralCategory::Cf | GeneralCategory::Lm | GeneralCategory::Sk
    ) || matches!(
        c as u32,
        0x0027
            | 0x002E
            | 0x003A
            | 0x00B7
            | 0x0387
            | 0x055F
            | 0x05F4
            | 0x2018
            | 0x2019
            | 0x2024
            | 0x2027
            | 0xFE13
            | 0xFE52
            | 0xFE55
            | 0xFF07
            | 0xFF0E
            | 0xFF1A
    )
}

/// `WordpieceEncoder.isCased`: upper, lower or title case.
fn cased(c: char) -> bool {
    c.is_uppercase() || c.is_lowercase() || general_category(c) == GeneralCategory::Lt
}

/// `cleanAndIsolateCjk`: controls dropped, whitespace and separators made
/// spaces, every ideograph between two empty spaces.
fn clean_and_isolate_cjk(text: &str) -> Vec<Unit> {
    let mut out = Vec::with_capacity(text.len());
    for (i, c) in text.char_indices() {
        let (start, end) = (i as u32, (i + c.len_utf8()) as u32);
        if c == '\0' || c == '\u{FFFD}' || is_control(c) {
            continue;
        }
        if is_whitespace(c) || is_line_or_paragraph_separator(c) {
            out.push(Unit { c: ' ', start, end });
        } else if is_cjk(c) {
            out.push(Unit { c: ' ', start, end: start });
            out.push(Unit { c, start, end });
            out.push(Unit { c: ' ', start: end, end });
        } else {
            out.push(Unit { c, start, end });
        }
    }
    out
}

/// `isFinalSigma`: the capital sigma at `index` of the run `run` follows
/// a cased character and no cased character follows it, case ignorable
/// characters skipped on both sides and the search kept to the run.
fn final_sigma(run: &[Unit], index: usize) -> bool {
    let follows_cased = run[..index].iter().rev().find(|u| !case_ignorable(u.c)).is_some_and(|u| cased(u.c));
    follows_cased && !run[index + 1..].iter().find(|u| !case_ignorable(u.c)).is_some_and(|u| cased(u.c))
}

/// `lowerCaseRun`: `Character.toLowerCase` of each code point, final sigma
/// where the sigma is word final.
fn lowercase_run(run: &[Unit], used: &mut Used, sigmas: &mut Vec<u32>) -> Vec<Unit> {
    run.iter()
        .enumerate()
        .map(|(i, u)| {
            let c = if u.c == '\u{03A3}' && final_sigma(run, i) {
                used.sigma = true;
                sigmas.push(u.start);
                '\u{03C2}'
            } else {
                to_lower_char(u.c)
            };
            Unit { c, ..*u }
        })
        .collect()
}

fn nfd(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    normalize(Form::Nfd, text, &mut out);
    out
}

/// `appendWithoutAccents`: every nonspacing mark is left out, and its end
/// given to the character before it in the same run.
fn append_without_accents(units: &[Unit], out: &mut Vec<Unit>, used: &mut Used) {
    let base = out.len();
    for u in units {
        if general_category(u.c) == GeneralCategory::Mn {
            if out.len() > base {
                used.stripped = true;
                let last = out.len() - 1;
                out[last].end = u.end;
            }
        } else {
            out.push(*u);
        }
    }
}

/// `transformRun`: lowercase, then NFD and accent stripping, with each
/// decomposed character keeping its source character's span unless the
/// run's NFD is not its characters' NFDs in turn, when every character
/// spans the run.
fn transform_run(run: &[Unit], out: &mut Vec<Unit>, used: &mut Used, sigmas: &mut Vec<u32>) {
    let lower = lowercase_run(run, used, sigmas);
    let text: String = lower.iter().map(|u| u.c).collect();
    if is_normalized(Form::Nfd, &text) {
        append_without_accents(&lower, out, used);
        return;
    }
    let mut by_code_point = Vec::with_capacity(lower.len());
    for u in &lower {
        for c in nfd(u.c.encode_utf8(&mut [0; 4])).chars() {
            by_code_point.push(Unit { c, ..*u });
        }
    }
    let whole = nfd(&text);
    if whole.chars().eq(by_code_point.iter().map(|u| u.c)) {
        append_without_accents(&by_code_point, out, used);
    } else {
        used.reordered = true;
        let (start, end) = (run[0].start, run[run.len() - 1].end);
        out.extend(whole.chars().filter(|&c| general_category(c) != GeneralCategory::Mn).map(|c| Unit {
            c,
            start,
            end,
        }));
    }
}

/// `lowerCaseAndStripAccents`: every run between spaces transformed.
fn lowercase_and_strip_accents(units: &[Unit], used: &mut Used, sigmas: &mut Vec<u32>) -> Vec<Unit> {
    let mut out = Vec::with_capacity(units.len());
    let mut from = 0;
    while from < units.len() {
        if units[from].c == ' ' {
            out.push(units[from]);
            from += 1;
            continue;
        }
        let to = units[from..].iter().position(|u| u.c == ' ').map_or(units.len(), |i| from + i);
        transform_run(&units[from..to], &mut out, used, sigmas);
        from = to;
    }
    out
}

/// `isolatePunctuation`: an empty space on either side of every
/// punctuation character.
fn isolate_punctuation(units: &[Unit]) -> Vec<Unit> {
    let mut out = Vec::with_capacity(units.len());
    for u in units {
        if is_punctuation(u.c) {
            out.push(Unit { c: ' ', start: u.start, end: u.start });
            out.push(*u);
            out.push(Unit { c: ' ', start: u.end, end: u.end });
        } else {
            out.push(*u);
        }
    }
    out
}

/// `WordpieceEncoder`'s mapped text for `text`.
struct Mapped {
    units: Vec<Unit>,
    /// `text` with every capital sigma the JVM lowercased to final sigma
    /// written as final sigma, the same length.
    as_jvm: String,
    used: Used,
}

fn map(text: &str) -> Mapped {
    let mut used = Used::default();
    let mut sigmas = Vec::new();
    let units = clean_and_isolate_cjk(text);
    let units = lowercase_and_strip_accents(&units, &mut used, &mut sigmas);
    let units = isolate_punctuation(&units);
    let mut as_jvm = text.to_string();
    for at in sigmas {
        let at = at as usize;
        // Both sigmas are two bytes, so the text keeps its offsets.
        as_jvm.replace_range(at..at + '\u{03A3}'.len_utf8(), "\u{03C2}");
    }
    Mapped { units, as_jvm, used }
}

impl WordPiece {
    /// The spans `WordpieceEncoder` gives pieces with the ids `ids` over
    /// the mapped text, word by word: a word over the length limit, or
    /// one the port made unknown, is one piece over the word; otherwise
    /// each piece covers as many code points as its vocabulary entry has.
    /// None when the ids do not fit the words.
    fn jvm_spans(&self, units: &[Unit], ids: &[u32]) -> Option<Vec<[u32; 2]>> {
        let mut spans = Vec::with_capacity(ids.len());
        let mut k = 0;
        let mut from = 0;
        while from < units.len() {
            if units[from].c == ' ' {
                from += 1;
                continue;
            }
            let to = units[from..].iter().position(|u| u.c == ' ').map_or(units.len(), |i| from + i);
            let word = &units[from..to];
            from = to;
            let &next = ids.get(k)?;
            if word.len() > MAX_WORD_CODE_POINTS || next == self.unknown {
                if word.len() > MAX_WORD_CODE_POINTS && next != self.unknown {
                    return None;
                }
                spans.push([word[0].start, word[word.len() - 1].end]);
                k += 1;
                continue;
            }
            let mut at = 0;
            while at < word.len() {
                let len = *self.lengths.get(*ids.get(k)? as usize)?;
                if len == 0 || at + len > word.len() {
                    return None;
                }
                spans.push([word[at].start, word[at + len - 1].end]);
                at += len;
                k += 1;
            }
        }
        (k == ids.len()).then_some(spans)
    }

    /// Which documented difference of `analysis/embed/tests/goldens.rs`
    /// explains `jvm` against `rust`, if one does.
    fn documented(&self, text: &str, jvm: &[Piece], rust: &[Piece]) -> Option<String> {
        if SPECIAL_TOKENS.iter().any(|t| text.contains(t)) {
            return Some("a special token's string is that token here and ordinary text on the JVM, as upstream matches added tokens before normalizing (analysis/embed/tests/goldens.rs, special_token_strings_are_special_tokens_here_only)".into());
        }
        let mapped = map(text);
        // With final sigma written as the JVM lowercases it, the port's ids
        // are the JVM's.
        let final_sigma_pieces;
        let mut rust = rust;
        if mapped.as_jvm != text {
            final_sigma_pieces = analyze(&self.analyzer, &mapped.as_jvm).map(|d| rust_pieces(&d)).unwrap_or_default();
            rust = &final_sigma_pieces;
        }
        let ids: Vec<u32> = rust.iter().map(|p| p.0).collect();
        let jvm_ids: Vec<u32> = jvm.iter().map(|p| p.0).collect();
        let cjk = text.chars().any(jvm_only_cjk);
        if ids != jvm_ids {
            // The ideographs the JVM isolates and the port does not: each is
            // a word of its own, an unknown piece, where the port has one
            // unknown piece over the whole word.
            if !cjk {
                return None;
            }
            let mut split = Vec::new();
            for &(id, [a, b]) in rust.iter() {
                let piece = &text[a as usize..b as usize];
                if id == self.unknown && piece.chars().any(jvm_only_cjk) && piece.chars().count() > 1 {
                    let mut at = a;
                    for c in piece.chars() {
                        split.push((id, [at, at + c.len_utf8() as u32]));
                        at += c.len_utf8() as u32;
                    }
                } else {
                    split.push((id, [a, b]));
                }
            }
            return (split == jvm).then(|| "the ideographs U+2B820 to U+2B91F are CJK on the JVM and not upstream (analysis/embed/tests/goldens.rs, the_ideographs_from_u2b820_are_words_on_the_jvm_only)".to_string());
        }
        let spans = self.jvm_spans(&mapped.units, &ids)?;
        if spans.iter().zip(jvm).any(|(s, j)| *s != j.1) {
            return None;
        }
        let mut why = Vec::new();
        if mapped.used.sigma {
            why.push("a word final capital sigma lowercases to final sigma on the JVM and to sigma here (a_word_final_capital_sigma_is_final_sigma_on_the_jvm_only)");
        }
        if mapped.used.stripped {
            why.push("WordpieceEncoder runs a piece's span over the nonspacing marks accent stripping removed after it (a_stripped_mark_is_in_the_jvm_span_and_not_in_ours)");
        }
        if mapped.used.reordered {
            why.push("marks out of canonical order: WordpieceEncoder gives every piece of the run the whole run's span (marks_out_of_order_give_every_jvm_piece_the_whole_run)");
        }
        if cjk {
            why.push("the ideographs U+2B820 to U+2B91F are CJK on the JVM and not upstream (the_ideographs_from_u2b820_are_words_on_the_jvm_only)");
        }
        // The model gives the JVM's spans without any documented rule: a
        // difference nobody has written down yet.
        (!why.is_empty()).then(|| format!("{} (analysis/embed/tests/goldens.rs)", why.join("; ")))
    }
}

impl Run for WordPiece {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    /// Exactly, or as one of the differences `analysis/embed/tests/goldens.rs`
    /// documents and tests: a piece's span run over stripped marks or over a
    /// run with marks out of order, final sigma, the CJK range, and special
    /// token strings. The port follows upstream `tokenizers` in each.
    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        let mismatch = match matches_golden::<Subwords>(doc, golden, Tolerance::exact()) {
            Ok(()) => return Outcome::Match,
            Err(m) => m,
        };
        let Some(jvm) = jvm_pieces(golden) else { return Outcome::Divergence(mismatch.to_string()) };
        match self.documented(doc.source(), &jvm, &rust_pieces(doc)) {
            Some(why) => Outcome::Documented(why),
            None => Outcome::Divergence(mismatch.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(units: &[Unit]) -> String {
        units.iter().map(|u| u.c).collect()
    }

    fn spans_of(units: &[Unit]) -> Vec<[u32; 2]> {
        units.iter().map(|u| [u.start, u.end]).collect()
    }

    #[test]
    fn final_sigma_follows_the_jvm_rule_within_a_run() {
        assert_eq!(map("ΟΔΥΣΣΕΥΣ ΣΟΦΟΣ ΣΑΣ Σ ΑΣ. λόγος ΛΌΓΟΣ\n").as_jvm, "ΟΔΥΣΣΕΥς ΣΟΦΟς ΣΑς Σ Ας. λόγος ΛΌΓΟς\n");
        // A mark after the sigma is skipped; a letter after it is not, and
        // the search stops at the run's end, so a word after a space does
        // not count.
        assert_eq!(map("ΟΣ\u{0E48} ΟΣa ΟΣ b").as_jvm, "Ος\u{0E48} ΟΣa Ος b");
        assert!(map("ΟΣ b").used.sigma);
        assert!(!map("σ Σ").used.sigma);
    }

    #[test]
    fn whitespace_is_a_space_and_controls_are_dropped() {
        // The no-break space is whitespace; the vertical tab is a control
        // and gone, so the characters around it are one word.
        let m = map("Cafe\u{301} \u{a0}x\u{b}y");
        assert_eq!(text_of(&m.units), "cafe  xy");
        assert_eq!(spans_of(&m.units), [[0, 1], [1, 2], [2, 3], [3, 6], [6, 7], [7, 9], [9, 10], [11, 12]]);
        assert!(m.used.stripped && !m.used.reordered);
        // A mark at the start of a run is dropped without a span.
        assert_eq!(spans_of(&map("\u{301}a").units), [[2, 3]]);
    }

    #[test]
    fn ideographs_and_punctuation_are_isolated() {
        let m = map("a.b生x");
        assert_eq!(text_of(&m.units), "a . b 生 x");
        assert_eq!(spans_of(&m.units), [[0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 6], [6, 6], [6, 7]]);
    }

    #[test]
    fn reordered_marks_give_the_run_to_every_character() {
        let m = map("Vie\u{302}\u{323}t x");
        assert_eq!(text_of(&m.units), "viet x");
        assert_eq!(spans_of(&m.units), [[0, 8], [0, 8], [0, 8], [0, 8], [8, 9], [9, 10]]);
        assert!(m.used.reordered);
    }
}
