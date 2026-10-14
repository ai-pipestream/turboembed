//! The normalizer runs: each normalizer the generator runs on the `norm`
//! set, under the run name its golden directory has, with the policy of
//! `analysis/normalize/tests/goldens.rs`: a golden made by an aligned
//! method compares exactly; one reconstructed by a diff compares on the
//! text and on the regions the edits change; a normalizer whose section of
//! `docs/normalizers.md` records differences from OpenNLP gives a
//! documented difference where every region the two sides change
//! differently is the documented construct, as a rule per normalizer
//! recognizes it, and a divergence anywhere else.

use turbo_analysis::{Alignment, Document, Key, Provenance, View};
use turbo_goldens::expect::{Disagreement, ViewEdit, view_edit_disagreements};
use turbo_goldens::{Golden, Tolerance, matches_golden_key};
use turbo_normalize::tables::{is_horizontal_whitespace, is_line_break, is_whitespace};
use turbo_normalize::{
    AccentFold, BoundedUrls, Bullets, CaseFold, Chain, ConfusableSkeleton, Dashes, Dehyphenate, Digits, Ellipsis,
    EmojiRemove, EmojiToEmoticon, EmoticonToEmoji, EnglishContractions, GermanUmlauts, Invisible, Normalizer, Numbers,
    Quotes, Shrink, SocialMedia, SymbolWords, UnicodeForm, Urls, Whitespace, normalize,
};
use turbo_unicode::category::general_category;
use turbo_unicode::confusables::{prototype, skeleton};
use turbo_unicode::java::is_letter;
use turbo_unicode::normalize::{Form, normalize as unicode_normalize};

use super::confusables_17::CHANGES_17;
use super::{Outcome, Run, RustError};
use crate::resources::Unavailable;

/// The view the normalized text goes in, as the golden test names it.
const NORM: Key<View> = Key::named("norm");

/// Where a run's golden edits come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Edits {
    /// OpenNLP's normalizer reports an alignment; its edits are compared
    /// one for one.
    Reported,
    /// The generator rebuilt the edits by a diff of source and result;
    /// edit boundaries are the diff's choice.
    Reconstructed,
}

type Normalize = Box<dyn Fn(&str) -> (String, Alignment) + Send + Sync>;

fn boxed<N: Normalizer>(n: N) -> Normalize {
    Box::new(move |text| normalize(&n, text))
}

/// A documented difference from OpenNLP: the rule that recognizes it in
/// one region the two sides change differently, and the words of
/// `docs/normalizers.md` for it.
#[derive(Clone, Copy)]
struct Documented {
    what: &'static str,
    rule: fn(&str, &Disagreement) -> bool,
}

/// The run names and their normalizers, the edit policy of each golden
/// and the documented difference of `docs/normalizers.md` for it (its
/// "Differences from OpenNLP" subsection), if it has one.
pub(crate) const RUN_NAMES: [&str; 28] = [
    "whitespace",
    "whitespace-keep-lines",
    "whitespace-keep-paragraphs",
    "dash",
    "quote",
    "bullet",
    "ellipsis",
    "digit",
    "invisible",
    "dehyphenation",
    "nfc",
    "nfkc",
    "case-fold",
    "full-case-fold",
    "accent-fold",
    "german-umlaut",
    "confusable-skeleton",
    "url",
    "bounded-url",
    "social-media",
    "number",
    "shrink",
    "english-contraction",
    "emoji",
    "emoji-to-emoticon",
    "emoticon-to-emoji",
    "symbol-joiner",
    "default-chain",
];

fn table(name: &str) -> Option<(Normalize, Edits, Option<Documented>)> {
    use Edits::{Reconstructed, Reported};
    let documented = |what, rule| Some(Documented { what, rule });
    Some(match name {
        "whitespace" => (boxed(Whitespace::collapse()), Reported, None),
        "whitespace-keep-lines" => (boxed(Whitespace::keep_line_breaks()), Reported, None),
        "whitespace-keep-paragraphs" => (boxed(Whitespace::keep_paragraphs()), Reported, None),
        "dash" => (boxed(Dashes::new()), Reported, None),
        "quote" => {
            (boxed(Quotes), Reported, documented("U+2E42 is folded to U+0022 here and left by OpenNLP", quote_rule))
        }
        "bullet" => (
            boxed(Bullets),
            Reported,
            documented("U+204C, U+204D, U+25D8, U+29BE and U+29BF are bullets here and not in OpenNLP", bullet_rule),
        ),
        "ellipsis" => (
            boxed(Ellipsis),
            Reported,
            documented("U+2024, U+FE19 and U+FE30 expand to dots here and pass through OpenNLP", ellipsis_rule),
        ),
        "digit" => (boxed(Digits), Reported, None),
        "invisible" => (
            boxed(Invisible),
            Reported,
            documented("U+206A to U+206F are deleted here and kept by OpenNLP", invisible_rule),
        ),
        "dehyphenation" => (
            boxed(Dehyphenate::new()),
            Reported,
            documented(
                "a word that ends in combining marks joins across the break here and not in OpenNLP",
                dehyphenation_rule,
            ),
        ),
        "nfc" => (boxed(UnicodeForm::nfc()), Reconstructed, None),
        "nfkc" => (boxed(UnicodeForm::nfkc()), Reconstructed, None),
        "case-fold" => (boxed(CaseFold::lowercase()), Reconstructed, None),
        "full-case-fold" => (boxed(CaseFold::full()), Reported, None),
        "accent-fold" => (boxed(AccentFold::new()), Reconstructed, None),
        "german-umlaut" => (boxed(GermanUmlauts::new()), Reported, None),
        "confusable-skeleton" => (
            boxed(ConfusableSkeleton),
            Reconstructed,
            documented("the skeleton uses confusables.txt 18.0.0 here and 17.0.0 on the JVM", confusable_rule),
        ),
        "url" => (boxed(Urls), Reconstructed, None),
        "bounded-url" => (
            boxed(BoundedUrls),
            Reconstructed,
            documented("an internationalized e-mail address is one space here and text on the JVM", bounded_url_rule),
        ),
        "social-media" => (boxed(SocialMedia), Reconstructed, None),
        "number" => (boxed(Numbers), Reconstructed, None),
        "shrink" => (boxed(Shrink), Reconstructed, None),
        "english-contraction" => (
            boxed(EnglishContractions),
            Reported,
            documented("y'all and the longer subject lists for 're, 've and 'll expand here only", contraction_rule),
        ),
        // The supplementary mode is OpenNLP's loop; the differences the
        // docs record are the sequence mode and the annotator.
        "emoji" => (boxed(EmojiRemove::supplementary()), Reconstructed, None),
        "emoji-to-emoticon" => (boxed(EmojiToEmoticon), Reported, None),
        "emoticon-to-emoji" => (boxed(EmoticonToEmoji), Reported, None),
        "symbol-joiner" => (
            boxed(SymbolWords),
            Reconstructed,
            documented("a symbol between whitespace in running text becomes its word here only", symbol_rule),
        ),
        "default-chain" => (
            boxed(
                Chain::new()
                    .then(Invisible)
                    .then(UnicodeForm::nfc())
                    .then(Whitespace::collapse())
                    .then(Quotes)
                    .then(Dashes::new())
                    .then(CaseFold::lowercase())
                    .then(AccentFold::new()),
            ),
            Reconstructed,
            documented(
                "the chain's Invisible deletes U+206A to U+206F and its Quotes folds U+2E42, which OpenNLP's leave",
                chain_rule,
            ),
        ),
        // `case-fold-tr` is OpenNLP's Turkish locale lowercasing, which the
        // port has no counterpart of; its golden test documents the texts
        // where the Turkic simple fold differs, and the differential
        // leaves it out.
        _ => return None,
    })
}

/// `golden` with each character `f` maps written as its replacement.
fn fold(golden: &str, f: impl Fn(char) -> Option<&'static str>) -> String {
    let mut out = String::with_capacity(golden.len());
    for c in golden.chars() {
        match f(c) {
            Some(with) => out.push_str(with),
            None => out.push(c),
        }
    }
    out
}

/// The port's result is the JVM's with the addition `f` applied to the
/// characters the source holds; the source must hold at least one.
fn addition(text: &str, d: &Disagreement, f: impl Fn(char) -> Option<&'static str>) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    d.source(text).chars().any(|c| f(c).is_some()) && rust == fold(golden, &f)
}

fn quote_rule(text: &str, d: &Disagreement) -> bool {
    addition(text, d, |c| (c == '\u{2E42}').then_some("\""))
}

fn bullet_rule(text: &str, d: &Disagreement) -> bool {
    addition(text, d, |c| matches!(c, '\u{204C}' | '\u{204D}' | '\u{25D8}' | '\u{29BE}' | '\u{29BF}').then_some(" "))
}

fn ellipsis_rule(text: &str, d: &Disagreement) -> bool {
    addition(text, d, |c| match c {
        '\u{2024}' => Some("."),
        '\u{FE19}' => Some("..."),
        '\u{FE30}' => Some(".."),
        _ => None,
    })
}

fn invisible_rule(text: &str, d: &Disagreement) -> bool {
    addition(text, d, |c| ('\u{206A}'..='\u{206F}').contains(&c).then_some(""))
}

fn chain_rule(text: &str, d: &Disagreement) -> bool {
    addition(text, d, |c| match c {
        '\u{206A}'..='\u{206F}' => Some(""),
        '\u{2E42}' => Some("\""),
        _ => None,
    })
}

/// The region is a hyphen, a line break and indentation that the port
/// deleted and the JVM kept, after a letter with one or more combining
/// marks and before a letter.
fn dehyphenation_rule(text: &str, d: &Disagreement) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    let source = d.source(text);
    if !rust.is_empty() || golden != source {
        return false;
    }
    let mut chars = source.chars();
    if !matches!(chars.next(), Some('-' | '\u{00AD}' | '\u{2010}')) {
        return false;
    }
    let rest = chars.as_str();
    let Some(first) = rest.chars().next().filter(|&c| is_line_break(c)) else { return false };
    let mut after = &rest[first.len_utf8()..];
    if first == '\r' {
        after = after.strip_prefix('\n').unwrap_or(after);
    }
    if !after.chars().all(is_horizontal_whitespace) {
        return false;
    }
    let mut before = text[..d.start].chars().rev().peekable();
    let mut marks = 0;
    while before.peek().is_some_and(|&c| general_category(c).is_mark()) {
        before.next();
        marks += 1;
    }
    marks > 0 && before.next().is_some_and(is_letter) && text[d.end..].chars().next().is_some_and(is_letter)
}

/// The word of ASCII letters and apostrophes around the region.
fn word_around(text: &str, start: usize, end: usize) -> (usize, usize) {
    let in_word = |c: char| c.is_ascii_alphabetic() || matches!(c, '\'' | '\u{2019}' | '\u{02BC}' | '\u{FF07}');
    let before = text[..start].chars().rev().take_while(|&c| in_word(c)).map(char::len_utf8).sum::<usize>();
    let after = text[end..].chars().take_while(|&c| in_word(c)).map(char::len_utf8).sum::<usize>();
    (start - before, end + after)
}

/// The region is inside a contraction the port expands and OpenNLP does
/// not: `y'all`, or one of the added subjects before 're, 've or 'll. The
/// JVM left the region alone and the port's text is its own expansion of
/// the word.
fn contraction_rule(text: &str, d: &Disagreement) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    if golden != d.source(text) {
        return false;
    }
    let (ws, we) = word_around(text, d.start, d.end);
    let word = &text[ws..we];
    let lower = word.to_ascii_lowercase();
    let apostrophe =
        |s: &str| s.chars().next().is_some_and(|c| matches!(c, '\'' | '\u{2019}' | '\u{02BC}' | '\u{FF07}'));
    let added = |stems: &[&str], suffix: &str| {
        lower.strip_suffix(suffix).is_some_and(|s| {
            let (stem, apo) = s.split_at(s.len() - s.chars().last().map_or(0, char::len_utf8));
            apostrophe(apo) && stems.contains(&stem)
        })
    };
    let expands_here_only =
        lower.len() == 5 && lower.starts_with('y') && lower.ends_with("all") && apostrophe(&lower[1..])
            || added(&["here", "these", "those"], "re")
            || added(&["may", "there", "these", "those", "what", "where"], "ve")
            || added(&["here", "this", "these", "those", "which"], "ll");
    if !expands_here_only {
        return false;
    }
    let (expanded, _) = normalize(&EnglishContractions, word);
    expanded == format!("{}{rust}{}", &text[ws..d.start], &text[d.end..we])
}

/// `SymbolWords`' ten symbols and their words.
fn symbol_word(c: char) -> Option<&'static str> {
    Some(match c {
        '&' => "and",
        '+' => "plus",
        '@' => "at",
        '%' => "percent",
        '\u{00A7}' => "section",
        '\u{00B6}' => "paragraph",
        '\u{00B0}' => "degree",
        '\u{00A9}' => "copyright",
        '\u{00AE}' => "registered",
        '\u{2122}' => "trademark",
        _ => return None,
    })
}

/// The region is one symbol with whitespace or the text's edge on both
/// sides, not the whole text (which OpenNLP maps too), left by the JVM
/// and written as its word here.
fn symbol_rule(text: &str, d: &Disagreement) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    let source = d.source(text);
    let mut chars = source.chars();
    let (Some(c), None) = (chars.next(), chars.next()) else { return false };
    let Some(word) = symbol_word(c) else { return false };
    let bounded = text[..d.start].chars().next_back().is_none_or(is_whitespace)
        && text[d.end..].chars().next().is_none_or(is_whitespace);
    bounded && source != text && golden == source && rust == word
}

/// The region holds an e-mail address with a non-ASCII letter, digit or
/// mark, which `BoundedUrls` takes as part of the address and OpenNLP's
/// ASCII scan does not. Where the two results differ, the JVM kept the
/// address (whole, or what is left of it on either side of the ASCII
/// part it replaced by a space) and the port wrote nothing or one space
/// in its place: what the JVM kept is one or two stretches without
/// whitespace, around at most one space, holding a non-ASCII letter,
/// digit or mark.
///
/// What the JVM kept is the source's own text, in order, and each kept
/// stretch is in a whitespace-delimited token of the source that holds an
/// `@`: the address the port matched (the region can hold other tokens
/// too, a URL both sides replaced next to the address among them).
///
/// What this does not check: that the port's match is the address the
/// port's `BoundedUrls` should take, or the one OpenNLP's scan would take
/// up to its first non-ASCII character. The rule sees the region's source
/// and the two results, and it cannot run OpenNLP's scan; whether the
/// port's scan takes the right stretch is for `turbo-normalize`'s own
/// tests, and the JVM's ASCII stretch is whatever it replaced. A port
/// match that took a wrong whitespace-free stretch with an `@` and a
/// non-ASCII letter would pass here as the documented difference.
fn bounded_url_rule(text: &str, d: &Disagreement) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    let source = d.source(text);
    if !source.contains('@') {
        return false;
    }
    let prefix = rust.char_indices().zip(golden.chars()).take_while(|((_, a), b)| a == b).count();
    let prefix = rust.char_indices().nth(prefix).map_or(rust.len(), |(i, _)| i);
    let (r_rest, g_rest) = (&rust[prefix..], &golden[prefix..]);
    let suffix: usize =
        r_rest.chars().rev().zip(g_rest.chars().rev()).take_while(|(a, b)| a == b).map(|(a, _)| a.len_utf8()).sum();
    let (removed, kept) = (&r_rest[..r_rest.len() - suffix], &g_rest[..g_rest.len() - suffix]);
    let international = |c: char| !c.is_ascii() && (c.is_alphanumeric() || general_category(c).is_mark());
    let parts: Vec<&str> = kept.split(' ').collect();
    // Each kept stretch is in the source, after the one before it, in a
    // token with an `@`.
    let mut from = 0;
    let in_source = parts.iter().all(|part| match source[from..].find(part) {
        Some(i) => {
            let at = from + i;
            let token_start = source[..at]
                .rfind(char::is_whitespace)
                .map_or(0, |w| w + source[w..].chars().next().map_or(1, char::len_utf8));
            let token_end = source[at..].find(char::is_whitespace).map_or(source.len(), |w| at + w);
            from = at + part.len();
            source[token_start..token_end].contains('@')
        }
        None => false,
    });
    (removed.is_empty() || removed == " ")
        && parts.len() <= 2
        && parts.iter().all(|part| !part.is_empty() && !part.chars().any(char::is_whitespace))
        && in_source
        && kept.chars().any(international)
}

/// The skeleton of `text` by confusables.txt 17.0.0, the data OpenNLP
/// bundles: NFD, each character replaced by its 17.0.0 prototype
/// (`confusables_17::CHANGES_17` where that version differs from 18.0.0,
/// `turbo_unicode`'s prototype otherwise), NFD again.
fn skeleton_17(text: &str) -> String {
    let mut decomposed = String::with_capacity(text.len());
    unicode_normalize(Form::Nfd, text, &mut decomposed);
    let mut mapped = String::with_capacity(decomposed.len());
    for c in decomposed.chars() {
        let changed = CHANGES_17.binary_search_by_key(&(c as u32), |&(cp, _)| cp).ok().map(|i| CHANGES_17[i].1);
        match changed.unwrap_or_else(|| prototype(c)) {
            Some(target) => mapped.push_str(target),
            None => mapped.push(c),
        }
    }
    let mut out = String::with_capacity(mapped.len());
    unicode_normalize(Form::Nfd, &mapped, &mut out);
    out
}

/// The port's text is the Unicode 18.0 skeleton of the source and the
/// JVM's its 17.0 skeleton: the two data versions, nothing else.
fn confusable_rule(text: &str, d: &Disagreement) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    let mut of_source = String::new();
    skeleton(d.source(text), &mut of_source);
    rust == of_source && golden == skeleton_17(d.source(text))
}

pub(crate) struct NormalizerRun {
    name: String,
    normalize: Normalize,
    edits: Edits,
    documented: Option<Documented>,
}

impl NormalizerRun {
    pub(crate) fn open(name: &str) -> Result<Box<dyn Run>, Unavailable> {
        let (normalize, edits, documented) =
            table(name).ok_or_else(|| Unavailable::NotCovered(format!("normalize/{name}")))?;
        Ok(Box::new(NormalizerRun { name: name.to_string(), normalize, edits, documented }))
    }
}

/// The edits of the view as golden rows: source span and replacement.
fn view_edits(doc: &Document) -> Vec<ViewEdit> {
    let Some(view) = doc.get_key(&NORM) else { return Vec::new() };
    let out = view.text();
    view.alignment()
        .edits()
        .iter()
        .map(|e| ViewEdit {
            start: e.parent.start as usize,
            end: e.parent.end as usize,
            with: out[e.child.range()].into(),
        })
        .collect()
}

/// The golden's `turbo:view` rows.
fn golden_edits(golden: &Golden) -> Option<Vec<ViewEdit>> {
    let layer = golden.layer("turbo:view")?;
    let (start, end, with) = (layer.u32s("start")?, layer.u32s("end")?, layer.strs("with")?);
    Some(
        (0..start.len())
            .map(|i| ViewEdit { start: start[i] as usize, end: end[i] as usize, with: with.get(i).to_string() })
            .collect(),
    )
}

/// `rows` with the deletions that touch joined into one, as the edits the
/// generator reads off OpenNLP's alignment have them: that alignment
/// lists the source range of each output character and nothing for a
/// deleted one, so two deleted neighbours are one gap.
fn joined_deletions(rows: &[ViewEdit]) -> Vec<ViewEdit> {
    let mut out: Vec<ViewEdit> = Vec::with_capacity(rows.len());
    for r in rows {
        match out.last_mut() {
            Some(last) if last.with.is_empty() && r.with.is_empty() && last.end == r.start => last.end = r.end,
            _ => out.push(r.clone()),
        }
    }
    out
}

fn describe(text: &str, d: &Disagreement) -> String {
    format!(
        "bytes {}..{} {:?} give {:?} here and {:?} on the JVM",
        d.start,
        d.end,
        d.source(text),
        d.rust.as_deref().unwrap_or_else(|e| e),
        d.golden.as_deref().unwrap_or_else(|e| e)
    )
}

impl Run for NormalizerRun {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        let mut doc = Document::new(text);
        let (out, alignment) = (self.normalize)(text);
        let view = View::derived(doc.get::<View>().expect("the source view"), out, alignment);
        doc.add(&NORM, view, Provenance::by("turbo-fuzz"))
            .map_err(|e| RustError { message: e.to_string(), offset: None })?;
        Ok(doc)
    }

    /// Exactly; or, for reconstructed edits, on the regions the edits
    /// change; or as the documented difference, when every region the two
    /// sides change differently passes the normalizer's rule.
    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        let mismatch = match matches_golden_key(doc, &NORM, golden, Tolerance::exact()) {
            Ok(()) => return Outcome::Match,
            Err(m) => m.to_string(),
        };
        self.explain(doc, &golden_edits(golden).unwrap_or_default(), mismatch)
    }
}

impl NormalizerRun {
    /// What a mismatch of the view against the JVM's edits `golden_rows`
    /// is: the same text from the same regions is a match for reconstructed
    /// edits, every disagreeing region passing the documented rule is the
    /// documented difference, anything else a divergence.
    fn explain(&self, doc: &Document, golden_rows: &[ViewEdit], mismatch: String) -> Outcome {
        let text = doc.source();
        let rust_rows = view_edits(doc);
        let disagreements = view_edit_disagreements(text, &rust_rows, golden_rows);
        if disagreements.is_empty() {
            // The same text from the same regions: the diff's granularity
            // for reconstructed edits. Reported edits must be the same
            // rows, but for the deletions OpenNLP's alignment cannot keep
            // apart.
            return if self.edits == Edits::Reconstructed {
                Outcome::Match
            } else if *golden_rows == joined_deletions(&rust_rows) {
                Outcome::Documented(format!(
                    "{}: OpenNLP's alignment lists the source range of each output character, so neighbouring deletions are one gap in it where the port records each deletion; the text and every mapped span agree (docs/normalizers.md, Alignment checks); {mismatch}",
                    self.name
                ))
            } else {
                Outcome::Divergence(mismatch)
            };
        }
        let passes = |d: &Disagreement| self.documented.is_some_and(|r| (r.rule)(text, d));
        match disagreements.iter().find(|d| !passes(d)) {
            None => {
                let what = self.documented.map_or("", |r| r.what);
                Outcome::Documented(format!(
                    "{}: {what} (docs/normalizers.md, Differences from OpenNLP); {mismatch}",
                    self.name
                ))
            }
            Some(other) => Outcome::Divergence(format!("{mismatch}; {}", describe(text, other))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(start: usize, end: usize, with: &str) -> ViewEdit {
        ViewEdit { start, end, with: with.into() }
    }

    /// The outcome of the run `name` on `text` against the JVM edits
    /// `golden`, the exact comparison taken as failed.
    fn outcome(name: &str, text: &str, golden: &[ViewEdit]) -> Outcome {
        let (normalize, edits, documented) = table(name).unwrap();
        let run = NormalizerRun { name: name.into(), normalize, edits, documented };
        let doc = run.annotate(text, None).unwrap();
        run.explain(&doc, golden, "differs".into())
    }

    fn documented(name: &str, text: &str, golden: &[ViewEdit]) {
        let out = outcome(name, text, golden);
        assert!(matches!(out, Outcome::Documented(_)), "{name} on {text:?}: {out}");
    }

    fn divergence(name: &str, text: &str, golden: &[ViewEdit]) {
        let out = outcome(name, text, golden);
        assert!(matches!(out, Outcome::Divergence(_)), "{name} on {text:?}: {out}");
    }

    /// Explains one golden of a fuzz run: `TURBO_FUZZ_GOLDEN` names the
    /// golden file, `TURBO_FUZZ_INPUT` its input text and `TURBO_FUZZ_RUN`
    /// the normalizer run.
    #[test]
    #[ignore = "a diagnostic over files named by the environment"]
    fn explain_one_golden_from_the_environment() {
        let name = std::env::var("TURBO_FUZZ_RUN").unwrap();
        let (normalize, edits, documented) = table(&name).unwrap();
        let run = NormalizerRun { name: name.clone(), normalize, edits, documented };
        let golden = crate::read_golden(std::path::Path::new(&std::env::var("TURBO_FUZZ_GOLDEN").unwrap())).unwrap();
        let text = std::fs::read_to_string(std::env::var("TURBO_FUZZ_INPUT").unwrap()).unwrap();
        let doc = run.annotate(&text, Some(&golden)).unwrap();
        eprintln!("port: {:?}", view_edits(&doc));
        eprintln!("jvm:  {:?}", golden_edits(&golden));
        eprintln!("{}", run.compare(&doc, &golden));
    }

    #[test]
    fn the_character_additions_are_documented_and_other_characters_are_not() {
        // U+2E42 folded here only; the JVM folds U+201C as the port does.
        documented("quote", "\u{2E42}Wort\u{201C}", &[e(7, 10, "\"")]);
        divergence("quote", "\u{2E42}Wort\u{201C}", &[e(7, 10, "'")]);
        divergence("quote", "\u{2E42}Wort\u{201C}", &[]);
        documented("bullet", "\u{204C} a \u{2022} b", &[e(6, 9, " ")]);
        divergence("bullet", "\u{204C} a \u{2022} b", &[]);
        documented("ellipsis", "a\u{2024}b\u{2026}", &[e(5, 8, "...")]);
        divergence("ellipsis", "a\u{2024}b\u{2026}", &[e(5, 8, "..")]);
        documented("invisible", "a\u{206A}b\u{200B}", &[e(5, 8, "")]);
        divergence("invisible", "a\u{206A}b\u{200B}", &[]);
        documented("default-chain", "\u{2E42}A\u{206A}", &[e(3, 4, "a")]);
        divergence("default-chain", "\u{2E42}A\u{206A}", &[e(3, 4, "A")]);
    }

    #[test]
    fn a_join_after_combining_marks_is_documented_and_a_missing_join_is_not() {
        let text = "cafe\u{301}-\n  teria and well-\nknown";
        let well = text.find("well-").unwrap() + 4;
        documented("dehyphenation", text, &[e(well, well + 2, "")]);
        divergence("dehyphenation", text, &[]);
        // A letter right before the hyphen: both join, so a golden without
        // the join is a divergence, and so is a join of another kind.
        divergence("dehyphenation", "cafe-\n  teria", &[]);
        divergence("dehyphenation", text, &[e(well, well + 2, " ")]);
    }

    #[test]
    fn the_added_contractions_are_documented_and_the_shared_ones_are_not() {
        documented("english-contraction", "here're y'all", &[]);
        documented("english-contraction", "THESE\u{2019}LL do", &[]);
        divergence("english-contraction", "they're here", &[]);
        divergence("english-contraction", "here're y'all", &[e(0, 7, "we are")]);
    }

    #[test]
    fn a_symbol_in_running_text_is_documented_and_a_lone_one_is_not() {
        documented("symbol-joiner", "R&D & co \u{00A9} x", &[]);
        divergence("symbol-joiner", "&", &[]);
        divergence("symbol-joiner", "a & b", &[e(2, 3, "et")]);
    }

    #[test]
    fn an_international_address_is_documented_and_an_ascii_one_is_not() {
        documented("bounded-url", "mail Jos\u{E9}@example.com now", &[]);
        documented("bounded-url", "mail \u{E9}a@b.co now", &[e(7, 13, " ")]);
        divergence("bounded-url", "mail a@b.co now", &[]);
        divergence("bounded-url", "mail Jos\u{E9}@example.com now", &[e(5, 9, "Jos ")]);
        // Text the JVM kept that is not the source's, or out of its order.
        divergence("bounded-url", "mail Jos\u{E9}@example.com now", &[e(5, 8, "Zos")]);
        divergence("bounded-url", "see first\u{5B8}+tag@a.b.c\u{20DD} now", &[e(4, 24, "c\u{20DD} first\u{5B8}")]);
        // A URL both replace right before the address, one region for the
        // diff; and a kept stretch in a token without an `@`.
        let text = "needs https://s\u{E48}.d.ex/ x.y.z@sub.d\u{338}omain.example b";
        documented("bounded-url", text, &[e(6, 36, "   ")]);
        divergence("bounded-url", text, &[e(6, 36, "   "), e(49, 50, " \u{E9}")]);
        // The ASCII part replaced in the middle, the Greek kept by the JVM.
        let text = "x brien@example.com...\u{3AC}\u{3BD} y";
        documented("bounded-url", text, &[e(2, 22, " ")]);
        divergence("bounded-url", text, &[e(2, 22, "  ")]);
        // A URL both replace, then an address the JVM's scan leaves at a
        // combining mark inside the domain.
        let text = "a https://h.example/ x.y@sub.d\u{338}omain.example b";
        documented("bounded-url", text, &[e(2, 20, " "), e(21, 30, " ")]);
        divergence("bounded-url", text, &[e(2, 20, " "), e(21, 30, "  ")]);
        // Marks on both sides of the ASCII part: the JVM keeps both ends.
        let text = "see first\u{5B8}+tag@a.b.c\u{20DD} now";
        documented("bounded-url", text, &[e(11, 20, " ")]);
        divergence("bounded-url", text, &[e(11, 20, "  ")]);
        divergence("bounded-url", text, &[e(11, 23, " ")]);
    }

    #[test]
    fn neighbouring_deletions_the_jvm_lists_as_one_are_documented() {
        // Two invisible characters side by side: two edits here, one gap
        // in OpenNLP's alignment.
        documented("invisible", "a\u{200E}\u{FEFF}b", &[e(1, 7, "")]);
        divergence("invisible", "a\u{200E}\u{FEFF}b", &[e(1, 4, ""), e(4, 7, "x")]);
        assert_eq!(
            joined_deletions(&[e(1, 4, ""), e(4, 7, ""), e(8, 9, ""), e(9, 10, "y")]),
            [e(1, 7, ""), e(8, 9, ""), e(9, 10, "y")]
        );
    }

    #[test]
    fn the_jvms_skeleton_by_the_older_data_is_documented() {
        // 18.0 maps U+030B to two acute accents and U+3099 to U+FF9E; 17.0
        // left both. U+309A goes to U+030A in both.
        // NFD puts the kana marks (class 8) before the accent (class 230).
        assert_eq!(skeleton_17("o\u{30B}\u{3099}\u{309A}"), "o\u{3099}\u{30A}\u{30B}");
        documented("confusable-skeleton", "o\u{30B}", &[]);
        documented("confusable-skeleton", "a\u{3099}\u{309A}", &[e(4, 7, "\u{30A}")]);
        divergence("confusable-skeleton", "o\u{30B}", &[e(0, 3, "x")]);
        // A character both versions map the same way: the JVM must map it.
        documented("confusable-skeleton", "\u{43E}\u{30B}", &[e(0, 2, "o")]);
        divergence("confusable-skeleton", "\u{43E}\u{30B}", &[]);
        // The table is sorted for the binary search, and holds the
        // differences only.
        assert!(CHANGES_17.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(CHANGES_17.iter().all(|&(cp, t)| prototype(char::from_u32(cp).unwrap()) != t));
    }
}
