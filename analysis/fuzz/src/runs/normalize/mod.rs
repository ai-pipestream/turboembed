//! The normalizer runs: each normalizer the generator runs on the `norm`
//! set, under the run name its golden directory has, with the policy of
//! `analysis/normalize/tests/goldens.rs`: a golden made by an aligned
//! method compares exactly; one reconstructed by a diff compares on the
//! text and on the regions the edits change; a normalizer whose section of
//! `docs/normalizers.md` records differences from OpenNLP gives a
//! documented difference where every region the two sides change
//! differently is the documented construct, as a rule per normalizer
//! recognizes it, and a divergence anywhere else.

mod rules;

use turbo_analysis::{Alignment, Document, Key, Provenance, View};
use turbo_goldens::expect::{Disagreement, ViewEdit, view_edit_disagreements};
use turbo_goldens::{Golden, Tolerance, matches_golden_key};
use turbo_normalize::{
    AccentFold, BoundedUrls, Bullets, CaseFold, Chain, ConfusableSkeleton, Dashes, Dehyphenate, Digits, Ellipsis,
    EmojiRemove, EmojiToEmoticon, EmoticonToEmoji, EnglishContractions, GermanUmlauts, Invisible, Normalizer, Numbers,
    Quotes, Shrink, SocialMedia, SymbolWords, UnicodeForm, Urls, Whitespace, normalize,
};

use self::rules::{
    bounded_url_rule, bullet_rule, chain_rule, confusable_rule, contraction_rule, dehyphenation_rule, ellipsis_rule,
    invisible_rule, quote_rule, symbol_rule,
};
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
pub(crate) const RUN_NAMES: [&str; 30] = [
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
    "case-fold-tr",
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
    "umlaut-full-case-whitespace-chain",
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
        "case-fold-tr" => (boxed(CaseFold::turkish_lowercase()), Reconstructed, None),
        "full-case-fold" => (boxed(CaseFold::full()), Reported, None),
        "accent-fold" => (boxed(AccentFold::new()), Reconstructed, None),
        "german-umlaut" => (boxed(GermanUmlauts::new()), Reported, None),
        "umlaut-full-case-whitespace-chain" => (
            boxed(Chain::new().then(GermanUmlauts::new()).then(CaseFold::full()).then(Whitespace::collapse())),
            Reported,
            None,
        ),
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
mod tests;
