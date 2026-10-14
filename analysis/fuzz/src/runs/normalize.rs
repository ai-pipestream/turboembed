//! The normalizer runs: each normalizer the generator runs on the `norm`
//! set, under the run name its golden directory has, with the policy of
//! `analysis/normalize/tests/goldens.rs`: a golden made by an aligned
//! method compares exactly; one reconstructed by a diff compares on the
//! text and on the regions the edits change; a normalizer whose section of
//! `docs/normalizers.md` records differences from OpenNLP gives a
//! documented difference rather than a divergence.

use turbo_analysis::{Alignment, Document, Key, Provenance, View};
use turbo_goldens::expect::{ViewEdit, view_edits_equivalent};
use turbo_goldens::{Golden, Mismatch, Tolerance, matches_golden_key};
use turbo_normalize::{
    AccentFold, BoundedUrls, Bullets, CaseFold, Chain, ConfusableSkeleton, Dashes, Dehyphenate, Digits, Ellipsis,
    EmojiRemove, EmojiToEmoticon, EmoticonToEmoji, EnglishContractions, GermanUmlauts, Invisible, Normalizer, Numbers,
    Quotes, Shrink, SocialMedia, SymbolWords, UnicodeForm, Urls, Whitespace, normalize,
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

/// The run names and their normalizers, the edit policy of each golden
/// and whether `docs/normalizers.md` records differences from OpenNLP for
/// it (its "Differences from OpenNLP" subsection, which the golden test
/// lists input by input).
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

fn table(name: &str) -> Option<(Normalize, Edits, bool)> {
    use Edits::{Reconstructed, Reported};
    Some(match name {
        "whitespace" => (boxed(Whitespace::collapse()), Reported, false),
        "whitespace-keep-lines" => (boxed(Whitespace::keep_line_breaks()), Reported, false),
        "whitespace-keep-paragraphs" => (boxed(Whitespace::keep_paragraphs()), Reported, false),
        "dash" => (boxed(Dashes::new()), Reported, false),
        "quote" => (boxed(Quotes), Reported, true),
        "bullet" => (boxed(Bullets), Reported, true),
        "ellipsis" => (boxed(Ellipsis), Reported, true),
        "digit" => (boxed(Digits), Reported, false),
        "invisible" => (boxed(Invisible), Reported, true),
        "dehyphenation" => (boxed(Dehyphenate::new()), Reported, true),
        "nfc" => (boxed(UnicodeForm::nfc()), Reconstructed, false),
        "nfkc" => (boxed(UnicodeForm::nfkc()), Reconstructed, false),
        "case-fold" => (boxed(CaseFold::lowercase()), Reconstructed, false),
        "full-case-fold" => (boxed(CaseFold::full()), Reported, false),
        "accent-fold" => (boxed(AccentFold::new()), Reconstructed, false),
        "german-umlaut" => (boxed(GermanUmlauts::new()), Reported, false),
        "confusable-skeleton" => (boxed(ConfusableSkeleton), Reconstructed, true),
        "url" => (boxed(Urls), Reconstructed, false),
        "bounded-url" => (boxed(BoundedUrls), Reconstructed, true),
        "social-media" => (boxed(SocialMedia), Reconstructed, false),
        "number" => (boxed(Numbers), Reconstructed, false),
        "shrink" => (boxed(Shrink), Reconstructed, false),
        "english-contraction" => (boxed(EnglishContractions), Reported, true),
        "emoji" => (boxed(EmojiRemove::supplementary()), Reconstructed, true),
        "emoji-to-emoticon" => (boxed(EmojiToEmoticon), Reported, false),
        "emoticon-to-emoji" => (boxed(EmoticonToEmoji), Reported, false),
        "symbol-joiner" => (boxed(SymbolWords), Reconstructed, true),
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
            true,
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
    documented: bool,
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

impl Run for NormalizerRun {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        let mut doc = Document::new(text);
        let (out, alignment) = (self.normalize)(text);
        let view = View::derived(doc.get::<View>().expect("the source view"), out, alignment);
        doc.add(&NORM, view, Provenance::by("turbo-fuzz"))
            .map_err(|e| RustError { message: e.to_string(), offset: None })?;
        Ok(doc)
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        let text = doc.source();
        let result = matches_golden_key(doc, &NORM, golden, Tolerance::exact());
        let mismatch = match result {
            Ok(()) => return Outcome::Match,
            Err(Mismatch::Item { .. }) if self.edits == Edits::Reconstructed => {
                let golden_rows = golden_edits(golden).unwrap_or_default();
                if view_edits_equivalent(text, &view_edits(doc), &golden_rows) {
                    return Outcome::Match;
                }
                "the edits change the source differently".to_string()
            }
            Err(m) => m.to_string(),
        };
        if self.documented {
            Outcome::Documented(format!(
                "{}: docs/normalizers.md records differences from OpenNLP for this normalizer; {mismatch}",
                self.name
            ))
        } else {
            Outcome::Divergence(mismatch)
        }
    }
}
