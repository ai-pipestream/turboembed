//! The tokenizer runs: the maxent tokenizer with every published or
//! fixture model and its options, the simple and whitespace tokenizers in
//! every mode, the UAX #29 word tokenizer, the unigram segmenter with
//! each lexicon, and the lattice tokenizer with each dictionary.

use turbo_analysis::{Analyzer, Annotator, Document, Key, Provenance, View};
use turbo_analysis_layers::{Confidence, Tokens};
use turbo_goldens::expect::{
    JvmSpan, joined_me_spans, jvm_simple_newline_spans, jvm_token_spans, simple_newline_spans, whole_character_spans,
};
use turbo_goldens::{Golden, Tolerance, matches_golden};
use turbo_maxent::Dictionary;
use turbo_tokenize::lattice::Scratch as LatticeScratch;
use turbo_tokenize::me::{TokenizerMe, TokenizerModel};
use turbo_tokenize::{
    LatticeTokenizer, MecabDictionary, SimpleTokenizer, TokenizerOptions, UnigramLexicon, UnigramSegmenter,
    WhitespaceTokenizer, WordTokenizer, WordTypes, simple_spans, whitespace_spans,
};
use turbo_unicode::utf16::byte_to_utf16;

use super::sentences::whitespace_mode;
use super::{Checked, Outcome, Run, RustError, analyze, has_cut, jvm_threw, outcome, token_spans};
use crate::resources::{Resources, Unavailable};

/// The published tokenizer model whose run with the English abbreviation
/// dictionary has goldens.
const ENGLISH_MODEL: &str = "opennlp-en-ud-ewt-tokens-1.3-2.5.4";

/// The probabilities the golden tests accept: `assert_close` in
/// `analysis/tokenize/tests/common/mod.rs` allows 1e-9.
fn confidence_tolerance() -> Tolerance {
    Tolerance::abs(1e-9)
}

fn broken(what: impl std::fmt::Display) -> Unavailable {
    Unavailable::Broken(what.to_string())
}

/// The JVM's token spans of `golden`, or why they do not read.
fn jvm_spans(golden: &Golden, text: &str) -> Result<Vec<JvmSpan>, Outcome> {
    jvm_token_spans(golden, text).map_err(|e| Outcome::Skipped(format!("the golden's tokens do not read: {e}")))
}

/// `TokenizerME` with a model: `<model>`, `<model>+alphanumeric` or
/// `<model>+english-dictionary`.
pub(crate) struct Maxent {
    analyzer: Analyzer,
}

impl Maxent {
    pub(crate) fn open(model: &str, options: &[&str], res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let path = if model == "tokenizer-abbreviations" {
            res.fixture(&format!("{model}.bin"))?
        } else {
            res.model(&format!("{model}.bin"))?
        };
        let mut m = TokenizerModel::read(&path).map_err(|e| broken(format!("{}: {e}", path.display())))?;
        match options {
            [] => {}
            ["alphanumeric"] => {
                let pattern = m.alphanumeric_pattern().to_string();
                m = m.with_alphanumeric_optimization(Some(&pattern)).map_err(broken)?;
            }
            ["english-dictionary"] if model == ENGLISH_MODEL => {
                let xml = res.fixture("english-dictionary.xml")?;
                let bytes = std::fs::read(&xml).map_err(|e| broken(format!("{}: {e}", xml.display())))?;
                m = m.with_abbreviations(Some(Dictionary::read(&bytes).map_err(broken)?));
            }
            other => return Err(Unavailable::NotCovered(format!("tokenize/{model} with {other:?}"))),
        }
        let tokenizer = TokenizerMe::new(m, TokenizerOptions::default());
        let analyzer = Analyzer::builder().add(tokenizer).build().map_err(broken)?;
        Ok(Box::new(Maxent { analyzer }))
    }
}

impl Run for Maxent {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    /// Where no JVM span cuts a pair, the layers compare through
    /// `matches_golden`. Where one does, the port does not take that split
    /// and multiplies on, so the JVM spans are joined as
    /// `joined_me_spans` says before comparing.
    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        if !has_cut(golden, "opennlp:tokens") {
            let spans = outcome(matches_golden::<Tokens>(doc, golden, Tolerance::exact()));
            if spans != Outcome::Match {
                return spans;
            }
            return outcome(matches_golden::<Confidence<Tokens>>(doc, golden, confidence_tolerance()));
        }
        let text = doc.source();
        let jvm = match jvm_spans(golden, text) {
            Ok(j) => j,
            Err(o) => return o,
        };
        let expected = match joined_me_spans(text, &jvm) {
            Ok(e) => e,
            Err(e) => return Outcome::Skipped(format!("the JVM's cut spans do not pair up: {e}")),
        };
        let got = token_spans(doc);
        let probs = doc.get::<Confidence<Tokens>>().map(|c| c.value.clone()).unwrap_or_default();
        let want: Vec<(u32, u32)> = expected.iter().map(|&(a, b, _)| (a, b)).collect();
        if let Some(what) = first_difference(text, &got, &want) {
            return Outcome::Divergence(format!("after joining the JVM's split inside a pair, {what}"));
        }
        for (i, (&p, &(a, b, want))) in probs.iter().zip(&expected).enumerate() {
            if (p - want).abs() > 1e-9 {
                return Outcome::Divergence(format!("token {i} at {a}..{b}: probability {p}, the JVM gave {want}"));
            }
        }
        Outcome::Match
    }
}

/// `SimpleTokenizer` or `WhitespaceTokenizer`, with or without
/// `keepNewLines`, in either whitespace mode.
pub(crate) struct Plain {
    simple: bool,
    options: TokenizerOptions,
    analyzer: Analyzer,
}

impl Plain {
    pub(crate) fn open(simple: bool, options: &[&str]) -> Result<Box<dyn Run>, Unavailable> {
        let (keep_newlines, rest): (bool, &[&str]) = match options {
            ["keep-new-lines", rest @ ..] => (true, rest),
            rest => (false, rest),
        };
        let options = TokenizerOptions { whitespace: whitespace_mode(rest)?, keep_newlines };
        let builder = Analyzer::builder();
        let analyzer = if simple {
            builder.add(SimpleTokenizer::new(options)).build()
        } else {
            builder.add(WhitespaceTokenizer::new(options)).build()
        };
        Ok(Box::new(Plain { simple, options, analyzer: analyzer.map_err(broken)? }))
    }

    /// The spans without `keepNewLines`, from which the expectation with
    /// it follows.
    fn plain_spans(&self, text: &str) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        let options = TokenizerOptions { keep_newlines: false, ..self.options };
        if self.simple {
            simple_spans(text, options, &mut out);
        } else {
            whitespace_spans(text, options, &mut out);
        }
        out.iter().map(|s| (s.start, s.end)).collect()
    }
}

impl Run for Plain {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        compare_whole_characters(doc, golden)
    }

    /// `SimpleTokenizer` with `keepNewLines` has two defects the port
    /// fixes (it throws when a line break comes before the first token,
    /// and gives a line break after whitespace a stale span), so the
    /// port's spans are checked against the plain spans plus a span per
    /// line break, as the golden test does, and the JVM's against the
    /// model of its loop, `jvm_simple_newline_spans`, exactly: a JVM result
    /// the model gives is a documented difference where it is not the
    /// port's, and anything else is a divergence.
    fn check(&self, text: &str, golden: &Golden) -> Checked {
        let doc = match self.annotate(text, Some(golden)) {
            Ok(doc) => doc,
            Err(e) => return Checked { doc: None, outcome: Outcome::Divergence(format!("the port failed: {e}")) },
        };
        if !(self.simple && self.options.keep_newlines) {
            let outcome = jvm_threw(golden).unwrap_or_else(|| self.compare(&doc, golden));
            return Checked { doc: Some(doc), outcome };
        }
        let corrected = simple_newline_spans(text, &self.plain_spans(text));
        let got = token_spans(&doc);
        if got != corrected {
            let what = format!(
                "the port's spans with keepNewLines {got:?} are not its plain spans plus the line breaks {corrected:?}"
            );
            return Checked { doc: Some(doc), outcome: Outcome::Divergence(what) };
        }
        let model = jvm_simple_newline_spans(text, self.options.whitespace);
        let outcome = match (&golden.header.error, model) {
            (Some(e), Err(class)) if e.class.ends_with("IllegalArgumentException") && class == e.class => {
                Outcome::Documented(format!(
                    "SimpleTokenizer with keepNewLines threw {}: {}, as modelled (a line break before the first token asks for a span from -1); the port gives the plain spans plus the line breaks (analysis/tokenize/tests/golden.rs, simple_newlines)",
                    e.class, e.message
                ))
            }
            (Some(e), _) => Outcome::Divergence(format!(
                "the JVM threw {}: {}, which the model of SimpleTokenizer with keepNewLines does not",
                e.class, e.message
            )),
            (None, Err(class)) => Outcome::Divergence(format!(
                "the model of SimpleTokenizer with keepNewLines throws {class} where the JVM gave spans"
            )),
            (None, Ok(model)) => {
                let jvm = match jvm_spans(golden, text) {
                    Ok(j) => j,
                    Err(o) => return Checked { doc: Some(doc), outcome: o },
                };
                let jvm16: Vec<(u32, u32)> = jvm.iter().map(|s| s.utf16).collect();
                if jvm16 != model {
                    let i = (0..jvm16.len().max(model.len())).find(|&i| jvm16.get(i) != model.get(i)).unwrap_or(0);
                    Outcome::Divergence(format!(
                        "the JVM's spans with keepNewLines are not the modelled stale start spans: token {i} is {:?} on the JVM and {:?} in the model, in UTF-16 units ({} and {} tokens)",
                        jvm16.get(i),
                        model.get(i),
                        jvm16.len(),
                        model.len()
                    ))
                } else {
                    match self.compare(&doc, golden) {
                        Outcome::Match => Outcome::Match,
                        Outcome::Divergence(what) => Outcome::Documented(format!(
                            "SimpleTokenizer with keepNewLines gives a line break after whitespace the span starting where the last token or line break ended, as modelled; the port gives the plain spans plus the line breaks (analysis/tokenize/tests/golden.rs, simple_newlines; {what})"
                        )),
                        other => other,
                    }
                }
            }
        };
        Checked { doc: Some(doc), outcome }
    }
}

/// The spans the JVM wrote, with the two halves of a cut character
/// joined into one span, against the port's.
fn compare_whole_characters(doc: &Document, golden: &Golden) -> Outcome {
    if !has_cut(golden, "opennlp:tokens") {
        return outcome(matches_golden::<Tokens>(doc, golden, Tolerance::exact()));
    }
    let text = doc.source();
    let jvm = match jvm_spans(golden, text) {
        Ok(j) => j,
        Err(o) => return o,
    };
    let want: Vec<(u32, u32)> = match whole_character_spans(text, &jvm) {
        Ok(w) => w.iter().map(|s| s.utf8).collect(),
        Err(e) => return Outcome::Skipped(format!("the JVM's cut spans do not pair up: {e}")),
    };
    match first_difference(text, &token_spans(doc), &want) {
        None => Outcome::Match,
        Some(what) => Outcome::Divergence(format!("after joining the halves the JVM split a character into, {what}")),
    }
}

/// Where two span lists first differ, with the spans and their text, or
/// nothing when they are the same.
fn first_difference(text: &str, rust: &[(u32, u32)], jvm: &[(u32, u32)]) -> Option<String> {
    let show = |s: Option<&(u32, u32)>| match s {
        Some(&(a, b)) => format!("{a}..{b} {:?}", text.get(a as usize..b as usize).unwrap_or("?")),
        None => "nothing".to_string(),
    };
    let i = (0..rust.len().max(jvm.len())).find(|&i| rust.get(i) != jvm.get(i))?;
    Some(format!(
        "token {i} differs: the port has {} and the JVM {} ({} and {} tokens)",
        show(rust.get(i)),
        show(jvm.get(i)),
        rust.len(),
        jvm.len()
    ))
}

/// The UAX #29 `WordTokenizer`, with a word type per token.
pub(crate) struct Uax29 {
    analyzer: Analyzer,
}

impl Uax29 {
    pub(crate) fn open() -> Result<Box<dyn Run>, Unavailable> {
        let analyzer = Analyzer::builder().add(WordTokenizer::new()).build().map_err(broken)?;
        Ok(Box::new(Uax29 { analyzer }))
    }
}

impl Run for Uax29 {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        let spans = compare_whole_characters(doc, golden);
        if spans != Outcome::Match {
            return spans;
        }
        let text = doc.source();
        let jvm = match jvm_spans(golden, text) {
            Ok(j) => j,
            Err(o) => return o,
        };
        let joined = match whole_character_spans(text, &jvm) {
            Ok(w) => w,
            Err(e) => return Outcome::Skipped(e),
        };
        let kinds: Vec<&str> =
            doc.get::<WordTypes>().map_or_else(Vec::new, |w| w.kind.iter().map(|k| k.java_name()).collect());
        for (i, (got, want)) in kinds.iter().zip(&joined).enumerate() {
            if Some(*got) != want.kind.as_deref() {
                return Outcome::Divergence(format!("token {i}: word type {got}, the JVM gave {:?}", want.kind));
            }
        }
        if kinds.len() != joined.len() {
            return Outcome::Divergence(format!("{} word types for {} tokens", kinds.len(), joined.len()));
        }
        Outcome::Match
    }
}

/// The unigram segmenter with a fixture lexicon in either mode.
pub(crate) struct Unigram {
    analyzer: Analyzer,
}

impl Unigram {
    pub(crate) fn open(lexicon: &str, options: &[&str], res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let mode = whitespace_mode(options)?;
        let path = res.fixture(&format!("{lexicon}.txt"))?;
        let lexicon = UnigramLexicon::read(&path, mode).map_err(|e| broken(format!("{}: {e}", path.display())))?;
        let analyzer = Analyzer::builder().add(UnigramSegmenter::new(lexicon, mode)).build().map_err(broken)?;
        Ok(Box::new(Unigram { analyzer }))
    }
}

impl Run for Unigram {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        if has_cut(golden, "opennlp:tokens") {
            return Outcome::Skipped("a JVM token starts or ends inside a surrogate pair".into());
        }
        outcome(matches_golden::<Tokens>(doc, golden, Tolerance::exact()))
    }
}

/// The lattice tokenizer with a MeCab dictionary: a fixture, or IPADIC
/// (its UTF-8 copy) or mecab-ko-dic from the models directory. The
/// dictionary is large and loaded once; the tokenizer is called as an
/// annotator directly rather than through an analyzer, which would need
/// a second one.
pub(crate) struct Lattice {
    tokenizer: LatticeTokenizer,
}

impl Lattice {
    pub(crate) fn open(dictionary: &str, options: &[&str], res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let mode = whitespace_mode(options)?;
        let dir = if dictionary.starts_with("mecab-fixture-") {
            res.fixture(dictionary)?
        } else if dictionary.starts_with("mecab-ipadic-") {
            res.model(&format!("{dictionary}.utf8"))?
        } else {
            res.model(dictionary)?
        };
        let loaded = MecabDictionary::load(&dir, mode).map_err(|e| broken(format!("{}: {e}", dir.display())))?;
        Ok(Box::new(Lattice { tokenizer: LatticeTokenizer::new(loaded, mode) }))
    }

    /// The byte offset the tokenizer finds no candidate at, if any.
    fn no_candidate(&self, text: &str) -> Option<usize> {
        let mut out = Vec::new();
        self.tokenizer.analyze_with(text, &mut LatticeScratch::new(), &mut out).err().map(|e| e.offset)
    }
}

/// The UTF-16 position in the JVM's "no candidate" message.
fn jvm_no_candidate(message: &str) -> Option<u32> {
    let rest = message.strip_prefix("dictionary provides no candidate at position ")?;
    rest.split(';').next()?.trim().parse().ok()
}

impl Run for Lattice {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        if let Some(offset) = self.no_candidate(text) {
            return Err(RustError { message: format!("no candidate at byte {offset}"), offset: Some(offset) });
        }
        let mut doc = Document::new(text);
        let view = doc.get::<View>().expect("the source view");
        let (tokens, tags) =
            self.tokenizer.annotate((view, None)).map_err(|e| RustError { message: e.to_string(), offset: None })?;
        let by = Provenance::by(LatticeTokenizer::ID);
        doc.add(&Key::DEFAULT, tokens, by.clone()).map_err(|e| RustError { message: e.to_string(), offset: None })?;
        doc.add(&Key::DEFAULT, tags, by).map_err(|e| RustError { message: e.to_string(), offset: None })?;
        Ok(doc)
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        if has_cut(golden, "opennlp:tokens") {
            return Outcome::Skipped("a JVM token starts or ends inside a surrogate pair".into());
        }
        outcome(matches_golden::<Tokens>(doc, golden, Tolerance::exact()))
    }

    /// Both sides may refuse a text whose dictionary offers no candidate
    /// at some position; then the positions must agree.
    fn check(&self, text: &str, golden: &Golden) -> Checked {
        let jvm = golden.header.error.as_ref();
        match (self.annotate(text, Some(golden)), jvm) {
            (Ok(doc), None) => {
                let outcome = self.compare(&doc, golden);
                Checked { doc: Some(doc), outcome }
            }
            (Ok(doc), Some(e)) => {
                let what = format!(
                    "the JVM threw {}: {}; the port returned {} tokens",
                    e.class,
                    e.message,
                    token_spans(&doc).len()
                );
                Checked { doc: Some(doc), outcome: Outcome::Divergence(what) }
            }
            (Err(e), None) => Checked { doc: None, outcome: Outcome::Divergence(format!("the port failed: {e}")) },
            (Err(e), Some(j)) => {
                let unit = e.offset.and_then(|b| byte_to_utf16(text, b)).map(|u| u as u32);
                let outcome = match (unit, jvm_no_candidate(&j.message)) {
                    (Some(u), Some(w)) if u == w => Outcome::Match,
                    (u, w) => Outcome::Divergence(format!(
                        "the JVM threw {}: {} (UTF-16 position {w:?}); the port failed with {e} (UTF-16 position {u:?})",
                        j.class, j.message
                    )),
                };
                Checked { doc: None, outcome }
            }
        }
    }
}
