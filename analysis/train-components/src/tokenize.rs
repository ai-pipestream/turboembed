//! Training the tokenizer: `TokenizerFactory`, the events of
//! `TokSpanEventStream`, and the model file `TokenizerME.train` returns
//! as a `TokenizerModel`.
//!
//! `TokSpanEventStream` cuts the text between the first token's start and
//! the last token's end into whitespace separated units, and for every
//! unit of more than one UTF-16 code unit (and, with the alphanumeric
//! optimization on, not matching the alphanumeric pattern) looks for the
//! sample's tokens inside it. It then asks `DefaultTokenContextGenerator`
//! for the features at every position inside each of those tokens, an
//! event with outcome `F` (no split), and at each token's end that is not
//! the unit's end, an event with outcome `T` (split). Positions are code
//! units, so a character outside the Basic Multilingual Plane gives an
//! event between its two halves.
//!
//! A token that overlaps a unit without being inside it (the sample's
//! tokens disagree with whitespace) is skipped with a warning on the JVM;
//! here it is skipped. The search for tokens starts after the last one
//! found and stops at the first token that ends after the unit, so the
//! tokens must be in text order.

use turbo_formats::TokenSample;
use turbo_maxent::Dictionary;
use turbo_tokenize::WhitespaceMode;
use turbo_tokenize::me::alnum::{AlphaNumeric, language_pattern};
use turbo_tokenize::me::{Features, UnitFacts, WhitespaceUnits, token_context};
use turbo_train::{
    Event, GisModel, JavaString, ModelPackage, TrainingConfiguration, TrainingParameters, TrainingReport, gis,
    write_gis,
};
use turbo_unicode::Utf16Cursor;

use crate::dictionary::AbbreviationDictionary;
use crate::error::{Error, Result};
use crate::package::{DICTIONARY_SERIALIZER, FACTORY_KEY, Stamp, add_artifacts, serializer_class_key};

/// `TokenizerModel.COMPONENT_NAME`.
pub const COMPONENT_NAME: &str = "TokenizerME";
/// `TokenizerModel.TOKENIZER_MODEL_ENTRY`.
pub const MODEL_ENTRY: &str = "token.model";
/// `TokenizerModel.ABBREVIATIONS_ENTRY_NAME`.
pub const ABBREVIATIONS_ENTRY: &str = "abbreviations.dictionary";
/// The factory class the manifest names.
pub const FACTORY: &str = "opennlp.tools.tokenize.TokenizerFactory";
/// `TokenizerFactory.USE_ALPHA_NUMERIC_OPTIMIZATION`.
pub const USE_ALPHA_NUMERIC_OPTIMIZATION_KEY: &str = "useAlphaNumericOptimization";
/// `TokenizerFactory.ALPHA_NUMERIC_PATTERN`.
pub const ALPHA_NUMERIC_PATTERN_KEY: &str = "alphaNumericPattern";
/// `TokenizerME.SPLIT`.
pub const SPLIT: &str = "T";
/// `TokenizerME.NO_SPLIT`.
pub const NO_SPLIT: &str = "F";

/// `TokenizerFactory(languageCode, abbreviationDictionary,
/// useAlphaNumericOptimization, alphaNumericPattern)` as a training run
/// uses it. With no pattern the factory takes the language's from
/// `opennlp.tools.tokenize.lang.Factory`, so the manifest always has
/// `alphaNumericPattern`.
#[derive(Debug, Clone)]
pub struct TokenizerFactory {
    language: String,
    abbreviations: Option<AbbreviationDictionary>,
    use_alpha_numeric_optimization: bool,
    alpha_numeric_pattern: String,
    alpha_numeric: AlphaNumeric,
}

impl TokenizerFactory {
    /// A pattern `turbo-tokenize` cannot evaluate is refused
    /// ([`crate::Error::Pattern`]), whether or not the optimization that
    /// would use it is on, as the JVM compiles it either way.
    pub fn new(
        language: &str,
        abbreviations: Option<AbbreviationDictionary>,
        use_alpha_numeric_optimization: bool,
        alpha_numeric_pattern: Option<&str>,
    ) -> Result<TokenizerFactory> {
        let pattern = alpha_numeric_pattern.unwrap_or_else(|| language_pattern(language));
        let alpha_numeric = AlphaNumeric::parse(pattern)?;
        Ok(TokenizerFactory {
            language: language.to_owned(),
            abbreviations,
            use_alpha_numeric_optimization,
            alpha_numeric_pattern: pattern.to_owned(),
            alpha_numeric,
        })
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn abbreviations(&self) -> Option<&AbbreviationDictionary> {
        self.abbreviations.as_ref()
    }

    /// `isUseAlphaNumericOptimization`.
    pub fn use_alpha_numeric_optimization(&self) -> bool {
        self.use_alpha_numeric_optimization
    }

    /// `getAlphaNumericPattern().pattern()`.
    pub fn alpha_numeric_pattern(&self) -> &str {
        &self.alpha_numeric_pattern
    }

    /// `createManifestEntries`, with `factory` as `BaseModel` adds it and
    /// the serializer class `BaseModel.serialize` records for the
    /// dictionary, in the order they are set.
    pub fn manifest_entries(&self) -> Vec<(String, String)> {
        let mut entries = vec![
            (FACTORY_KEY.to_owned(), FACTORY.to_owned()),
            (USE_ALPHA_NUMERIC_OPTIMIZATION_KEY.to_owned(), self.use_alpha_numeric_optimization.to_string()),
            (ALPHA_NUMERIC_PATTERN_KEY.to_owned(), self.alpha_numeric_pattern.clone()),
        ];
        if self.abbreviations.is_some() {
            entries.push((serializer_class_key(ABBREVIATIONS_ENTRY), DICTIONARY_SERIALIZER.to_owned()));
        }
        entries
    }
}

/// `TokSpanEventStream` for one factory, with the buffers its events are
/// built in.
pub struct TokSpanEventStream<'f> {
    alpha_numeric: &'f AlphaNumeric,
    skip_alpha_numerics: bool,
    dictionary: Option<&'f Dictionary>,
    whitespace: WhitespaceMode,
    features: Features,
    scratch: String,
}

impl<'f> TokSpanEventStream<'f> {
    /// The stream of `factory.isUseAlphaNumericOptimization()`,
    /// `factory.getAlphaNumericPattern()` and `factory.getContextGenerator()`,
    /// with `whitespace` where the Java calls `StringUtil.isWhitespace`
    /// (the whitespace tokenizer that cuts the units, and the `_ws`
    /// feature).
    pub fn new(factory: &'f TokenizerFactory, whitespace: WhitespaceMode) -> TokSpanEventStream<'f> {
        TokSpanEventStream {
            alpha_numeric: &factory.alpha_numeric,
            skip_alpha_numerics: factory.use_alpha_numeric_optimization,
            dictionary: factory.abbreviations().map(AbbreviationDictionary::lookup),
            whitespace,
            features: Features::new(),
            scratch: String::new(),
        }
    }

    /// `createEvents(sample)`: appends the sample's events to `out`. The
    /// tokens must be in text order ([`Error::TokensOutOfOrder`]), which
    /// the Java does not check. It throws only when the last token ends
    /// before the first starts (`String.substring`); for other orders it
    /// makes events of the tokens each whitespace unit contains, in their
    /// order, so a token inside the one before it (as MASC 3.0.0 has once)
    /// gives a split inside the outer token where that token gave a
    /// no-split. The port refuses every such sample.
    pub fn events(&mut self, sample: &TokenSample<'_>, out: &mut Vec<Event>) -> Result<()> {
        let tokens = sample.tokens();
        let (Some(first), Some(last)) = (tokens.first(), tokens.last()) else {
            return Ok(());
        };
        if let Some(i) = (1..tokens.len()).find(|&i| {
            let (p, t) = (tokens[i - 1], tokens[i]);
            t.start < p.start || t.end < p.end
        }) {
            return Err(Error::TokensOutOfOrder { token: i });
        }
        let text = sample.text();
        let start = first.start as usize;
        let sent = &text[start..last.end as usize];
        let mut last_training_token: Option<usize> = None;
        for (unit_start, unit_end) in WhitespaceUnits::new(sent, self.whitespace, false) {
            let ctok = &sent[unit_start..unit_end];
            // The candidate's span in the text.
            let (c_start, c_end) = (unit_start + start, unit_end + start);
            if one_code_unit(ctok) || (self.skip_alpha_numerics && self.alpha_numeric.test(ctok)) {
                continue;
            }
            // The training tokens inside the candidate.
            let mut found: Option<(usize, usize)> = None;
            for (ti, t) in tokens.iter().enumerate().skip(last_training_token.map_or(0, |l| l + 1)) {
                let (t_start, t_end) = (t.start as usize, t.end as usize);
                if c_start <= t_start && t_end <= c_end {
                    found = Some(found.map_or((ti, ti), |(f, _)| (f, ti)));
                    last_training_token = Some(ti);
                } else if c_end < t_end {
                    break;
                }
                // A token ending before the candidate is passed over; one
                // overlapping it is a bad training token, which the JVM
                // logs a warning for and skips.
            }
            let Some((first_ti, last_ti)) = found else {
                continue;
            };
            let abbreviation = self.dictionary.is_some_and(|d| d.contains_with(ctok, &mut self.scratch));
            let facts = UnitFacts::of(ctok, abbreviation);
            let mut at = Utf16Cursor::new(ctok);
            for t in &tokens[first_ti..=last_ti] {
                let (t_start, t_end) = (t.start as usize - c_start, t.end as usize - c_start);
                at.seek_byte(t_start);
                at.step_forward();
                // Every position inside the token, both halves of a
                // character outside the BMP included: the cursor's byte is
                // its character's first byte, and `t_end` is a character
                // boundary.
                while at.byte_offset() < t_end {
                    token_context(&at, facts, self.whitespace, &mut self.features);
                    out.push(Event::new(NO_SPLIT, java_strings(&self.features)));
                    at.step_forward();
                }
                if t_end != ctok.len() {
                    // The split is at the token's end, where the loop left
                    // the cursor unless the token is empty.
                    if at.byte_offset() != t_end || !at.char_start() {
                        at.seek_byte(t_end);
                    }
                    token_context(&at, facts, self.whitespace, &mut self.features);
                    out.push(Event::new(SPLIT, java_strings(&self.features)));
                }
            }
        }
        Ok(())
    }
}

/// Whether `unit` is one UTF-16 code unit: one character of up to three
/// bytes. `ctok.length() > 1` in the Java is the test for more.
fn one_code_unit(unit: &str) -> bool {
    unit.len() < 4 && unit.chars().nth(1).is_none()
}

/// The features as the predicates the JVM would hold.
fn java_strings(features: &Features) -> Vec<JavaString> {
    features.iter().map(|f| JavaString::from_modified_utf8(f).expect("a feature is modified UTF-8")).collect()
}

/// The events of `TokSpanEventStream` over `samples`.
pub fn tok_events<'a, 's>(
    samples: impl IntoIterator<Item = &'s TokenSample<'a>>,
    factory: &TokenizerFactory,
    whitespace: WhitespaceMode,
) -> Result<Vec<Event>>
where
    'a: 's,
{
    let mut stream = TokSpanEventStream::new(factory, whitespace);
    let mut out = Vec::new();
    for sample in samples {
        stream.events(sample, &mut out)?;
    }
    Ok(out)
}

/// A trained tokenizer model: the GIS model, what the trainer reports,
/// and the `.bin` file.
#[derive(Debug, Clone)]
pub struct TrainedTokenizerModel {
    pub model: GisModel,
    pub report: TrainingReport,
    pub package: ModelPackage,
}

/// `new TokenizerModel(tokenizerModel, manifestInfoEntries,
/// tokenizerFactory)` serialized: the manifest with the trainer's and the
/// factory's entries, the `token.model` entry holding `model_bytes` and,
/// with a dictionary, its `abbreviations.dictionary` entry, in the JVM's
/// zip order.
pub fn package_tokenizer_model(
    factory: &TokenizerFactory,
    report: &TrainingReport,
    model_bytes: Vec<u8>,
    stamp: Stamp<'_>,
) -> Result<ModelPackage> {
    let mut manifest = stamp.manifest(COMPONENT_NAME, factory.language());
    manifest.set_training(report);
    for (k, v) in factory.manifest_entries() {
        manifest.set(&k, &v);
    }
    let mut package = ModelPackage::new(manifest);
    let mut artifacts = Vec::new();
    if let Some(dictionary) = factory.abbreviations() {
        artifacts.push((ABBREVIATIONS_ENTRY, dictionary.serialize()?));
    }
    artifacts.push((MODEL_ENTRY, model_bytes));
    add_artifacts(&mut package, artifacts)?;
    Ok(package)
}

/// `TokenizerME.train(samples, factory, mlParams)`: the events of
/// [`tok_events`], the event trainer `mlParams` names, and the model file
/// stamped with `stamp`.
pub fn train_tokenizer_model<'a, 's>(
    samples: impl IntoIterator<Item = &'s TokenSample<'a>>,
    factory: &TokenizerFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    whitespace: WhitespaceMode,
    stamp: Stamp<'_>,
) -> Result<TrainedTokenizerModel>
where
    'a: 's,
{
    let events = tok_events(samples, factory, whitespace)?;
    train_tokenizer_model_from_events(&events, factory, params, config, stamp)
}

/// [`train_tokenizer_model`] from events already made.
pub fn train_tokenizer_model_from_events(
    events: &[Event],
    factory: &TokenizerFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedTokenizerModel> {
    let (model, report) = gis::train_events(events, params, config)?;
    let bytes = write_gis(&model)?;
    let package = package_tokenizer_model(factory, &report, bytes, stamp)?;
    Ok(TrainedTokenizerModel { model, report, package })
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_formats::native::TokenReader;

    fn samples(text: &str) -> Vec<TokenSample<'_>> {
        TokenReader::new(text, WhitespaceMode::Unicode).unwrap().map(|s| s.unwrap()).collect()
    }

    fn outcomes(events: &[Event]) -> String {
        events.iter().map(|e| e.outcome().to_string()).collect()
    }

    #[test]
    fn positions_inside_tokens_do_not_split_and_token_ends_inside_a_unit_do() {
        let f = TokenizerFactory::new("en", None, false, None).unwrap();
        let s = samples("Hi<SPLIT>, you<SPLIT>.\n");
        let events = tok_events(&s, &f, WhitespaceMode::Unicode).unwrap();
        // "Hi," gives F (inside Hi) and T (after Hi); "," at the unit's
        // end gives nothing. "you." gives F F T.
        assert_eq!(outcomes(&events), "FTFFT");
        let p: Vec<String> = events[1].context().iter().map(JavaString::to_string_lossy).collect();
        assert_eq!(&p[..2], ["p=Hi", "s=,"]);
    }

    #[test]
    fn one_unit_long_units_and_alphanumeric_units_give_no_events() {
        let plain = TokenizerFactory::new("en", None, false, None).unwrap();
        let optimized = TokenizerFactory::new("en", None, true, None).unwrap();
        let s = samples("a bcd ef<SPLIT>1\n");
        // "a" is one code unit; "bcd" is inside one token; "ef1" splits.
        assert_eq!(outcomes(&tok_events(&s, &plain, WhitespaceMode::Unicode).unwrap()), "FFFT");
        // With the optimization "ef1" is alphanumeric and is skipped too.
        assert_eq!(outcomes(&tok_events(&s, &optimized, WhitespaceMode::Unicode).unwrap()), "");
        // A single character outside the BMP is two code units: an event
        // between its halves.
        let s = samples("\u{1F600}\n");
        let events = tok_events(&s, &plain, WhitespaceMode::Unicode).unwrap();
        assert_eq!(outcomes(&events), "F");
        assert_eq!(events[0].context()[0].as_bytes(), b"p=\xED\xA0\xBD");
    }

    #[test]
    fn the_abbreviation_feature_marks_the_last_position_of_a_listed_unit() {
        let d = AbbreviationDictionary::new(false, vec![vec!["Dr.".to_string()]]).unwrap();
        let f = TokenizerFactory::new("en", Some(d), false, None).unwrap();
        let s = samples("Dr<SPLIT>. Who\n");
        let events = tok_events(&s, &f, WhitespaceMode::Unicode).unwrap();
        // "Dr.": F inside Dr, T after it; "Who": two F.
        assert_eq!(outcomes(&events), "FTFF");
        assert!(events[1].context().iter().any(|c| c.as_bytes() == b"pabb"));
        assert!(!events[0].context().iter().any(|c| c.as_bytes() == b"pabb"));
    }

    #[test]
    fn the_language_decides_the_pattern_and_the_manifest_always_lists_it() {
        let de = TokenizerFactory::new("de", None, true, None).unwrap();
        assert_eq!(de.alpha_numeric_pattern(), language_pattern("de"));
        let en = TokenizerFactory::new("en", None, false, Some("^[a-z]+$")).unwrap();
        assert_eq!(
            en.manifest_entries(),
            [("factory", FACTORY), ("useAlphaNumericOptimization", "false"), ("alphaNumericPattern", "^[a-z]+$")]
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
        );
        assert!(matches!(TokenizerFactory::new("en", None, false, Some("(")), Err(crate::Error::Pattern(_))));
    }

    #[test]
    fn a_token_overlapping_a_unit_is_skipped_as_the_jvm_skips_it() {
        // The sample's tokens: "ab" and "c d" (the second holds a space), so
        // the unit "c" overlaps the second token without containing it.
        let text = "ab c d";
        let s = TokenSample::new(
            std::borrow::Cow::Borrowed(text),
            vec![turbo_analysis_span(0, 2), turbo_analysis_span(3, 6)],
            vec![turbo_analysis_span(0, 2), turbo_analysis_span(3, 6)],
            turbo_analysis_span(0, 6),
        )
        .unwrap();
        let f = TokenizerFactory::new("en", None, false, None).unwrap();
        let events = tok_events([&s], &f, WhitespaceMode::Unicode).unwrap();
        // "ab" is one token in one unit: one F. "c" and "d" find no token.
        assert_eq!(outcomes(&events), "F");
    }

    #[test]
    fn an_empty_token_splits_at_its_position_even_at_the_start_of_the_unit() {
        // The sample's tokens: an empty one at 0, "xy", an empty one at 2
        // and "z" (CoNLL-U with empty word forms gives such samples; the
        // `train-events` goldens hold the JVM's events for one). The empty
        // token at the start splits at index 0 (`p1=bok`), "xy" makes a
        // non-split at 1 and a split at 2, the empty token at 2 splits at
        // 2 again, and "z" ends with the unit, so nothing splits there.
        let text = "xyz";
        let spans = vec![
            turbo_analysis_span(0, 0),
            turbo_analysis_span(0, 2),
            turbo_analysis_span(2, 2),
            turbo_analysis_span(2, 3),
        ];
        let s = TokenSample::new(std::borrow::Cow::Borrowed(text), spans.clone(), spans, turbo_analysis_span(0, 3))
            .unwrap();
        let f = TokenizerFactory::new("en", None, false, None).unwrap();
        let events = tok_events([&s], &f, WhitespaceMode::Unicode).unwrap();
        assert_eq!(outcomes(&events), "TFTT");
        assert_eq!(events[0].context()[2].to_string_lossy(), "p1=bok");
        assert_eq!(events[3].context()[0].to_string_lossy(), "p=xy");
    }

    #[test]
    fn a_unit_of_one_code_unit_is_one_character_of_up_to_three_bytes() {
        assert!(one_code_unit("a") && one_code_unit("\u{E9}") && one_code_unit("\u{4E2D}"));
        assert!(!one_code_unit("ab") && !one_code_unit("\u{1F600}") && !one_code_unit("\u{E9}a"));
    }

    #[test]
    fn tokens_out_of_text_order_are_refused() {
        let text = "ab cd";
        let spans = vec![turbo_analysis_span(3, 5), turbo_analysis_span(0, 2)];
        let s = TokenSample::new(std::borrow::Cow::Borrowed(text), spans.clone(), spans, turbo_analysis_span(0, 5))
            .unwrap();
        let f = TokenizerFactory::new("en", None, false, None).unwrap();
        assert!(matches!(
            tok_events([&s], &f, WhitespaceMode::Unicode),
            Err(crate::Error::TokensOutOfOrder { token: 1 })
        ));
    }

    fn turbo_analysis_span(start: u32, end: u32) -> turbo_analysis::Span {
        turbo_analysis::Span::new(start, end)
    }
}
