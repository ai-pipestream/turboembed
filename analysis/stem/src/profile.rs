//! Per-language normalization profiles (`NormalizationProfile`,
//! `NormalizationProfiles`) and [`TermAnnotator`], the pipeline step that
//! writes [`Terms`] (`turbo:terms`) aligned to `opennlp:tokens`.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, OnceLock};

use turbo_analysis::{Annotator, Error, Layer, Refs, View};
use turbo_analysis_layers::{Language, Lemmas, PosTags, Strs, Terms, Tokens};
use turbo_langdetect::LanguageDetector;
use turbo_normalize::{AccentFold, GermanUmlauts};
use turbo_unicode::java::code_point::is_whitespace as is_java_whitespace;
use turbo_unicode::java::to_lowercase;

use crate::snowball::{SnowballAlgorithm, SnowballStemmer};
use crate::term::{Dimension, TermAnalyzer, TermError};

/// Which diacritic fold a [`NormalizationProfile`] applies for its
/// [`NormalizationProfile::matching_analyzer`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AccentFoldKind {
    /// Generic Latin/Greek/Cyrillic accent fold (`AccentFoldCharSequenceNormalizer.getInstance()`),
    /// used for English and the major Romance languages (`cat`, `eng`, `fra`, `ita`, `por`, `spa`).
    Latin,
    /// German DIN 5007-2 digraph fold (`GermanUmlautCharSequenceNormalizer.getInstance()`),
    /// mapping `ä`/`ö`/`ü`/`ß` to `ae`/`oe`/`ue`/`ss`.
    German,
}

/// Per-language normalization settings (`opennlp.tools.util.normalizer.NormalizationProfile`).
///
/// Pairs an ISO 639-3 language code with its Snowball [`SnowballAlgorithm`]
/// and the diacritic fold appropriate for that language (if any).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormalizationProfile {
    language: String,
    stemmer_algorithm: SnowballAlgorithm,
    accent_fold: Option<AccentFoldKind>,
}

impl NormalizationProfile {
    /// Constructs a new [`NormalizationProfile`].
    ///
    /// # Errors
    ///
    /// Returns [`TermError::BlankLanguage`] if `language` is empty or
    /// contains only Java whitespace.
    pub fn new(
        language: impl Into<String>,
        stemmer_algorithm: SnowballAlgorithm,
        accent_fold: Option<AccentFoldKind>,
    ) -> Result<NormalizationProfile, TermError> {
        let language = language.into();
        if language.chars().all(|c| is_java_whitespace(c as u32)) {
            return Err(TermError::BlankLanguage);
        }
        Ok(NormalizationProfile { language, stemmer_algorithm, accent_fold })
    }

    const fn const_new(
        language: &'static str,
        stemmer_algorithm: SnowballAlgorithm,
        accent_fold: Option<AccentFoldKind>,
    ) -> (&'static str, SnowballAlgorithm, Option<AccentFoldKind>) {
        (language, stemmer_algorithm, accent_fold)
    }

    /// Returns the ISO 639-3 language code (for example `"eng"`).
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Returns the Snowball algorithm for the language.
    pub fn stemmer_algorithm(&self) -> SnowballAlgorithm {
        self.stemmer_algorithm
    }

    /// Returns the diacritic fold for the language, or `None` when diacritics
    /// mark distinct letters (Nordic languages, non-Latin scripts).
    pub fn accent_fold(&self) -> Option<AccentFoldKind> {
        self.accent_fold
    }

    /// Returns a new [`SnowballStemmer`] for this language (`NormalizationProfile.newStemmer()`).
    pub fn new_stemmer(&self) -> SnowballStemmer {
        SnowballStemmer::new(self.stemmer_algorithm)
    }

    /// Returns a matching [`TermAnalyzer`] for this language
    /// (`NormalizationProfile.matchingAnalyzer()`): `NFC`, `CASE_FOLD`, the
    /// language's [`NormalizationProfile::accent_fold`] when present, then
    /// `STEM` wrapped in a bounded LRU [`crate::CachingStemmer`].
    pub fn matching_analyzer(&self) -> TermAnalyzer {
        let mut builder = TermAnalyzer::builder().nfc().case_fold().expect("only CaseFold is enabled");
        builder = match self.accent_fold {
            Some(AccentFoldKind::Latin) => builder
                .transform(Dimension::AccentFold, AccentFold::new())
                .expect("AccentFold is a character-level dimension"),
            Some(AccentFoldKind::German) => builder
                .transform(Dimension::AccentFold, GermanUmlauts::new())
                .expect("AccentFold is a character-level dimension"),
            None => builder,
        };
        builder.stem_cached(self.new_stemmer()).build()
    }
}

/// The 22 ISO 639-3 profiles in `NormalizationProfiles.BY_LANGUAGE`, sorted by code.
const PROFILES: [(&str, SnowballAlgorithm, Option<AccentFoldKind>); 22] = [
    NormalizationProfile::const_new("ara", SnowballAlgorithm::Arabic, None),
    NormalizationProfile::const_new("cat", SnowballAlgorithm::Catalan, Some(AccentFoldKind::Latin)),
    NormalizationProfile::const_new("dan", SnowballAlgorithm::Danish, None),
    NormalizationProfile::const_new("deu", SnowballAlgorithm::German, Some(AccentFoldKind::German)),
    NormalizationProfile::const_new("ell", SnowballAlgorithm::Greek, None),
    NormalizationProfile::const_new("eng", SnowballAlgorithm::English, Some(AccentFoldKind::Latin)),
    NormalizationProfile::const_new("fin", SnowballAlgorithm::Finnish, None),
    NormalizationProfile::const_new("fra", SnowballAlgorithm::French, Some(AccentFoldKind::Latin)),
    NormalizationProfile::const_new("gle", SnowballAlgorithm::Irish, None),
    NormalizationProfile::const_new("hun", SnowballAlgorithm::Hungarian, None),
    NormalizationProfile::const_new("ind", SnowballAlgorithm::Indonesian, None),
    NormalizationProfile::const_new("ita", SnowballAlgorithm::Italian, Some(AccentFoldKind::Latin)),
    NormalizationProfile::const_new("nld", SnowballAlgorithm::Dutch, None),
    NormalizationProfile::const_new("nno", SnowballAlgorithm::Norwegian, None),
    NormalizationProfile::const_new("nob", SnowballAlgorithm::Norwegian, None),
    NormalizationProfile::const_new("nor", SnowballAlgorithm::Norwegian, None),
    NormalizationProfile::const_new("por", SnowballAlgorithm::Portuguese, Some(AccentFoldKind::Latin)),
    NormalizationProfile::const_new("ron", SnowballAlgorithm::Romanian, None),
    NormalizationProfile::const_new("rus", SnowballAlgorithm::Russian, None),
    NormalizationProfile::const_new("spa", SnowballAlgorithm::Spanish, Some(AccentFoldKind::Latin)),
    NormalizationProfile::const_new("swe", SnowballAlgorithm::Swedish, None),
    NormalizationProfile::const_new("tur", SnowballAlgorithm::Turkish, None),
];

/// Supported ISO 639-3 language codes in `NormalizationProfiles`.
const SUPPORTED_LANGUAGES: [&str; 22] = [
    "ara", "cat", "dan", "deu", "ell", "eng", "fin", "fra", "gle", "hun", "ind", "ita", "nld", "nno", "nob", "nor",
    "por", "ron", "rus", "spa", "swe", "tur",
];

/// Maps two-letter ISO 639-1 codes to their ISO 639-3 counterparts, matching
/// `Locale.of(code).getISO3Language()` on JDK 21 for all supported languages.
fn iso1_to_iso3(code: &str) -> Option<&'static str> {
    Some(match code {
        "ar" => "ara",
        "ca" => "cat",
        "da" => "dan",
        "de" => "deu",
        "el" => "ell",
        "en" => "eng",
        "es" => "spa",
        "fi" => "fin",
        "fr" => "fra",
        "ga" => "gle",
        "hu" => "hun",
        "id" | "in" => "ind",
        "it" => "ita",
        "nb" => "nob",
        "nl" => "nld",
        "nn" => "nno",
        "no" => "nor",
        "pt" => "por",
        "ro" => "ron",
        "ru" => "rus",
        "sv" => "swe",
        "tr" => "tur",
        _ => return None,
    })
}

/// Registry of [`NormalizationProfile`]s by language (`opennlp.tools.util.normalizer.NormalizationProfiles`).
pub struct NormalizationProfiles;

impl NormalizationProfiles {
    /// Returns the [`NormalizationProfile`] for an ISO 639-3 or ISO 639-1
    /// language code (case-insensitive, stripped of leading/trailing Java
    /// whitespace), or `None` if the language has no Snowball stemmer
    /// (`NormalizationProfiles.forLanguage(String)`).
    pub fn for_language(language: &str) -> Option<NormalizationProfile> {
        Self::profile_index(language).map(Self::profile_at)
    }

    /// Index into [`PROFILES`] of `language`, resolved as [`Self::for_language`] does.
    fn profile_index(language: &str) -> Option<usize> {
        let stripped = language.trim_matches(|c: char| is_java_whitespace(c as u32));
        let mut lower = String::with_capacity(stripped.len());
        to_lowercase(stripped, &mut lower);
        let code: &str = if lower.len() == 2 { iso1_to_iso3(&lower).unwrap_or(&lower) } else { &lower };
        PROFILES.binary_search_by_key(&code, |&(lang, _, _)| lang).ok()
    }

    fn profile_at(idx: usize) -> NormalizationProfile {
        let (lang, alg, fold) = PROFILES[idx];
        NormalizationProfile { language: lang.to_owned(), stemmer_algorithm: alg, accent_fold: fold }
    }

    /// Detects the best language of `text` with `detector` and returns its
    /// [`NormalizationProfile`], or `None` if the detected language has no
    /// Snowball stemmer (`NormalizationProfiles.detect(CharSequence, LanguageDetector)`).
    pub fn detect(text: &str, detector: &LanguageDetector) -> Option<NormalizationProfile> {
        let mut scratch = turbo_langdetect::Scratch::default();
        let prediction = detector.predict_language(text, &mut scratch);
        Self::for_language(prediction.code)
    }

    /// Returns the 22 ISO 639-3 language codes of the supported profiles
    /// (`NormalizationProfiles.supportedLanguages()`).
    pub fn supported_languages() -> &'static [&'static str] {
        &SUPPORTED_LANGUAGES
    }
}

/// How [`TermAnnotator`] selects a [`TermAnalyzer`] for a document.
#[derive(Clone, Debug)]
enum AnalyzerChoice {
    /// Uses a fixed [`TermAnalyzer`].
    Fixed(TermAnalyzer),
    /// Uses the `opennlp:language` layer's best language profile when present,
    /// falling back to `fallback` (or `NFC` + `CASE_FOLD` when the language has
    /// no Snowball profile).
    ByLanguage { fallback: Option<String>, analyzers: Arc<ProfileAnalyzers> },
}

/// The analyzers [`AnalyzerChoice::ByLanguage`] has built so far, one per entry
/// of [`PROFILES`] plus the `NFC` + `CASE_FOLD` default, so every document in
/// a language shares one analyzer and its stem cache.
#[derive(Debug, Default)]
struct ProfileAnalyzers {
    by_profile: [OnceLock<TermAnalyzer>; PROFILES.len()],
    default: OnceLock<TermAnalyzer>,
}

/// Pipeline step that reads `opennlp:tokens` (and optionally `opennlp:language`,
/// `opennlp:pos`, and `opennlp:lemmas`) and writes [`Terms`] (`turbo:terms`),
/// aligned to `opennlp:tokens`.
#[derive(Clone, Debug)]
pub struct TermAnnotator {
    choice: AnalyzerChoice,
}

impl Default for TermAnnotator {
    fn default() -> Self {
        Self::by_language(None)
    }
}

impl TermAnnotator {
    /// Annotates documents with a fixed [`TermAnalyzer`].
    pub fn new(analyzer: TermAnalyzer) -> TermAnnotator {
        TermAnnotator { choice: AnalyzerChoice::Fixed(analyzer) }
    }

    /// Annotates documents using the [`NormalizationProfile::matching_analyzer`]
    /// for `language` (an ISO 639-3 or ISO 639-1 code).
    pub fn for_language(language: &str) -> Option<TermAnnotator> {
        let profile = NormalizationProfiles::for_language(language)?;
        Some(TermAnnotator::new(profile.matching_analyzer()))
    }

    /// Selects a [`NormalizationProfile`] from the document's `opennlp:language`
    /// layer at run time, with an optional `fallback` language code when the
    /// document has no `opennlp:language` layer or its best language is not in
    /// [`NormalizationProfiles`].
    pub fn by_language(fallback: Option<String>) -> TermAnnotator {
        TermAnnotator { choice: AnalyzerChoice::ByLanguage { fallback, analyzers: Arc::default() } }
    }

    fn resolve_analyzer(&self, language: Option<&Language>) -> TermAnalyzer {
        match &self.choice {
            AnalyzerChoice::Fixed(analyzer) => analyzer.clone(),
            AnalyzerChoice::ByLanguage { fallback, analyzers } => {
                let idx = language
                    .and_then(Language::best)
                    .and_then(|(best_code, _)| NormalizationProfiles::profile_index(best_code))
                    .or_else(|| fallback.as_deref().and_then(NormalizationProfiles::profile_index));
                match idx {
                    Some(idx) => analyzers.by_profile[idx]
                        .get_or_init(|| NormalizationProfiles::profile_at(idx).matching_analyzer())
                        .clone(),
                    None => analyzers
                        .default
                        .get_or_init(|| {
                            TermAnalyzer::builder().nfc().case_fold().expect("only CaseFold is enabled").build()
                        })
                        .clone(),
                }
            }
        }
    }
}

impl Annotator for TermAnnotator {
    const ID: &'static str = "turbo-stem:terms";
    type Reads = (View, Tokens, Option<Language>, Option<PosTags>, Option<Lemmas>);
    type Writes = (Terms,);

    fn annotate(&self, (view, tokens, language, pos, lemmas): Refs<'_, Self>) -> Result<(Terms,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "the tokens are in another view".into() });
        }
        let n = tokens.spans.len();
        if let Some(pos) = pos
            && pos.len() != n
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("pos has {} items, tokens has {n}", pos.len()),
            });
        }
        if let Some(lemmas) = lemmas
            && lemmas.len() != n
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("lemmas has {} items, tokens has {n}", lemmas.len()),
            });
        }

        let analyzer = self.resolve_analyzer(language);
        let text = view.text();
        let token_strs: Vec<&str> = tokens.spans.iter().map(|(_, s)| &text[s.range()]).collect();
        let tag_strs: Vec<Option<&str>> = match pos {
            Some(p) => p.tags.iter().map(|&t| Some(p.tagset.label(t))).collect(),
            None => vec![None; n],
        };

        let analyzed = analyzer
            .analyze_tokens(&token_strs, &tag_strs)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;

        let mut surface = Strs::default();
        let mut folded = Strs::default();
        let write_stem = analyzer.has_stemmer();
        let write_lemma = lemmas.is_some() || analyzer.has_lemmatizer();
        let mut stem_col = write_stem.then(Strs::default);
        let mut lemma_col = write_lemma.then(Strs::default);

        // Determine which dimension gives the character-folded slot: the last
        // character-level dimension in `analyzer.dimensions()`, or `Original`
        // when no character-level dimensions are configured.
        let folded_dim =
            analyzer.dimensions().iter().copied().rfind(|d| d.is_character_level()).unwrap_or(Dimension::Original);

        for (i, term) in analyzed.iter().enumerate() {
            surface.push(term.original());
            let f = term.at(folded_dim).map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
            folded.push(f);
            if let Some(stems) = &mut stem_col {
                let s = term
                    .at(Dimension::Stem)
                    .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
                stems.push(s);
            }
            if let Some(lemma_out) = &mut lemma_col {
                if let Some(doc_lemmas) = lemmas {
                    lemma_out.push(doc_lemmas.lemma.get(i));
                } else {
                    let l = term
                        .at(Dimension::Lemma)
                        .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
                    lemma_out.push(l);
                }
            }
        }

        Ok((Terms { surface, folded, stem: stem_col, lemma: lemma_col },))
    }

    fn config_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        match &self.choice {
            AnalyzerChoice::Fixed(a) => {
                0u8.hash(&mut h);
                a.dimensions().hash(&mut h);
                a.has_stemmer().hash(&mut h);
                a.has_lemmatizer().hash(&mut h);
            }
            AnalyzerChoice::ByLanguage { fallback, .. } => {
                1u8.hash(&mut h);
                fallback.hash(&mut h);
            }
        }
        h.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use turbo_analysis::{Analyzer, Document, Ix, Key, Provenance, Span, Spans, ViewId};
    use turbo_analysis_layers::TermSlot;

    use super::*;

    #[test]
    fn english_german_french_spanish_and_nordic_profiles_match_opennlp() {
        let eng = NormalizationProfiles::for_language("eng").unwrap();
        assert_eq!(eng.stemmer_algorithm(), SnowballAlgorithm::English);
        assert_eq!(eng.accent_fold(), Some(AccentFoldKind::Latin));
        assert_eq!(
            eng.matching_analyzer().dimensions(),
            &[Dimension::Nfc, Dimension::CaseFold, Dimension::AccentFold, Dimension::Stem]
        );
        assert_eq!(eng.matching_analyzer().analyze("Cats").unwrap()[0].normalized(), "cat");

        let deu = NormalizationProfiles::for_language("deu").unwrap();
        assert_eq!(deu.stemmer_algorithm(), SnowballAlgorithm::German);
        assert_eq!(deu.accent_fold(), Some(AccentFoldKind::German));
        assert_eq!(
            deu.matching_analyzer().dimensions(),
            &[Dimension::Nfc, Dimension::CaseFold, Dimension::AccentFold, Dimension::Stem]
        );
        let deu_term = &deu.matching_analyzer().analyze("B\u{00FC}cher").unwrap()[0];
        assert_eq!(deu_term.peel(), "buecher");
        assert_eq!(deu_term.normalized(), "buch");

        for code in ["fra", "spa", "por", "ita", "cat"] {
            let p = NormalizationProfiles::for_language(code).unwrap();
            assert_eq!(p.accent_fold(), Some(AccentFoldKind::Latin), "{code}");
        }
        let fra = NormalizationProfiles::for_language("fr").unwrap();
        assert_eq!(fra.language(), "fra");
        assert_eq!(fra.stemmer_algorithm(), SnowballAlgorithm::French);
        let fra_term = &fra.matching_analyzer().analyze("\u{00C9}l\u{00E8}ves").unwrap()[0];
        assert_eq!(fra_term.peel(), "eleves");
        assert_eq!(fra_term.normalized(), "elev");

        let spa = NormalizationProfiles::for_language("es").unwrap();
        assert_eq!(spa.language(), "spa");
        assert_eq!(spa.stemmer_algorithm(), SnowballAlgorithm::Spanish);
        let spa_term = &spa.matching_analyzer().analyze("Canciones").unwrap()[0];
        assert_eq!(spa_term.peel(), "canciones");
        assert_eq!(spa_term.normalized(), "cancion");

        let swe = NormalizationProfiles::for_language("swe").unwrap();
        assert_eq!(swe.accent_fold(), None);
        assert_eq!(swe.matching_analyzer().dimensions(), &[Dimension::Nfc, Dimension::CaseFold, Dimension::Stem]);
    }

    #[test]
    fn two_letter_codes_and_norwegian_standards_resolve_as_on_jvm() {
        assert_eq!(NormalizationProfiles::for_language("de").unwrap().stemmer_algorithm(), SnowballAlgorithm::German);
        assert_eq!(
            NormalizationProfiles::for_language("  EN ").unwrap().stemmer_algorithm(),
            SnowballAlgorithm::English
        );
        for code in ["nb", "nn", "no", "nob", "nno", "nor"] {
            assert_eq!(
                NormalizationProfiles::for_language(code).unwrap().stemmer_algorithm(),
                SnowballAlgorithm::Norwegian,
                "{code}"
            );
        }
        assert!(NormalizationProfiles::for_language("jpn").is_none());
        assert!(NormalizationProfiles::for_language("zzz").is_none());
        assert!(NormalizationProfiles::for_language("qq").is_none());
        assert_eq!(NormalizationProfile::new("   ", SnowballAlgorithm::English, None), Err(TermError::BlankLanguage));
    }

    #[test]
    fn supported_languages_cover_every_snowball_language_except_porter() {
        let covered: BTreeSet<SnowballAlgorithm> = NormalizationProfiles::supported_languages()
            .iter()
            .map(|&code| NormalizationProfiles::for_language(code).unwrap().stemmer_algorithm())
            .collect();
        let expected: BTreeSet<SnowballAlgorithm> =
            SnowballAlgorithm::ALL.iter().copied().filter(|&alg| alg != SnowballAlgorithm::Porter).collect();
        assert_eq!(covered, expected);
        for code in ["nob", "nno", "nor"] {
            assert!(NormalizationProfiles::supported_languages().contains(&code));
        }
    }

    #[test]
    fn matching_analyzer_is_thread_safe_for_stemming() {
        let analyzer = Arc::new(NormalizationProfiles::for_language("eng").unwrap().matching_analyzer());
        let expected = analyzer.analyze("running").unwrap()[0].at(Dimension::Stem).unwrap().to_owned();
        let mut handles = Vec::with_capacity(8);
        for _ in 0..8 {
            let a = Arc::clone(&analyzer);
            let exp = expected.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..50 {
                    assert_eq!(a.analyze("running").unwrap()[0].at(Dimension::Stem).unwrap(), exp);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn by_language_annotator_reuses_one_analyzer_per_language() {
        let annotator = TermAnnotator::by_language(Some("deu".to_owned()));
        let mut code = Strs::default();
        code.push("en");
        let english = Language { code, score: vec![1.0] };

        let first = annotator.resolve_analyzer(Some(&english));
        let second = annotator.resolve_analyzer(Some(&english));
        assert!(first.shares_state_with(&second));
        assert_eq!(first.analyze("Cats").unwrap()[0].normalized(), "cat");

        let fallback = annotator.resolve_analyzer(None);
        assert!(fallback.shares_state_with(&annotator.resolve_analyzer(None)));
        assert!(!fallback.shares_state_with(&first));
        assert_eq!(fallback.analyze("B\u{00FC}cher").unwrap()[0].normalized(), "buch");

        let plain = TermAnnotator::default();
        let mut unknown = Strs::default();
        unknown.push("jpn");
        let japanese = Language { code: unknown, score: vec![1.0] };
        let default = plain.resolve_analyzer(Some(&japanese));
        assert!(default.shares_state_with(&plain.resolve_analyzer(None)));
        assert_eq!(default.dimensions(), &[Dimension::Nfc, Dimension::CaseFold]);
    }

    #[test]
    fn term_annotator_reads_language_and_lemmas_and_writes_turbo_terms() {
        let text = "Caf\u{00E9}s Running";
        let mut doc = Document::new(text);
        let mut tok_spans = Spans::new(ViewId::SOURCE);
        tok_spans.push(Span::new(0, 6));
        tok_spans.push(Span::new(7, 14));
        doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("test")).unwrap();

        let mut lang_code = Strs::default();
        lang_code.push("eng");
        doc.add(&Key::DEFAULT, Language { code: lang_code, score: vec![1.0] }, Provenance::by("test")).unwrap();

        let mut lemma_strs = Strs::default();
        lemma_strs.push("cafe");
        lemma_strs.push("run");
        doc.add(&Key::DEFAULT, Lemmas { lemma: lemma_strs }, Provenance::by("test")).unwrap();

        let pipeline = Analyzer::builder()
            .given(Key::<View>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Language>::DEFAULT)
            .given(Key::<Lemmas>::DEFAULT)
            .add(TermAnnotator::default())
            .build()
            .unwrap();
        let status = pipeline.run(std::slice::from_mut(&mut doc));
        assert_eq!(status[0], Ok(()));

        let terms = doc.get::<Terms>().unwrap();
        assert_eq!(terms.len(), 2);
        assert_eq!(
            terms.get(Ix::new(0)),
            TermSlot { surface: "Caf\u{00E9}s", folded: "cafes", stem: Some("cafe"), lemma: Some("cafe") }
        );
        assert_eq!(
            terms.get(Ix::new(1)),
            TermSlot { surface: "Running", folded: "running", stem: Some("run"), lemma: Some("run") }
        );
    }
}
