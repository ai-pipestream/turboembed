//! Per-language normalization profiles (`NormalizationProfile`,
//! `NormalizationProfiles`) and [`TermAnnotator`], the pipeline step that
//! writes [`Terms`] (`turbo:terms`) aligned to `opennlp:tokens`.

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
        let (starts, ends) = tokens.spans.columns();

        let mut surface = Strs::default();
        let mut folded = Strs::default();
        let mut stem_col = analyzer.has_stemmer().then(Strs::default);
        let mut lemma_col = (lemmas.is_some() || analyzer.has_lemmatizer()).then(Strs::default);

        for (i, (&start, &end)) in starts.iter().zip(ends).enumerate() {
            let token = (start <= end).then(|| text.get(start as usize..end as usize)).flatten().ok_or_else(|| {
                Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("token {i} span {start}..{end} is not a valid slice of a {}-byte view", text.len()),
                }
            })?;
            let pos_tag = pos.map(|p| p.tagset.label(p.tags[i]));
            let (folded_tok, stem_tok, lemma_tok) = analyzer
                .analyze_for_annotator(token, pos_tag, lemmas.is_some())
                .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;

            surface.push(token);
            folded.push(&folded_tok);
            if let (Some(stems), Some(s)) = (&mut stem_col, &stem_tok) {
                stems.push(s);
            }
            if let Some(lemma_out) = &mut lemma_col {
                if let Some(doc_lemmas) = lemmas {
                    lemma_out.push(doc_lemmas.lemma.get(i));
                } else if let Some(l) = &lemma_tok {
                    lemma_out.push(l);
                }
            }
        }

        Ok((Terms { surface, folded, stem: stem_col, lemma: lemma_col },))
    }

    fn config_hash(&self) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0100_0000_01b3;
        fn write(h: &mut u64, bytes: &[u8]) {
            for &b in bytes {
                *h = (*h ^ u64::from(b)).wrapping_mul(FNV_PRIME);
            }
        }
        let mut h = FNV_OFFSET;
        match &self.choice {
            AnalyzerChoice::Fixed(a) => {
                write(&mut h, &[0u8]);
                write(&mut h, &(a.dimensions().len() as u64).to_le_bytes());
                for &dim in a.dimensions() {
                    write(&mut h, &[dim.index() as u8]);
                    if let Some((is_custom, id, norm_hash)) = a.normalizer_info(dim) {
                        write(&mut h, &[u8::from(is_custom)]);
                        write(&mut h, &(id.len() as u64).to_le_bytes());
                        write(&mut h, id.as_bytes());
                        write(&mut h, &norm_hash.to_le_bytes());
                    }
                }
                match a.stemmer_config_hash() {
                    Some(sh) => {
                        write(&mut h, &[1u8]);
                        write(&mut h, &sh.to_le_bytes());
                    }
                    None => write(&mut h, &[0u8]),
                }
                write(&mut h, &[u8::from(a.has_lemmatizer())]);
            }
            AnalyzerChoice::ByLanguage { fallback, .. } => {
                write(&mut h, &[1u8]);
                match fallback {
                    Some(lang) => {
                        write(&mut h, &[1u8]);
                        write(&mut h, &(lang.len() as u64).to_le_bytes());
                        write(&mut h, lang.as_bytes());
                    }
                    None => write(&mut h, &[0u8]),
                }
            }
        }
        h
    }
}

#[cfg(test)]
mod tests;
