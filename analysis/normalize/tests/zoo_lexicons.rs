//! Spell correction and noise scoring from the zoo's frequency lexicons
//! (#935): SymSpell's English unigram and bigram lists, fetched by zoo id
//! and checked by digest, drive `SpellCorrect`, `lookup_compound`,
//! `ContextSpellCorrector` and the noise scorer's `Misspelled` severity on
//! real text. `accuracy` measures them on SymSpell's own misspelling set.
//!
//! The lists are MIT (docs/licences.md, "Frequency lexicons") and are
//! fetched into the zoo's home (`TURBO_ZOO_HOME`), so the tests are
//! ignored by default. Run them with
//! `cargo test --release -p turbo-normalize --test zoo_lexicons -- --ignored --nocapture`.

mod folds_support;

use std::sync::{Arc, OnceLock};

use folds_support::check_alignment;
use turbo_analysis::{Annotator, Document, Layer, Refs, View};
use turbo_ngram::NGramLanguageModel;
use turbo_normalize::spellcheck::{
    EN_BIGRAMS, EN_UNIGRAMS, Lexicon, SpellCorrect, SymSpell, SymSpellConfig, language_model,
};
use turbo_normalize::{ContextSpellCorrector, NoiseScorer, normalize};
use turbo_normalize_layers::NoiseSeverity;
use turbo_zoo::{Acceptances, Home, Http, Index, Policy, UserId, Zoo};

#[path = "zoo_lexicons/accuracy.rs"]
mod accuracy;
#[path = "zoo_lexicons/codespell.rs"]
mod codespell;

/// Runs `f` with the zoo in the home the environment names.
fn with_zoo<T>(f: impl FnOnce(&Zoo<'_>, &UserId, &Home) -> T) -> T {
    let home = Home::from_env().expect("a zoo home");
    let zoo = Zoo::new(Index::builtin(), home.artifacts(), Policy::default(), &Http);
    let user = UserId::from_env().unwrap_or_else(|_| "test".parse().unwrap());
    f(&zoo, &user, &home)
}

/// The lexicon `id`, fetched (or read from the cache) and verified.
pub(crate) fn lexicon(id: &str) -> Lexicon {
    with_zoo(|zoo, user, home| {
        Lexicon::fetch(zoo, id, user, || Acceptances::load(&home.acceptances())).unwrap_or_else(|e| panic!("{id}: {e}"))
    })
}

/// The text of the one-file dataset `id`, fetched and verified.
pub(crate) fn dataset(id: &str) -> String {
    with_zoo(|zoo, user, home| {
        let fetched = zoo
            .fetch_with(id, user, || Acceptances::load(&home.acceptances()))
            .unwrap_or_else(|e| panic!("{id}: {e}; a share-alike set needs `turbo-zoo accept` first"));
        std::fs::read_to_string(fetched.single().unwrap()).unwrap()
    })
}

/// The engine over both English lists, built once.
pub(crate) fn engine() -> Arc<SymSpell> {
    static ENGINE: OnceLock<Arc<SymSpell>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let (uni, bi) = (lexicon(EN_UNIGRAMS), lexicon(EN_BIGRAMS));
            Arc::new(SymSpell::from_lexicons(SymSpellConfig::default(), &uni, Some(&bi)).unwrap())
        })
        .clone()
}

/// The bigram language model over both English lists, built once.
pub(crate) fn lm() -> Arc<NGramLanguageModel> {
    static LM: OnceLock<Arc<NGramLanguageModel>> = OnceLock::new();
    LM.get_or_init(|| Arc::new(language_model(&lexicon(EN_UNIGRAMS), &lexicon(EN_BIGRAMS)).unwrap())).clone()
}

/// Whitespace kept, or a stretch with text replaced by another.
fn edit_ok(parent: &str, child: &str) -> bool {
    !parent.trim().is_empty() && !child.is_empty() || parent.trim().is_empty() && (child.is_empty() || child == " ")
}

#[test]
#[ignore = "fetches the SymSpell lexicons through the zoo"]
fn the_lists_load_by_zoo_id() {
    let (uni, bi) = (lexicon(EN_UNIGRAMS), lexicon(EN_BIGRAMS));
    assert_eq!((uni.id(), bi.id()), (EN_UNIGRAMS, EN_BIGRAMS));
    let e = engine();
    // 82,765 in the file's name; the file has 82,834 lines.
    assert_eq!(e.word_count(), 82_834);
    assert_eq!(e.bigram_count(), 242_342);
    assert_eq!(e.count("the"), Some(23_135_851_162));
    assert_eq!(e.bigram("of", "the"), Some(177_045_273_024));
}

#[test]
#[ignore = "fetches the SymSpell lexicons through the zoo"]
fn spell_correct_fixes_real_typos_with_exact_alignment() {
    let n = SpellCorrect::new(engine());
    let src = "I beleive the goverment will recieve teh report tomorow.";
    let (out, a) = normalize(&n, src);
    // "teh" is shorter than the default minimum of four, so it is kept.
    assert_eq!(out, "I believe the government will receive teh report tomorrow.");
    check_alignment(src, &out, &a, edit_ok);
    let doc = Document::new(src);
    let view = doc.get::<View>().unwrap();
    let input: Refs<'_, SpellCorrect> = (view, None);
    let (corrected, corrections) = n.annotate(input).unwrap();
    assert_eq!(corrected.text(), out);
    corrections.check().unwrap();
    for (ix, s) in corrections.spans.iter() {
        let child = corrected.alignment().span_to_child(s);
        assert_eq!(&corrected.text()[child.range()], corrections.suggestion.get(ix.get()));
        assert_eq!(corrected.alignment().span_to_parent(child), s, "{:?} maps back", &src[s.range()]);
    }
}

#[test]
#[ignore = "fetches the SymSpell lexicons through the zoo"]
fn compound_lookup_uses_the_bigrams() {
    // The example in SymSpell's README. The README shows "sixth grade";
    // "six grade" is one edit from "sixtgrade" and "sixth grade" two, and
    // the split with fewer edits wins here as in OpenNLP's port.
    let input = "whereis th elove hehad dated forImuch of thepast who couqdn'tread in sixtgrade and ins pired him";
    let c = engine().lookup_compound(input, 2).unwrap();
    assert_eq!(
        c.term,
        "where is the love he had dated for much of the past who couldn't read in six grade and inspired him"
    );
    // Each part maps back to the input it replaces.
    let merged = c.parts.iter().find(|p| &c.term[p.output.clone()] == "inspired").unwrap();
    assert_eq!(&input[merged.source.clone()], "ins pired");
    let split = c.parts.iter().find(|p| &input[p.source.clone()] == "whereis").unwrap();
    assert_eq!(&c.term[split.output.clone()], "where is");
}

#[test]
#[ignore = "fetches the SymSpell lexicons through the zoo"]
fn the_context_corrector_runs_on_the_zoo_language_model() {
    let n = ContextSpellCorrector::new(engine(), lm());
    for (src, want) in [
        ("I want to by a new car.", "I want to buy a new car."),
        ("The cat sat on teh mat.", "The cat sat on the mat."),
        // "too the" becomes "to the". "Their going" is left: the bigram
        // list has no pair "they're" starts and no "their going" or
        // "there going", so the most frequent of the three stays.
        ("Their going too the store.", "Their going to the store."),
    ] {
        let (out, a) = normalize(&n, src);
        assert_eq!(out, want, "{src}");
        check_alignment(src, &out, &a, edit_ok);
    }
}

#[test]
#[ignore = "fetches the SymSpell lexicons through the zoo"]
fn the_noise_scorer_reports_misspelled_words() {
    let scorer = NoiseScorer::with_dictionary(engine());
    let text = "The rnodern tirnes of the vvorld are here.";
    let found = scorer.score(text, &[]).unwrap();
    let got: Vec<(&str, NoiseSeverity)> = found.iter().map(|f| (&text[f.span.range()], f.severity)).collect();
    assert_eq!(got, [("rnodern tirnes", NoiseSeverity::Misspelled), ("vvorld", NoiseSeverity::Misspelled)]);
    // Correct text has no findings.
    assert_eq!(scorer.score("The modern times of the world are here.", &[]).unwrap(), vec![]);
}
