//! Built-in first-party [`turbo_analysis::Annotator`] step registration for
//! `AnalyzeDocumentProgressive` (issue `#1218`). Every step registered here is
//! a real workspace annotator from `turbo-structure`, `turbo-sentdetect`,
//! `turbo-tokenize`, `turbo-abbrev`, `turbo-stem`, `turbo-stopword`,
//! `turbo-keywords`, or `turbo-chunker`.

use turbo_abbrev::AbbreviationAnnotator;
use turbo_analysis::analyzer::Builder;
use turbo_analysis::{Analyzer, Document, Key};
use turbo_analysis_layers::{Boundaries, Levels, Sentences, Stems, Structure, Terms, Tokens};
use turbo_chunker::terms::TermIndexer;
use turbo_chunker::{Costs, LexicalBoundaries, StructureBoundaries};
use turbo_keywords::{Stopwords as KeywordStopwords, YakeAnnotator};
use turbo_sentdetect::RuleSentenceDetector;
use turbo_stem::{Algorithm, StemAnnotator, TermAnnotator};
use turbo_stopword::{ListChoice, StopwordAnnotator, StopwordLanguage};
use turbo_stopword_layers::Stopwords;
use turbo_structure::{StructureAnnotator, Syntax};
use turbo_tokenize::uax29::UntypedWordTokenizer;

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};

pub(super) const DEFAULT_STEPS: &[&str] = &[
    "turbo-structure:blocks",
    "turbo-sentdetect:rules",
    "turbo-tokenize:uax29-untyped",
    "turbo-abbrev:schwartz-hearst",
    "turbo-stem:terms",
    "turbo-stem:stemmer",
    "turbo-stopword:bundled",
    "turbo-keywords:yake",
    "turbo:structure-boundaries",
    "turbo:lexical-boundaries",
];

pub(super) const MINIMAL_STEPS: &[&str] = &["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped"];

pub(super) const CHUNKING_STEPS: &[&str] = &[
    "turbo-structure:blocks",
    "turbo-sentdetect:rules",
    "turbo-tokenize:uax29-untyped",
    "turbo-stem:terms",
    "turbo-stopword:bundled",
    "turbo:structure-boundaries",
    "turbo:lexical-boundaries",
];

fn bad(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg)
}

/// Registers any layers already present on `initial_doc` as `given` on `builder`.
pub(super) fn register_given_layers(mut builder: Builder, initial_doc: &Document) -> Builder {
    if initial_doc.present(Key::<Structure>::DEFAULT.id()) {
        builder = builder.given(Key::<Structure>::DEFAULT);
    }
    if initial_doc.present(Key::<Sentences>::DEFAULT.id()) {
        builder = builder.given(Key::<Sentences>::DEFAULT);
    }
    if initial_doc.present(Key::<Tokens>::DEFAULT.id()) {
        builder = builder.given(Key::<Tokens>::DEFAULT);
    }
    if initial_doc.present(Key::<Terms>::DEFAULT.id()) {
        builder = builder.given(Key::<Terms>::DEFAULT);
    }
    if initial_doc.present(Key::<Stems>::DEFAULT.id()) {
        builder = builder.given(Key::<Stems>::DEFAULT);
    }
    if initial_doc.present(Key::<Stopwords>::DEFAULT.id()) {
        builder = builder.given(Key::<Stopwords>::DEFAULT);
    }
    if initial_doc.present(Key::<Boundaries>::DEFAULT.id()) {
        builder = builder.given(Key::<Boundaries>::DEFAULT);
    }
    if initial_doc.present(Key::<Levels>::DEFAULT.id()) {
        builder = builder.given(Key::<Levels>::DEFAULT);
    }
    builder
}

/// Adds a named first-party annotator step to `builder`.
pub(super) fn add_step(builder: Builder, name: &str) -> Result<Builder, Failure> {
    let next = match name {
        "turbo-structure:blocks" | "turbo-structure:structure" | "turbo:structure" => {
            builder.add(StructureAnnotator::new(Syntax::Markdown))
        }
        "turbo-sentdetect:rules" | "turbo:sentences" | "opennlp:sentences" => {
            builder.add(RuleSentenceDetector::new("eng"))
        }
        "turbo-tokenize:uax29-untyped" | "turbo:tokens" | "opennlp:tokens" => {
            builder.add(UntypedWordTokenizer::default())
        }
        "turbo-abbrev:schwartz-hearst" | "turbo-abbrev:abbreviations" | "turbo:abbreviations" => {
            builder.add(AbbreviationAnnotator)
        }
        "turbo-stem:terms" | "turbo:terms" => builder.add(TermAnnotator::default()),
        "turbo-stem:stemmer" | "turbo:stems" => builder.add(StemAnnotator::new(Algorithm::Porter)),
        "turbo-stopword:bundled" | "turbo-stopword:stopwords" | "turbo:stopwords" => {
            builder.add(StopwordAnnotator::new(ListChoice::Fixed(StopwordLanguage::English)))
        }
        "turbo-keywords:yake" | "turbo:keywords" => {
            builder.add(YakeAnnotator::new(KeywordStopwords::bundled(StopwordLanguage::English)))
        }
        "turbo:structure-boundaries" | "turbo:boundaries" => {
            builder.add(StructureBoundaries { costs: Costs::default() })
        }
        "turbo:lexical-boundaries" => builder.step(LexicalBoundaries::default().bound()),
        "turbo:term-indexer" | "turbo:term-index" => builder.add(TermIndexer),
        unknown => return Err(bad(format!("unknown pipeline step '{unknown}'"))),
    };
    Ok(next)
}

/// Builds an [`Analyzer`] from `initial_doc` and the requested `step_names`.
pub fn compile_analyzer(initial_doc: &Document, step_names: &[&str]) -> Result<Analyzer, Failure> {
    let mut builder = register_given_layers(Analyzer::builder(), initial_doc);
    for &name in step_names {
        builder = add_step(builder, name)?;
    }
    builder.build().map_err(|e| bad(format!("invalid pipeline DAG: {e}")))
}
