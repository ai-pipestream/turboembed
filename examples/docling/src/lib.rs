//! A document in docling's JSON form, analyzed end to end on the CPU:
//! the docling structure is imported over the source text, the text is
//! normalized into a `clean` view and the structure projected onto it,
//! then sentences, tokens, the embedding model's own subwords, entities,
//! sentence vectors, structural and drift boundaries weighed into one
//! cost, the chunker, and a vector per chunk.
//!
//! Every step reads what the one before wrote: one tokenization, one
//! subword pass, no text joined or copied after the source. Every span
//! maps back through `clean` to the source and from there to the docling
//! item and the offset in its `text` ([`docling::Docling::locate`]).
//! docs/integration.md walks through it; `tests/end_to_end.rs` holds it
//! to `data/report.expected.json`.

pub mod docling;
pub mod report;

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use turbo_analysis::{Analyzer, Bound, Document, Key, Provenance};
use turbo_analysis_embed::{
    ChunkVectors, EncoderOptions, SentenceVectors, SubwordAnnotator, VectorAnnotator, VectorOptions,
};
use turbo_analysis_layers::Structure;
use turbo_chunker::{Budget, Chunker, Costs, Drift, DriftBoundaries, STRUCTURE, StructureBoundaries, WeighBoundaries};
use turbo_namefind::dictionary::DictionaryNameFinder;
use turbo_normalize::{NormalizeStep, Whitespace};
use turbo_structure::StructureProjector;

/// The instance the imported docling structure is written to.
pub const SOURCE: &str = "source";
/// The normalized view every later step reads.
pub const CLEAN: &str = "clean";

/// Chunk sizes in the model's subwords, under the small test model's 64.
pub const BUDGET: Budget = Budget { max: 48, target: 32, size_weight: 1, overhead: 2 };

/// The pipeline over the embedding bundle at `bundle`, finding the names
/// in `places` as `location` entities.
pub fn pipeline(bundle: &Path, places: &[&str]) -> Result<Analyzer, String> {
    let options = VectorOptions {
        encoder: EncoderOptions { max_seq: 64, ..EncoderOptions::default() },
        ..VectorOptions::default()
    };
    let vectors = Arc::new(VectorAnnotator::open(bundle, &options).map_err(|e| e.to_string())?);
    let entries: Vec<Vec<String>> = places
        .iter()
        .map(|p| {
            let mut spans = Vec::new();
            turbo_tokenize::simple_spans(p, Default::default(), &mut spans);
            spans.iter().map(|s| p[s.range()].to_owned()).collect()
        })
        .collect();
    let dictionary = turbo_maxent::Dictionary::from_entries(false, entries.iter().map(Vec::as_slice));
    let drift = Drift { window: NonZeroU32::new(2).expect("not zero"), ..Drift::default() };
    Analyzer::builder()
        .given(Key::<Structure>::named(SOURCE))
        .step(Bound::new(NormalizeStep::new(Whitespace::keep_paragraphs())).write(0, CLEAN))
        .step(Bound::new(StructureProjector).read(0, CLEAN).read(1, SOURCE))
        .step(Bound::new(turbo_sentdetect::RuleSentenceDetector::new("en")).read(0, CLEAN))
        .step(Bound::new(turbo_tokenize::SimpleTokenizer::default()).read(0, CLEAN))
        .step(Bound::new(SubwordAnnotator::open(bundle).map_err(|e| e.to_string())?).read(0, CLEAN))
        .step(Bound::new(DictionaryNameFinder::new(dictionary, "location")).read(0, CLEAN))
        .add(SentenceVectors(Arc::clone(&vectors)))
        .step(Bound::new(StructureBoundaries { costs: Costs::default() }).write(0, STRUCTURE))
        .step(DriftBoundaries { drift }.bound())
        .step(WeighBoundaries::default().bound())
        .add(Chunker { budget: BUDGET })
        .add(ChunkVectors(vectors))
        .build()
        .map_err(|e| e.to_string())
}

/// Reads the docling JSON and runs `analyzer` over it.
pub fn analyze(analyzer: &Analyzer, json: &str) -> Result<(docling::Docling, Document), String> {
    let (d, structure) = docling::parse(json)?;
    let mut doc = Document::new(d.text.as_str());
    doc.add(&Key::named(SOURCE), structure, Provenance::by("docling:import")).map_err(|e| e.to_string())?;
    let mut docs = [doc];
    analyzer.run(&mut docs).remove(0).map_err(|e| e.to_string())?;
    let [doc] = docs;
    Ok((d, doc))
}
