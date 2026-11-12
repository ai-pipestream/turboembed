//! GLiREL in process, for `--glirel`. No serving step runs it, so the same
//! zero-shot relation annotator the served GLiNER2 step runs is given a
//! GLiREL scorer and run over the same initial graph: the same mentions,
//! the same labels and threshold, and at sentence scope the same
//! `Sentences` and `within_sentence`.

use std::path::Path;
use std::sync::Arc;

use turbo_analysis::{Analyzer, Document, Key};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_relation::{GlirelRelations, RelationLabel, Relations, ZeroShotRelationAnnotator, ZeroShotRelationConfig};

use super::{Fail, SUBSET};

/// The analyzer, and the scorer's model digest for the receipt.
pub fn analyzer(package: &Path, spm: &Path, sentences: bool, threshold: f32) -> Result<(Analyzer, String), Fail> {
    // A record's model is a pinned one (analysis/pins/models.txt).
    let model = Arc::new(GlirelRelations::load(&turbo_pins::require(package), spm)?);
    let digest = model.model().sha256.clone();
    let labels = SUBSET.iter().map(|(_, name)| RelationLabel::new(*name)).collect();
    let config = ZeroShotRelationConfig {
        threshold: threshold.into(),
        within_sentence: sentences,
        ..ZeroShotRelationConfig::new(labels)
    };
    let mut builder = Analyzer::builder().given(Key::<Tokens>::DEFAULT).given(Key::<Entities>::DEFAULT);
    if sentences {
        builder = builder.given(Key::<Sentences>::DEFAULT);
    }
    let analyzer = builder.add(ZeroShotRelationAnnotator::new(config, model)?).build()?;
    Ok((analyzer, digest))
}

/// `doc` with the scorer's relations, or the error that refused it.
pub fn run(analyzer: &Analyzer, doc: Document) -> Result<Document, Fail> {
    let mut docs = vec![doc];
    for result in analyzer.run(&mut docs) {
        result?;
    }
    let doc = docs.pop().ok_or("no document")?;
    doc.get::<Relations>().ok_or("no relations")?;
    Ok(doc)
}
