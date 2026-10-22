use std::path::Path;

use turbo_analysis_embed::{Device, Precision, VectorOptions};
use turbo_chunker::{Budget, Costs};
use turbo_structure::Syntax;

use super::*;
use crate::analysis::{Analysis, Sentencer};
use crate::chunkers::Signals;
use crate::chunkers::tests::{analyzed, texts};
use crate::embedder::TurboEmbedder;
use crate::embedder::stub::Stub;

/// A line before the first heading, a section with a nested one, and a
/// second section.
const TEXT: &str =
    "cat one two\n\n# Alpha\n\ncat a b\ncat c d\n\n## Inner\n\ndog e f\n\n# Beta\n\nrust g h\nrust i j\n";

fn leveled(level: Retrieve, vectors: Option<(Arc<VectorAnnotator>, VectorKind)>) -> LevelChunker {
    // At most two of these sentences in a chunk.
    let chunker = TurboChunker {
        budget: Budget { max: 9, target: 8, size_weight: 1000, overhead: 2 },
        costs: Costs::default(),
        signals: Signals::Structure,
        cohesion: None,
    };
    LevelChunker { chunker, level, vectors }
}

#[test]
fn each_level_retrieves_its_units() {
    let mut e = Stub::new(&["cat", "dog", "rust"]);
    let doc = analyzed(TEXT, &e, Some(Syntax::Markdown));
    let at = |l, e: &mut Stub| leveled(l, None).chunk(&doc, e).unwrap();
    assert_eq!(
        texts(TEXT, &at(Retrieve::Chunk, &mut e)),
        ["cat one two", "# Alpha\n\ncat a b", "cat c d", "## Inner\n\ndog e f", "# Beta\n\nrust g h", "rust i j"]
    );
    // Both sections, the nested one too, and the line outside them.
    assert_eq!(
        texts(TEXT, &at(Retrieve::Section, &mut e)),
        [
            "cat one two",
            "# Alpha\n\ncat a b\ncat c d\n\n## Inner\n\ndog e f",
            "## Inner\n\ndog e f",
            "# Beta\n\nrust g h\nrust i j"
        ]
    );
    assert_eq!(texts(TEXT, &at(Retrieve::Document, &mut e)), [TEXT]);
    // With no structure there are no sections: the chunks are the units.
    let plain = analyzed("cat a b\ncat c d\ndog e f\n", &e, None);
    let c = leveled(Retrieve::Chunk, None).chunk(&plain, &mut e).unwrap();
    assert_eq!(leveled(Retrieve::Section, None).chunk(&plain, &mut e).unwrap(), c);
    // The embedded text is left to the pipeline.
    assert!(leveled(Retrieve::Section, None).chunk_vectors(&doc, &mut e).unwrap().1.is_none());
    let config = leveled(Retrieve::Section, None).config();
    assert!(config.contains(&setting("level", "section")));
    assert!(config.contains(&setting("level_vectors", "each unit's text embedded as one row")));
}

/// The small BERT in testdata, as the pipeline's embedder and as the
/// pooled vectors' model.
fn tiny() -> (TurboEmbedder, Arc<VectorAnnotator>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let e = TurboEmbedder::open(&dir, Device::Cpu, Precision::Model).unwrap();
    (e, Arc::new(VectorAnnotator::open(&dir, &VectorOptions::default()).unwrap()))
}

#[test]
fn pooled_vectors_are_one_per_unit_of_the_kind_asked() {
    let (mut e, a) = tiny();
    let doc = Analysis { sentences: Sentencer::Newline, structure: Some(Syntax::Markdown) }.analyze(TEXT, &e).unwrap();
    let pooled = |l, k, e: &mut TurboEmbedder| leveled(l, Some((a.clone(), k))).chunk_vectors(&doc, e).unwrap();
    let dim = 32;
    for l in [Retrieve::Chunk, Retrieve::Section, Retrieve::Document] {
        let (local_spans, local) = pooled(l, VectorKind::Local, &mut e);
        let (spans, contextual) = pooled(l, VectorKind::Contextual, &mut e);
        assert_eq!(spans, local_spans);
        assert_eq!(spans, leveled(l, None).chunk(&doc, &mut e).unwrap());
        let (local, contextual) = (local.unwrap(), contextual.unwrap());
        assert_eq!((local.len(), contextual.len()), (spans.len() * dim, spans.len() * dim));
        for v in local.chunks(dim).chain(contextual.chunks(dim)) {
            let norm: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            assert!((norm - 1.0).abs() < 1e-5, "{norm}");
        }
        // The document has no parent: its contextual vector is its local
        // one. A chunk inside a section is pooled with the section's
        // other text around it, so its two vectors differ.
        match l {
            Retrieve::Document => assert_eq!(local, contextual),
            _ => assert_ne!(local, contextual),
        }
    }
    let config = leveled(Retrieve::Chunk, Some((a, VectorKind::Contextual))).config();
    assert!(config.contains(&setting("level_vectors", "contextual, pooled by LevelVectors")));
}

#[test]
fn the_pipeline_retrieves_by_the_pooled_vectors() {
    use crate::lexical::{Lexical, TermRule};
    use crate::pipeline::tests::set;
    use crate::pipeline::{Pipeline, Titles};
    use turbo_chunker::terms::Bm25;

    let run = |level, kind, lexical: Option<Lexical>| {
        let (e, a) = tiny();
        let mut p = Pipeline {
            analysis: Analysis { sentences: Sentencer::Newline, structure: Some(Syntax::Markdown) },
            chunker: Box::new(leveled(level, Some((a, kind)))),
            embedder: Box::new(e),
            titles: Titles::Ignore,
            lexical,
            hybrid: None,
        };
        p.evaluate(&set(None))
    };
    // A document's two vectors are one, so its two runs rank alike; at
    // the chunk level they differ.
    let local = run(Retrieve::Document, VectorKind::Local, None).unwrap();
    assert_eq!(local.run, run(Retrieve::Document, VectorKind::Contextual, None).unwrap().run);
    assert_eq!(local.cost.chunks, 3);
    let chunks = run(Retrieve::Chunk, VectorKind::Contextual, None).unwrap();
    assert_ne!(chunks.run, run(Retrieve::Chunk, VectorKind::Local, None).unwrap().run);
    // Only the queries were embedded by the pipeline: two tokens of
    // overhead and a piece or two each.
    assert!(chunks.cost.tokens_embedded < 20, "{}", chunks.cost.tokens_embedded);
    let bm25 = Lexical { model: Bm25::LUCENE, terms: TermRule::Folded, dump: None };
    let e = run(Retrieve::Chunk, VectorKind::Local, Some(bm25)).unwrap_err();
    assert!(e.contains("dense retrieval only"), "{e}");
}
