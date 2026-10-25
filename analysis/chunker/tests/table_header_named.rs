use std::path::Path;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Analyzer, Bound, Document, Key, View};
use turbo_analysis_embed::SubwordAnnotator;
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{ChunkTableHeaders, Chunks, Tables};
use turbo_chunker::{Budget, Chunker, TableBoundaryAnnotator, TableHeaderAnnotator};
use turbo_normalize::{NormalizeStep, UnicodeForm};
use turbo_sentdetect::NewlineDetector;
use turbo_structure::TableGridAnnotator;
use turbo_tokenize::{SimpleTokenizer, TokenizerOptions};

#[test]
fn named_dependencies_and_normalized_source_survive_binary_graph_roundtrip() {
    let text = "| Cafe\u{301} | Total |\n| --- | ---: |\n| Montre\u{301}al | 12 |\n| Que\u{301}bec | 34 |\n";
    let pipeline = Analyzer::builder()
        .step(Bound::new(NormalizeStep::new(UnicodeForm::nfc())).write(0, "nfc"))
        .step(Bound::new(NewlineDetector::default()).read(0, "nfc").write(0, "rows"))
        .step(
            Bound::new(SimpleTokenizer::new(TokenizerOptions::default()))
                .read(0, "nfc")
                .read(1, "rows")
                .write(0, "words"),
        )
        .step(
            Bound::new(
                SubwordAnnotator::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle"))
                    .unwrap(),
            )
            .read(0, "nfc")
            .read(1, "words")
            .write(0, "pieces"),
        )
        .step(Bound::new(TableGridAnnotator::default()).read(0, "nfc").write(0, "grid"))
        .step(
            Bound::new(TableBoundaryAnnotator::default())
                .read(0, "nfc")
                .read(1, "rows")
                .read(2, "words")
                .read(3, "grid")
                .write(0, "row-cost"),
        )
        .step(
            Bound::new(Chunker { budget: Budget { max: 15, target: 15, size_weight: 1000, overhead: 0 } })
                .read(0, "rows")
                .read(1, "pieces")
                .read(2, "row-cost")
                .read(5, "grid")
                .write(0, "table-chunks"),
        )
        .step(
            Bound::new(TableHeaderAnnotator)
                .read(0, "nfc")
                .read(1, "table-chunks")
                .read(2, "grid")
                .write(0, "contexts"),
        )
        .build()
        .unwrap();
    let mut docs = [Document::new(text)];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let doc = &docs[0];
    let view = doc.get_key(&Key::<View>::named("nfc")).unwrap();
    let contexts = doc.get_key(&Key::<ChunkTableHeaders>::named("contexts")).unwrap();
    assert_eq!(contexts.spans.view(), view.id());
    assert!(contexts.contexts.len() > 1);
    assert_eq!(contexts.contexts[1].header_prefix.as_ref(), "| Café | Total |\n| --- | ---: |\n");
    assert_eq!(&view.text()[contexts.contexts[1].header_cells[0].range()], "Café");
    assert!(contexts.contexts[1].is_continuation);
    assert!(doc.get::<Tables>().is_none());
    assert!(doc.get::<Chunks>().is_none());
    let graph = export(doc, &codecs()).unwrap();
    let restored = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&restored, &codecs()).unwrap(), graph);
    assert_eq!(
        doc.provenance(Key::<ChunkTableHeaders>::named("contexts").id()),
        restored.provenance(Key::<ChunkTableHeaders>::named("contexts").id())
    );
}
