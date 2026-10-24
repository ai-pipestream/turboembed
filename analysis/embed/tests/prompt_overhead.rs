//! Prompt overhead recording (#946): chunks budgeted for one
//! prompt role are refused by a vector step embedding with a
//! different overhead, on the real tiny BERT bundle.

use std::path::Path;
use std::sync::Arc;

use turbo::manifest::PromptRole;
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Chunks, Cut, Sentences, Vectors};

fn bundle() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn annotator(prompt: PromptRole) -> Arc<VectorAnnotator> {
    Arc::new(VectorAnnotator::open(&bundle(), &VectorOptions { prompt, ..VectorOptions::default() }).unwrap())
}

fn run(ann: Arc<VectorAnnotator>, recorded: u32) -> Result<(), Error> {
    let text = "Embedding models turn text into vectors. Similar passages land near each other.";
    let mut doc = Document::new(text);
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 41));
    s.push(Span::new(42, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();
    let mut c_spans = Spans::new(ViewId::SOURCE);
    c_spans.push(Span::new(0, text.len() as u32));
    doc.add(
        &Key::DEFAULT,
        Chunks {
            spans: c_spans,
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 2)],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![16],
            over_budget: vec![false],
            prompt_overhead: recorded,
        },
        Provenance::by("test"),
    )
    .unwrap();
    let a = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .add(SubwordAnnotator::open(&bundle()).unwrap())
        .add(ChunkVectors(ann))
        .build()
        .unwrap();
    let mut docs = [doc];
    a.run(&mut docs).remove(0)?;
    assert!(docs[0].get::<Vectors<Chunks>>().is_some());
    Ok(())
}

#[test]
fn mismatched_prompt_overhead_is_refused() {
    let ann = annotator(PromptRole::Query);
    let want = ann.overhead();
    // Chunked for the document role: a different recorded overhead.
    let err = run(ann.clone(), want + 5).unwrap_err();
    assert!(format!("{err}").contains("prompt overhead"), "{err}");
    // The same overhead the step embeds with, and unrecorded (0), pass.
    run(ann.clone(), want).unwrap();
    run(ann, 0).unwrap();
}
