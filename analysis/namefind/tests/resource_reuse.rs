//! Parsed dictionary resources must not be rebuilt for each document.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
struct Counting;
thread_local! { static COUNT: Cell<usize> = const { Cell::new(0) }; }
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

use std::sync::Arc;
use turbo_analysis::{Annotator, Document, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_maxent::{JavaString, ModelFile};
use turbo_namefind::CrfNameAnnotator;
use turbo_sequence::{
    crf::CrfSequenceModel,
    namefind::{CODEC_PROPERTY, NameCodec, crf},
};
use turbo_train::{Manifest, ModelPackage};

pub fn package(extra_entries: usize) -> Arc<ModelFile> {
    let mut weights = vec![0.0; 10];
    weights[1] = 5.0;
    let crf = CrfSequenceModel::new(
        vec!["other".into(), "person-unit".into()],
        vec![JavaString::new(":w=dic")],
        NameCodec::Bilou,
        weights,
    )
    .unwrap();
    let mut manifest = Manifest::new("TokenNameFinder", "en", 1700000000000);
    manifest.set(CODEC_PROPERTY, NameCodec::Bilou.class_name());
    manifest.set("Model-Type", crf::MODEL_TYPE);
    let mut package = ModelPackage::new(manifest);
    package.add_entry(crf::MODEL_ENTRY, crf.to_bytes());
    package.add_entry("generator.featuregen", br#"<featureGenerators name="nameFinder"><generator class="opennlp.tools.util.featuregen.DictionaryFeatureGeneratorFactory"><str name="dict">names.dictionary</str></generator></featureGenerators>"#.to_vec());
    let mut dictionary = r#"<dictionary case_sensitive="true"><entry><token>Alice</token></entry>"#.to_owned();
    for i in 0..extra_entries {
        dictionary.push_str(&format!("<entry><token>Extra{i}</token></entry>"));
    }
    dictionary.push_str("</dictionary>");
    package.add_entry("names.dictionary", dictionary.into_bytes());
    Arc::new(ModelFile::open(package.to_bytes()).unwrap())
}

fn allocations(entries: usize) -> usize {
    let file = package(entries);
    let crf = CrfNameAnnotator::new((*file).clone(), 1).unwrap();
    let doc = Document::new("Alice");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5));
    let sentences = Sentences { spans };
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5));
    let tokens = Tokens { spans, sentence: None };
    let view = doc.get::<View>().unwrap();
    // Warm the path so only repeated document work is counted.
    assert_eq!(crf.annotate((view, &sentences, &tokens)).unwrap().0.len(), 1);
    let before = COUNT.get();
    for _ in 0..10 {
        assert_eq!(crf.annotate((view, &sentences, &tokens)).unwrap().0.len(), 1);
    }
    COUNT.get() - before
}

#[test]
fn repeated_documents_do_not_allocate_per_resource_entry() {
    let small = allocations(0);
    let large = allocations(2000);
    assert!(large <= small + 10, "CRF resource allocation grew: {small} -> {large}");
}
