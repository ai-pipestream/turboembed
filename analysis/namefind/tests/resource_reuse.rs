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
use turbo_calibration::{Calibration, DataIdentity, Policy, Transform, digest};
use turbo_maxent::{JavaString, ModelFile};
use turbo_namefind::{
    CrfNameAnnotator,
    calibrated::{CalibratedNameAnnotator, identity},
};
use turbo_sequence::{
    crf::CrfSequenceModel,
    namefind::{CODEC_PROPERTY, MODEL_ENTRY_NAME, NameCodec, crf},
};
use turbo_train::{Event, Key, Manifest, ModelPackage, TrainingConfiguration, TrainingParameters};

fn package(extra_entries: usize) -> Arc<ModelFile> {
    let events = [Event::from_strs("unit", &[":w=dic"]), Event::from_strs("other", &["missing"])];
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 50);
    params.put(Key::CUTOFF, 0);
    let (model, _) = turbo_train::gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    let mut manifest = Manifest::new("TokenNameFinder", "en", 1700000000000);
    manifest.set(CODEC_PROPERTY, NameCodec::Bilou.class_name());
    manifest.set("Model-Type", crf::MODEL_TYPE);
    let mut package = ModelPackage::new(manifest);
    package.add_entry(MODEL_ENTRY_NAME, turbo_train::write_gis(&model).unwrap());
    let mut weights = vec![0.0; 10];
    weights[1] = 5.0;
    let crf = CrfSequenceModel::new(
        vec!["other".into(), "unit".into()],
        vec![JavaString::new(":w=dic")],
        NameCodec::Bilou,
        weights,
    )
    .unwrap();
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

fn allocations(entries: usize) -> (usize, usize) {
    let file = package(entries);
    let crf = CrfNameAnnotator::new((*file).clone(), 1).unwrap();
    let artifact = Calibration::new(
        identity(&file, "test").unwrap(),
        Transform::Temperature(1.0),
        DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
        DataIdentity { split: "conf".into(), sha256: digest(b"conf") },
        vec![0.1; 20],
    )
    .unwrap();
    let calibrated = CalibratedNameAnnotator::new(
        file,
        artifact,
        "test",
        Policy { coverage: 0.9, max_set_size: 1, min_confidence: 0.0 },
    )
    .unwrap();
    let doc = Document::new("Alice");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5));
    let sentences = Sentences { spans };
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5));
    let tokens = Tokens { spans, sentence: None };
    let view = doc.get::<View>().unwrap();
    // Warm both paths so only repeated document work is counted.
    assert_eq!(crf.annotate((view, &sentences, &tokens)).unwrap().0.len(), 1);
    assert_eq!(calibrated.annotate((view, &sentences, &tokens)).unwrap().0.len(), 1);
    let before = COUNT.get();
    for _ in 0..10 {
        assert_eq!(crf.annotate((view, &sentences, &tokens)).unwrap().0.len(), 1);
    }
    let crf_count = COUNT.get() - before;
    let before = COUNT.get();
    for _ in 0..10 {
        assert_eq!(calibrated.annotate((view, &sentences, &tokens)).unwrap().0.len(), 1);
    }
    (crf_count, COUNT.get() - before)
}

#[test]
fn repeated_documents_do_not_allocate_per_resource_entry() {
    let small = allocations(0);
    let large = allocations(2000);
    assert!(large.0 <= small.0 + 10, "CRF resource allocation grew: {small:?} -> {large:?}");
    assert!(large.1 <= small.1 + 10, "calibrated resource allocation grew: {small:?} -> {large:?}");
}
