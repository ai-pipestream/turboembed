//! The kind table zero-shot extraction builds follows the mentions, not
//! the entity tag set: two mentions under a tag set of thousands of kinds
//! and 64 labels make no large allocation. A counting allocator records
//! the largest single request the extracting thread makes while `extract`
//! runs; other threads' requests are not its.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use turbo_analysis::{Document, IxRange, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::Entities;
use turbo_relation::{LexicalRelationScorer, RelationLabel, ZeroShotRelationAnnotator, ZeroShotRelationConfig};

struct Largest;

thread_local! {
    /// Whether this thread's requests are watched, and the largest one.
    static WATCHING: Cell<bool> = const { Cell::new(false) };
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

fn watch(layout: Layout) {
    if WATCHING.try_with(Cell::get).unwrap_or(false) {
        let _ = LARGEST.try_with(|l| l.set(l.get().max(layout.size())));
    }
}

unsafe impl GlobalAlloc for Largest {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        watch(layout);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        watch(layout);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Largest = Largest;

#[test]
fn two_mentions_under_thousands_of_kinds_allocate_little() {
    let doc = Document::new("Ann knows Bob.");
    let view = doc.get::<View>().unwrap();
    let labels: Vec<RelationLabel> = (0..64)
        .map(|i| RelationLabel { domain: Some(vec!["k0".into()]), ..RelationLabel::new(format!("knows {i}")) })
        .collect();
    let a = ZeroShotRelationAnnotator::new(ZeroShotRelationConfig::new(labels), LexicalRelationScorer).unwrap();
    for kinds in [512, 4096] {
        let names: Vec<String> = (0..kinds).map(|k| format!("k{k}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, 3));
        spans.push(Span::new(10, 13));
        let e = Entities {
            spans,
            tokens: vec![IxRange::new(0, 1), IxRange::new(2, 3)],
            kind: vec![Tag(0), Tag(0)],
            kinds: Arc::clone(&TagSet::new("kinds", &names)),
            score: vec![1.0; 2],
        };
        LARGEST.set(0);
        WATCHING.set(true);
        let out = a.extract(view, &e, None);
        WATCHING.set(false);
        let (relations, _) = out.unwrap();
        // "knows" is between Ann and Bob, half of each label's words.
        assert_eq!(relations.subject.len(), 128, "{kinds} kinds");
        let largest = LARGEST.get();
        assert!(largest < 16 * 1024, "{kinds} kinds: a {largest}-byte allocation");
    }
}
