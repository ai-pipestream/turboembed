//! The measurement kernel reuses its allocations; owned output layers are
//! deliberately outside this contract.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use turbo_analysis::{Analyzer, Bound, Document, View};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_normalize_layers::{QualityScratch, TextQualityAnnotator};
use turbo_sentdetect::RuleSentenceDetector;
use turbo_tokenize::SimpleTokenizer;

struct Counting;
thread_local! { static ALLOCS: Cell<usize> = const { Cell::new(0) }; }
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn warmed_measurement_reuses_all_hash_tables_arenas_and_summary_buffers() {
    let text = "A careful reader can use this passage to learn about clear writing. Each sentence presents a useful idea and gives enough context to understand the entire explanation.\n\nHome | About | Contact | Privacy\n\nÉté e\u{301}té 東京 inter-\nnational con-\ntent.\n\n".repeat(64);
    let analyzer = Analyzer::builder()
        .step(Bound::new(RuleSentenceDetector::new("en")))
        .step(Bound::new(SimpleTokenizer::default()))
        .build()
        .unwrap();
    let mut docs = [Document::new(text)];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let (view, tokens, sentences) =
        (docs[0].get::<View>().unwrap(), docs[0].get::<Tokens>().unwrap(), docs[0].get::<Sentences>().unwrap());
    let mut scratch = QualityScratch::default();
    let expected = TextQualityAnnotator.measure(view, tokens, sentences, None, &mut scratch).unwrap().0;
    let before = ALLOCS.get();
    for _ in 0..32 {
        let (m, rows) = TextQualityAnnotator.measure(view, tokens, sentences, None, &mut scratch).unwrap();
        assert_eq!(m, expected);
        assert_eq!(rows.len(), 192);
        std::hint::black_box((m, rows));
    }
    assert_eq!(ALLOCS.get() - before, 0);
}
