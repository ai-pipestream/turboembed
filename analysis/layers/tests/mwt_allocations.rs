//! Allocation counts for the actual MWT lookup and its owned results.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use turbo_analysis::Span;
use turbo_analysis_layers::{MwtLanguage, RuleMwtExpander, TokenExpansion};
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
#[test]
fn misses_and_english_subspans_allocate_nothing() {
    for language in [
        MwtLanguage::Spanish,
        MwtLanguage::Catalan,
        MwtLanguage::French,
        MwtLanguage::Portuguese,
        MwtLanguage::Italian,
        MwtLanguage::German,
        MwtLanguage::English,
    ] {
        let expander = RuleMwtExpander::new(language);
        let before = COUNT.get();
        for _ in 0..1000 {
            for text in ["example", "EXAMPLE", "café", "CAFÉ", "İİİİİ", "ΣΣΣ", "", "longerthan10bytes"] {
                assert!(std::hint::black_box(expander.expand_token(text, Span::new(0, text.len() as u32))).is_none());
            }
        }
        assert_eq!(COUNT.get() - before, 0, "{language:?}");
    }
    let expander = RuleMwtExpander::new(MwtLanguage::English);
    let before = COUNT.get();
    for _ in 0..1000 {
        let result = expander.expand_token("can't", Span::new(0, 5)).unwrap();
        assert!(matches!(result, TokenExpansion::Subspans([Span { start: 0, end: 2 }, Span { start: 2, end: 5 }])));
    }
    assert_eq!(COUNT.get() - before, 0);
}

#[test]
fn successful_fusional_lookup_only_allocates_the_two_returned_forms() {
    let expander = RuleMwtExpander::new(MwtLanguage::Spanish);
    for text in ["del", "DEL"] {
        let before = COUNT.get();
        let TokenExpansion::Forms(forms) = expander.expand_token(text, Span::new(0, 3)).unwrap() else {
            panic!("forms")
        };
        let count = COUNT.get() - before;
        assert_eq!(count, 2);
        assert_eq!(forms, if text == "del" { ["de", "el"] } else { ["DE", "EL"] });
    }
    let german = RuleMwtExpander::new(MwtLanguage::German);
    let TokenExpansion::Forms(forms) = german.expand_token("FÜRS", Span::new(0, 5)).unwrap() else { panic!("forms") };
    assert_eq!(forms, ["FÜR", "DAS"]);
}
