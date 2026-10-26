//! Stored upstream reference fixtures exercise warmed caller buffers without allocation.
//! The reduced reference suites independently check the upstream numeric outputs.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use turbo::safetensors::File;
use turbo_relation::{Scratch, gliner2, glirel};

struct Counting;
thread_local! { static ALLOCS: Cell<usize> = const { Cell::new(0) }; }
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCS.with(Cell::get);
    f();
    ALLOCS.with(Cell::get) - before
}
fn fixture(model: &str) -> (Vec<u8>, serde_json::Value) {
    let dir = format!("{}/tests/fixtures/{model}", env!("CARGO_MANIFEST_DIR"));
    (
        std::fs::read(format!("{dir}/reduced.safetensors")).unwrap(),
        serde_json::from_slice(&std::fs::read(format!("{dir}/reduced.json")).unwrap()).unwrap(),
    )
}
fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn pairs(v: &serde_json::Value) -> Vec<(usize, usize)> {
    v.as_array().unwrap().iter().map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize)).collect()
}

#[test]
fn glirel_projection_chains_reuse_storage_and_clear_prior_states() {
    let (bytes, reference) = fixture("glirel");
    let heads = glirel::heads::Heads::load(&File::parse("heads", &bytes).unwrap(), 32).unwrap();
    let case = &reference["cases"][0];
    let projected = floats(&case["projected"]);
    let h = heads.width;
    let words: Vec<f32> = case["words"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|p| {
            let p = p.as_u64().unwrap() as usize;
            projected[p * h..(p + 1) * h].iter().copied()
        })
        .collect();
    let n = words.len() / h;
    let spans = pairs(&case["spans"]);
    let pairs = pairs(&case["pairs"]);
    let types = floats(&case["types"]);
    let expected_lstm = heads.lstm(&words, n);
    let expected_entities = heads.entities(&expected_lstm, &spans);
    let expected_logits = heads.logits(&expected_entities, &pairs, &types);
    assert!(
        allocations(|| {
            std::hint::black_box(heads.entities(&expected_lstm, &spans));
        }) > 0,
        "the compatibility API allocates its returned vector"
    );
    let mut scratch = Scratch::default();
    let (mut lstm, mut entities, mut logits) = (Vec::new(), Vec::new(), Vec::new());
    let mut run = || {
        // A long, empty, short, long sequence detects stale recurrent state and capacities.
        for len in [n, 0, 1, n] {
            heads.lstm_into(&words[..len * h], len, &mut scratch, &mut lstm);
        }
        assert_eq!(lstm, expected_lstm);
        heads.entities_into(&lstm, &spans, &mut scratch, &mut entities);
        assert_eq!(entities, expected_entities);
        heads.logits_into(&entities, &pairs, &types, &mut scratch, &mut logits);
        assert_eq!(logits, expected_logits);
        heads.entities_into(&lstm, &[], &mut scratch, &mut entities);
        assert!(entities.is_empty());
        heads.logits_into(&entities, &[], &types, &mut scratch, &mut logits);
        assert!(logits.is_empty());
    };
    run();
    run();
    assert_eq!(allocations(run), 0, "warmed real head execution must not allocate");
}

#[test]
fn gliner2_span_and_query_products_reuse_storage_including_tail_rows() {
    let (bytes, reference) = fixture("gliner2");
    let heads = gliner2::heads::Heads::load(&File::parse("heads", &bytes).unwrap(), 32, 8).unwrap();
    // The upstream fixture's actual query vectors are valid head-width inputs.
    let words = floats(&reference["cases"][0]["labels"][0]["queries"]);
    let n = words.len() / 32;
    assert!(n > 1);
    let queries = &words[..64];
    let expected: Vec<_> =
        [n, 1, 0, n].into_iter().map(|len| heads.logits_all(&heads.spans(&words[..len * 32], len), queries)).collect();
    let mut scratch = Scratch::default();
    let mut spans = gliner2::heads::Spans::default();
    let mut logits = Vec::new();
    let mut run = || {
        for (i, len) in [n, 1, 0, n].into_iter().enumerate() {
            heads.spans_into(&words[..len * 32], len, &mut scratch, &mut spans);
            heads.logits_all_into(&spans, queries, &mut scratch, &mut logits);
            assert_eq!(logits, expected[i]);
        }
    };
    run();
    run();
    assert_eq!(allocations(run), 0, "warmed span/query projections must not allocate");
}
