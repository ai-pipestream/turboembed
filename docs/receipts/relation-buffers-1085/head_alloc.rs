//! Standalone release harness: compile with --cfg reuse for caller buffers.
use std::hint::black_box;
use std::time::Instant;
use turbo::safetensors::File;
use turbo_relation::glirel::heads::Heads;

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn pairs(v: &serde_json::Value) -> Vec<(usize, usize)> {
    v.as_array().unwrap().iter().map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize)).collect()
}
fn main() {
    let dir = std::env::args().nth(1).expect("fixture directory");
    if dir.ends_with("gliner2") {
        return gliner2(&dir);
    }
    let bytes = std::fs::read(format!("{dir}/reduced.safetensors")).unwrap();
    let heads = Heads::load(&File::parse("reference fixture", &bytes).unwrap(), 32).unwrap();
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{dir}/reduced.json")).unwrap()).unwrap();
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
    #[cfg(reuse)]
    let mut scratch = turbo_relation::Scratch::default();
    #[cfg(reuse)]
    let (mut lstm, mut entities, mut logits) = (Vec::new(), Vec::new(), Vec::new());
    #[allow(unused_mut)] // Only the reuse variant mutates caller buffers.
    let mut run = || {
        #[cfg(reuse)]
        {
            heads.lstm_into(black_box(&words), n, &mut scratch, &mut lstm);
            heads.entities_into(&lstm, &spans, &mut scratch, &mut entities);
            heads.logits_into(&entities, &pairs, &types, &mut scratch, &mut logits);
            black_box(logits.iter().map(|&x| f64::from(x)).sum::<f64>())
        }
        #[cfg(not(reuse))]
        {
            let lstm = heads.lstm(black_box(&words), n);
            let entities = heads.entities(&lstm, &spans);
            let logits = heads.logits(&entities, &pairs, &types);
            black_box(logits.iter().map(|&x| f64::from(x)).sum::<f64>())
        }
    };
    for _ in 0..100 {
        black_box(run());
    }
    ALLOCS.store(0, std::sync::atomic::Ordering::Relaxed);
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..1 {
        checksum += run();
    }
    let allocated = ALLOCS.load(std::sync::atomic::Ordering::Relaxed);
    println!("{:.9} {checksum:.12}", start.elapsed().as_secs_f64());
    println!("allocations {}", allocated);
}

fn gliner2(dir: &str) {
    let bytes = std::fs::read(format!("{dir}/reduced.safetensors")).unwrap();
    let heads =
        turbo_relation::gliner2::heads::Heads::load(&File::parse("reference fixture", &bytes).unwrap(), 32, 8).unwrap();
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{dir}/reduced.json")).unwrap()).unwrap();
    let words = floats(&reference["cases"][0]["labels"][0]["queries"]);
    let n = words.len() / 32;
    let k = reference["fixed_count"].as_u64().unwrap() as usize;
    #[cfg(reuse)]
    let mut scratch = turbo_relation::Scratch::default();
    #[cfg(reuse)]
    let (mut spans, mut queries, mut logits) =
        (turbo_relation::gliner2::heads::Spans::default(), Vec::new(), Vec::new());
    #[allow(unused_mut)] // Only the reuse variant mutates caller buffers.
    let mut run = || {
        black_box(heads.count(&words[..32]));
        #[cfg(reuse)]
        {
            heads.spans_into(black_box(&words), n, &mut scratch, &mut spans);
            heads.queries_of_into(&[&words[..32], &words[32..64]], k, &mut scratch, &mut queries);
            heads.logits_all_into(&spans, &queries, &mut scratch, &mut logits);
            black_box(logits.iter().filter(|x| x.is_finite()).map(|&x| f64::from(x)).sum::<f64>())
        }
        #[cfg(not(reuse))]
        {
            let spans = heads.spans(black_box(&words), n);
            let queries = heads.queries([&words[..32], &words[32..64]], k);
            let logits = heads.logits_all(&spans, &queries);
            black_box(logits.iter().filter(|x| x.is_finite()).map(|&x| f64::from(x)).sum::<f64>())
        }
    };
    for _ in 0..20 {
        black_box(run());
    }
    ALLOCS.store(0, std::sync::atomic::Ordering::Relaxed);
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..1 {
        checksum += run();
    }
    let allocated = ALLOCS.load(std::sync::atomic::Ordering::Relaxed);
    println!("{:.9} {checksum:.12}", start.elapsed().as_secs_f64());
    println!("allocations {}", allocated);
}

static ALLOCS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
struct Counting;
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::GlobalAlloc::alloc(&std::alloc::System, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::GlobalAlloc::realloc(&std::alloc::System, p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        unsafe { std::alloc::GlobalAlloc::dealloc(&std::alloc::System, p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;
