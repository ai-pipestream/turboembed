//! `extract` checks its entities before it builds their pairs, which grow
//! as the square of their count: a list of invalid or too many entities is
//! refused with little allocated. A counting allocator records the bytes
//! the calling thread requests during the call; this binary holds one test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use turbo::deberta::{Config, Encoder};
use turbo::safetensors::File;
use turbo::sentencepiece::SentencePiece;
use turbo_analysis::ModelRef;
use turbo_relation::GlirelRelations;
use turbo_relation::RelationError;
use turbo_relation::glirel::heads::Heads;
use turbo_relation::glirel::prompt::Specials;

struct Counting;

thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = BYTES.try_with(|b| b.set(b.get() + layout.size()));
        unsafe { System.alloc(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = BYTES.try_with(|b| b.set(b.get() + new_size));
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glirel");

fn model() -> GlirelRelations {
    let config = Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 800,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let bytes = std::fs::read(format!("{DIR}/reduced.safetensors")).unwrap();
    let canonical = turbo_bundle::deberta::convert(&config, &bytes, "token_rep_layer.bert_layer.model.").unwrap();
    let encoder = Encoder::from_safetensors(config, &canonical).unwrap();
    let heads = Heads::load(&File::parse("reduced", &bytes).unwrap(), 32).unwrap();
    let spm = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/sentencepiece/unigram-identity.model");
    let tokenizer = SentencePiece::parse(&std::fs::read(spm).unwrap()).unwrap();
    let model = ModelRef { id: "reduced".into(), revision: "seeded".into(), sha256: "none".into() };
    GlirelRelations::new(encoder, heads, tokenizer, Specials { cls: 1, sep: 2, rel: 3 }, model, 512).unwrap()
}

fn allocated(f: impl FnOnce()) -> usize {
    let before = BYTES.with(Cell::get);
    f();
    BYTES.with(Cell::get) - before
}

#[test]
fn entities_are_checked_before_their_pairs_are_built() {
    let m = model();
    // 512 entities past the words: their 261,632 pairs would take 4 MB.
    let invalid = vec![(usize::MAX, usize::MAX); 512];
    let bytes = allocated(|| assert!(m.extract("a b", &invalid, &["label"], 0.5).is_err()));
    assert!(bytes < 1 << 20, "{bytes} bytes allocated for invalid entities");
    // More than the limit is refused before the encoder runs.
    let many = vec![(0, 0); 513];
    let bytes = allocated(|| {
        let got = m.extract("a b", &many, &["label"], 0.5);
        assert!(matches!(got, Err(RelationError::OverLimit { limit: "max_mentions", .. })), "{got:?}");
    });
    assert!(bytes < 1 << 10, "{bytes} bytes allocated for too many entities");
}
