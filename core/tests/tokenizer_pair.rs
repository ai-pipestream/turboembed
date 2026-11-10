//! `turbo_tokenizer_encode_pair` through the C interface: the ids and
//! segment types the Rust pair encoder gives for a BERT and a RoBERTa
//! pair, the checks the C call makes of its own (stride, capacity, NULL
//! outputs, the bundle's rerank `max_seq`), and one tokenizer used from
//! many threads at once.

mod common;
#[path = "rerank/common.rs"]
mod rerank;

use std::path::Path;
use std::ptr;

use common::*;
use turbo::bundle::Bundle;
use turbo::tokenizer::Tokenizer;
use turbo::*;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

fn opts(query_tokens: u32, max_tokens: u32, stride: u32) -> turbo_rerank_options {
    turbo_rerank_options { struct_size: size_of::<turbo_rerank_options>() as u32, query_tokens, max_tokens, stride }
}

/// One pair through the C call into buffers of `capacity` entries:
/// the ids and types written, or the failure.
fn encode_pair(
    t: *mut turbo_tokenizer,
    query: &str,
    passage: &str,
    o: Option<&turbo_rerank_options>,
    capacity: u32,
) -> Result<(Vec<i32>, Vec<i32>), Failure> {
    let mut ids = vec![-7; capacity as usize];
    let mut types = vec![-7; capacity as usize];
    let mut written = 0u32;
    let mut err = new_error();
    let pair = turbo_text_pair { query: text(query), passage: text(passage) };
    let rc = unsafe {
        turbo_tokenizer_encode_pair(
            t,
            pair,
            o.map_or(ptr::null(), |o| o as *const _),
            ids.as_mut_ptr(),
            types.as_mut_ptr(),
            capacity,
            &mut written,
            &mut err,
        )
    };
    if rc != 0 {
        return Err(failure(rc, &err));
    }
    ids.truncate(written as usize);
    types.truncate(written as usize);
    Ok((ids, types))
}

#[test]
fn pair_ids_and_types_match_the_rust_pair_encoder() {
    let f = rerank::Fixture::new(root());
    let tok = Tok::create(&f.dir).unwrap();
    let rust = Tokenizer::load(&Bundle::open(&f.dir).unwrap()).unwrap();
    let long = "passage ".repeat(45);
    let mut cases = f.cases();
    cases.push(("query ".repeat(20), long.clone()));
    cases.push(("a".into(), "b".into()));
    for (q, p) in &cases {
        for (query_tokens, max_tokens) in [(0, 0), (8, 32), (4, 16), (1, 12)] {
            let o = opts(query_tokens, max_tokens, 0);
            let want = &rust.encode_pair(q, p, query_tokens, max_tokens, 0).unwrap()[0];
            let got = encode_pair(tok.t, q, p, Some(&o), 32).unwrap();
            assert_eq!(got, (want.ids.clone(), want.types.clone()), "{q:?} / {p:?} at ({query_tokens}, {max_tokens})");
        }
        // NULL options are the bundle's defaults, as zeroed options are.
        assert_eq!(
            encode_pair(tok.t, q, p, None, 32).unwrap(),
            encode_pair(tok.t, q, p, Some(&opts(0, 0, 0)), 32).unwrap()
        );
    }
    // Only the first window comes back: a passage longer than one window
    // is cut where the Rust encoder's first window ends.
    let (ids, types) = encode_pair(tok.t, "q", &long, None, 32).unwrap();
    assert_eq!(ids.len(), 32);
    assert_eq!((ids[0], *ids.last().unwrap()), (101, 102));
    assert!(types.iter().skip_while(|&&t| t == 0).all(|&t| t == 1));
}

/// A RoBERTa bundle takes the RoBERTa pair template its family implies:
/// `<s> query </s></s> passage </s>`, four special tokens and every
/// segment type 0, as the Rust pair encoder gives.
#[test]
fn a_roberta_pair_matches_the_rust_pair_encoder() {
    let dir = root().join("testdata/tiny-sat-bundle");
    let tok = Tok::create(&dir).unwrap();
    let rust = Tokenizer::load(&Bundle::open(&dir).unwrap()).unwrap();
    let (bos, eos) = (0, 2);
    for (q, p) in [("a query", "a passage"), ("which one", "the long passage that follows it"), ("é", "ü")] {
        for (query_tokens, max_tokens) in [(0, 0), (4, 16), (1, 12)] {
            let o = opts(query_tokens, max_tokens, 0);
            let want = &rust.encode_pair(q, p, query_tokens, max_tokens, 0).unwrap()[0];
            let (ids, types) = encode_pair(tok.t, q, p, Some(&o), 512).unwrap();
            assert_eq!((&ids, &types), (&want.ids, &want.types), "{q:?} / {p:?} at ({query_tokens}, {max_tokens})");
            assert_eq!((ids[0], *ids.last().unwrap()), (bos, eos), "{q:?} / {p:?}");
            assert_eq!(ids.iter().filter(|&&i| i == bos || i == eos).count(), 4, "{q:?} / {p:?}: {ids:?}");
            assert!(ids.windows(2).any(|w| w == [eos, eos]), "{q:?} / {p:?}: no </s></s> between the two: {ids:?}");
            assert!(types.iter().all(|&t| t == 0), "{q:?} / {p:?}: {types:?}");
        }
    }
}

#[test]
fn pair_options_buffers_and_task_are_checked() {
    let f = rerank::Fixture::new(root());
    let tok = Tok::create(&f.dir).unwrap();
    let field = |r: Result<(Vec<i32>, Vec<i32>), Failure>| {
        let e = r.unwrap_err();
        (e.code, e.field)
    };
    // A stride asks for windows past the first, which this call does not return.
    assert_eq!(field(encode_pair(tok.t, "q", "p", Some(&opts(0, 0, 1)), 32)), (status::INVALID_ARGUMENT, 3));
    // max_tokens above the bundle's rerank max_seq (32), or leaving no passage room.
    assert_eq!(field(encode_pair(tok.t, "q", "p", Some(&opts(0, 33, 0)), 64)), (status::INVALID_ARGUMENT, 2));
    assert_eq!(field(encode_pair(tok.t, "q", "p", Some(&opts(0, 3, 0)), 64)), (status::INVALID_ARGUMENT, 2));
    // A query budget that leaves no passage token.
    assert_eq!(field(encode_pair(tok.t, "q", "p", Some(&opts(29, 32, 0)), 64)), (status::INVALID_ARGUMENT, 1));
    // A struct_size this library does not know.
    let mut short = opts(0, 0, 0);
    short.struct_size -= 4;
    assert_eq!(encode_pair(tok.t, "q", "p", Some(&short), 32).unwrap_err().code, status::INVALID_STRUCT_SIZE);
    // Buffers shorter than the pair, and NULL buffers, are capacity errors.
    let (ids, _) = encode_pair(tok.t, "a query", "a passage", None, 32).unwrap();
    let short_cap = ids.len() as u32 - 1;
    assert_eq!(encode_pair(tok.t, "a query", "a passage", None, short_cap).unwrap_err().code, status::CAPACITY);
    let mut written = 0u32;
    let mut err = new_error();
    let pair = turbo_text_pair { query: text("q"), passage: text("p") };
    let rc = unsafe {
        turbo_tokenizer_encode_pair(
            tok.t,
            pair,
            ptr::null(),
            ptr::null_mut(),
            ptr::null_mut(),
            32,
            &mut written,
            &mut err,
        )
    };
    assert_eq!(rc, status::CAPACITY);
    // NULL written and a NULL handle.
    let mut ids = [0i32; 32];
    let mut types = [0i32; 32];
    let rc = unsafe {
        turbo_tokenizer_encode_pair(
            tok.t,
            pair,
            ptr::null(),
            ids.as_mut_ptr(),
            types.as_mut_ptr(),
            32,
            ptr::null_mut(),
            &mut err,
        )
    };
    assert_eq!(rc, status::INVALID_ARGUMENT);
    let rc = unsafe {
        turbo_tokenizer_encode_pair(
            ptr::null_mut(),
            pair,
            ptr::null(),
            ids.as_mut_ptr(),
            types.as_mut_ptr(),
            32,
            &mut written,
            &mut err,
        )
    };
    assert_eq!(rc, status::INVALID_HANDLE);
    // An embedding BERT with token types takes the pair template its
    // family implies, as the Rust encoder does; a ModernBERT has none.
    let dir = root().join("testdata/tiny-bert-bundle");
    let embed = Tok::create(&dir).unwrap();
    let rust = Tokenizer::load(&Bundle::open(&dir).unwrap()).unwrap();
    let want = &rust.encode_pair("a query", "a passage", 0, 0, 0).unwrap()[0];
    assert_eq!(
        encode_pair(embed.t, "a query", "a passage", None, 512).unwrap(),
        (want.ids.clone(), want.types.clone())
    );
    let modern = Tok::create(&root().join("testdata/tiny-modernbert-bundle")).unwrap();
    assert_eq!(encode_pair(modern.t, "q", "p", None, 32).unwrap_err().code, status::UNSUPPORTED_TASK);
}

/// A tokenizer handle shared by the threads of one test: turbo.h says a
/// tokenizer is thread-safe, which is what this exercises.
#[derive(Clone, Copy)]
struct Shared(*mut turbo_tokenizer);
// SAFETY: turbo_tokenizer is documented thread-safe (turbo.h), and the
// handle outlives the scope the threads run in.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

#[test]
fn one_tokenizer_on_many_threads_gives_the_single_thread_ids() {
    let f = rerank::Fixture::new(root());
    let tok = Tok::create(&f.dir).unwrap();
    let cases = f.cases();
    let want: Vec<_> = cases.iter().map(|(q, p)| encode_pair(tok.t, q, p, None, 32).unwrap()).collect();
    let rows = tok.encode(&cases.iter().map(|(q, _)| q.as_str()).collect::<Vec<_>>(), None, 32).unwrap();
    let shared = Shared(tok.t);
    std::thread::scope(|s| {
        for k in 0..8 {
            let (cases, want, rows) = (&cases, &want, &rows);
            s.spawn(move || {
                let t = shared;
                for round in 0..50 {
                    let i = (k + round) % cases.len();
                    let (q, p) = &cases[i];
                    assert_eq!(&encode_pair(t.0, q, p, None, 32).unwrap(), &want[i], "thread {k} round {round}");
                }
                let again = encode_rows(t.0, cases, 32);
                assert_eq!(again, rows.ids, "thread {k}: turbo_tokenizer_encode");
            });
        }
    });
}

/// The padded ids `turbo_tokenizer_encode` writes for the queries.
fn encode_rows(t: *mut turbo_tokenizer, cases: &[(String, String)], stride: u32) -> Vec<i32> {
    let views: Vec<turbo_text> = cases.iter().map(|(q, _)| text(q)).collect();
    let n = views.len() * stride as usize;
    let (mut ids, mut mask) = (vec![-7; n], vec![-7; n]);
    let mut err = new_error();
    let rc = unsafe {
        turbo_tokenizer_encode(
            t,
            views.as_ptr(),
            views.len() as u32,
            ptr::null(),
            ids.as_mut_ptr(),
            mask.as_mut_ptr(),
            ptr::null_mut(),
            stride,
            ptr::null_mut(),
            &mut err,
        )
    };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    ids
}
