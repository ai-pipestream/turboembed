mod common;
#[path = "rerank/common.rs"]
mod fixture;
#[path = "rerank/held.rs"]
mod held;

use common::{Loaded, Session, new_error, session_desc, text};
use std::{path::Path, ptr};
use turbo::*;

#[test]
fn default_pair_options_use_the_session_sequence_limit() {
    let f = fixture::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap());
    let loaded = Loaded::load(&f.dir).unwrap();
    let desc = session_desc(1, 16, TURBO_PRECISION_EXACT);
    let session = Session::create(loaded.m, Some(&desc)).unwrap();
    assert!(session.info().max_seq < loaded.info().max_seq);
    let pair = turbo_text_pair { query: text("hello"), passage: text("world") };
    let defaults = turbo_rerank_options { struct_size: size_of::<turbo_rerank_options>() as u32, ..Default::default() };
    for options in [ptr::null(), &defaults] {
        let mut error = new_error();
        let code = unsafe { turbo_rerank_write_pairs(session.0, &pair, 1, options, &mut error) };
        assert_eq!(code, 0, "field {}", error.field);
        let mut result = ptr::null_mut();
        assert_eq!(unsafe { turbo_session_run(session.0, &mut result, &mut error) }, 0);
        unsafe { turbo_result_release(result) };
    }
    for (query_tokens, max_tokens, stride, field) in [(0, 17, 0, 2), (0, 4, 0, 2), (13, 16, 0, 1), (0, 16, 99, 3)] {
        let options = turbo_rerank_options { query_tokens, max_tokens, stride, ..defaults };
        let mut error = new_error();
        assert_ne!(unsafe { turbo_rerank_write_pairs(session.0, &pair, 1, &options, &mut error) }, 0);
        assert_eq!(error.field, field);
    }
}

#[test]
fn tokenizer_pair_option_errors_name_the_field() {
    let f = fixture::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap());
    let bundle = turbo::bundle::Bundle::open(&f.dir).unwrap();
    let tokenizer = turbo::tokenizer::Tokenizer::load(&bundle).unwrap();
    for (query, max, stride, field) in [(29, 32, 0, 1), (8, 4, 0, 2), (8, 33, 0, 2), (8, 32, 99, 3)] {
        assert_eq!(tokenizer.encode_pair("hello", "world", query, max, stride).unwrap_err().field, field);
    }
}

/// On the CPU the scores are in the backend's own output, which exports.
#[test]
fn held_rerank_scores_export_their_host_pointer_on_the_cpu() {
    let f = fixture::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap());
    let loaded = Loaded::load(&f.dir).unwrap();
    held::held_scores(&loaded, TURBO_STAGE_HOST, TURBO_HANDLE_CUDA_PTR);
}
