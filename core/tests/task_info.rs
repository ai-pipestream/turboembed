//! What turbo_model_get_info reports for a model of a task other than
//! embed, what a rerank write says when one pair is refused, and the
//! tokenizer template check on rerank and extractive QA manifests.

mod common;
#[path = "qa/common.rs"]
mod qa;
#[path = "rerank/common.rs"]
mod rerank;

use std::path::Path;
use std::ptr;

use common::{Loaded, Session, new_error, session_desc, text};
use turbo::bundle::Bundle;
use turbo::status::INVALID_ARGUMENT;
use turbo::*;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

#[test]
fn the_embed_fields_are_zero_for_rerank_and_extractive_qa() {
    let f = qa::Fixture::new(root());
    let l = Loaded::load(&f.dir).unwrap();
    let info = l.info();
    assert_eq!(info.task, TURBO_TASK_EXTRACTIVE_QA);
    assert_eq!((info.pooling, info.normalize), (0, 0));
    // A QA run's rows are the tokens' hidden states: dim is not an embed field.
    assert_eq!(info.dim, f.manifest["architecture"]["hidden"].as_u64().unwrap() as u32);
    assert_eq!((info.prefix_query[0], info.prefix_document[0], info.output_dims_count), (0, 0, 0));

    let f = rerank::Fixture::new(root());
    let l = Loaded::load(&f.dir).unwrap();
    let info = l.info();
    assert_eq!(info.task, TURBO_TASK_RERANK);
    assert_eq!((info.dim, info.pooling, info.normalize), (0, 0, 0));
    assert_eq!((info.prefix_query[0], info.prefix_document[0], info.output_dims_count), (0, 0, 0));
}

#[test]
fn a_refused_pair_is_named_by_its_index() {
    let f = rerank::Fixture::new(root());
    let l = Loaded::load(&f.dir).unwrap();
    let s = Session::create(l.m, Some(&session_desc(2, 0, TURBO_PRECISION_EXACT))).unwrap();
    let pairs = [
        turbo_text_pair { query: text("hello"), passage: text("world") },
        turbo_text_pair { query: turbo_text { ptr: ptr::null(), len: 3 }, passage: text("world") },
    ];
    let mut e = new_error();
    let code = unsafe { turbo_rerank_write_pairs(s.0, pairs.as_ptr(), 2, ptr::null(), &mut e) };
    assert_eq!(code, INVALID_ARGUMENT);
    let message = unsafe { std::ffi::CStr::from_ptr(e.message.as_ptr()) }.to_string_lossy().into_owned();
    assert!(message.starts_with("pairs[1]: "), "{message}");
}

#[test]
fn a_template_entry_that_is_no_special_token_is_refused_for_every_task() {
    let mut q = qa::Fixture::new(root());
    let mut r = rerank::Fixture::new(root());
    // Each refusal names the template; the QA manifest's own check, that
    // the template is start, $TEXT and end, can come first.
    for bad in [
        serde_json::json!(["[CLS]", "$TEXT", "[NOT_SPECIAL]"]),
        serde_json::json!(["[CLS]", "$TEXT", "$TEXT", "[SEP]"]),
    ] {
        q.manifest["tokenizer"]["template"] = bad.clone();
        q.save();
        r.manifest["tokenizer"]["template"] = bad;
        r.save();
        for dir in [&q.dir, &r.dir] {
            let e = Bundle::open(dir).err().unwrap_or_else(|| panic!("{} took the template", dir.display()));
            assert!(e.message.contains("template"), "{}", e.message);
        }
    }
}
