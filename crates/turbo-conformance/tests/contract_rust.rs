//! Group `contract`, Rust layer: what the safe API must accept and refuse.
//!
//! Every case asserts an exact status code. The cases that can only be
//! expressed through raw pointers (invalid UTF-8, NULL views, `struct_size`)
//! live in `contract_c.rs`.

use turbo::abi::*;
use turbo::buffer::BufferDesc;
use turbo::provider::{ClassifyOptions, EmbedOptions, RerankOptions, RunOptions, SessionDesc, TokenBatch};
use turbo::types::{DType, Modality, Placement, Pooling, Task, Truncate};
use turbo_conformance::{assert_err, needs, read_f32, BundleKind, Target};

fn embed_target() -> Target {
    Target::from_env()
}

#[test]
fn contract_empty_text_produces_a_vector() {
    let t = embed_target();
    needs!(t, Embedding);
    let (model, session) = t.session(BundleKind::Embedding);
    session.write_text(&[""], &EmbedOptions::default()).expect("empty text is a valid input");
    let result = session.run(&RunOptions::default()).expect("run on empty text");
    let v = read_f32(&result, 0);
    assert_eq!(v.len(), model.info().dim as usize, "an empty input still produces one full-width vector");
    assert!(v.iter().all(|x| x.is_finite()), "vector for empty text must be finite: {v:?}");
    // A row of zeros is what a provider returns when it skipped the input
    // rather than embedding it, and it is also the one vector no normalizer
    // can produce, so the contract rules it out.
    assert!(v.iter().any(|x| *x != 0.0), "an empty input must still be embedded, not zeroed: {v:?}");
    if model.info().normalize == Some(turbo::types::Normalize::L2) {
        let n = turbo_conformance::norm(&v);
        assert!((n - 1.0).abs() < 1e-5, "the bundle declares L2, so the row must be unit length (norm {n})");
    }
}

#[test]
fn contract_empty_and_nonempty_text_in_one_batch_are_accepted() {
    let t = embed_target();
    needs!(t, Embedding);
    let (model, session) = t.session(BundleKind::Embedding);
    session.write_text(&["", "hello world", ""], &EmbedOptions::default()).expect("mixed batch");
    let result = session.run(&RunOptions::default()).expect("run");
    let dim = model.info().dim as usize;
    let v = read_f32(&result, 0);
    assert_eq!(v.len(), 3 * dim);
    let first = &v[..dim];
    let second = &v[dim..2 * dim];
    let third = &v[2 * dim..];
    assert_eq!(first, third, "the same input must produce the same row");
    assert_ne!(first, second, "a different input in the same batch must produce a different row");
    assert!(v.iter().all(|x| x.is_finite()), "every row of a mixed batch must be finite");
    // Equal rows of zeros would satisfy the equality above without anything
    // having been embedded at all.
    assert!(first.iter().any(|x| *x != 0.0), "the empty rows were zeroed instead of embedded: {first:?}");
}

#[test]
fn contract_embedded_nul_is_preserved() {
    let t = embed_target();
    needs!(t, Embedding);
    let (_m, session) = t.session(BundleKind::Embedding);
    // An embedded NUL is ordinary UTF-8 data, not a terminator.
    session.write_text(&["a\0b"], &EmbedOptions::default()).expect("embedded NUL is accepted");
    let with_nul = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
    session.write_text(&["a"], &EmbedOptions::default()).expect("write");
    let truncated_at_nul = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
    // Whether the NUL itself changes the tokens is the bundle's tokenizer's
    // rule (a BERT normalizer drops control characters, the mock hashes
    // every byte); the library's contract is only that the text after it is
    // seen, so the vector cannot equal the C-string reading of the input.
    assert_eq!(with_nul.len(), truncated_at_nul.len(), "both inputs produce one full-width row");
    assert_ne!(with_nul, truncated_at_nul, "the NUL must not have terminated the view");
}

#[test]
fn contract_zero_count_batch_is_invalid_argument() {
    let t = embed_target();
    needs!(t, Embedding);
    let (_m, session) = t.session(BundleKind::Embedding);
    assert_err!(session.write_text(&[], &EmbedOptions::default()), TURBO_E_INVALID_ARGUMENT);
}

#[test]
fn contract_zero_count_rerank_and_classify_are_invalid_argument() {
    let t = embed_target();
    needs!(t, Reranker, Classifier);
    let (_m, rerank) = t.session(BundleKind::Reranker);
    assert_err!(rerank.write_pairs("q", &[], &RerankOptions::default()), TURBO_E_INVALID_ARGUMENT);
    let (_m2, classify) = t.session(BundleKind::Classifier);
    assert_err!(classify.write_text_classify(&[], &ClassifyOptions::default()), TURBO_E_INVALID_ARGUMENT);
}

#[test]
fn contract_batch_above_session_max_is_capacity() {
    let t = embed_target();
    needs!(t, Embedding);
    let (model, session) = t.session(BundleKind::Embedding);
    let max = model.info().max_batch as usize;
    let texts = vec!["x"; max + 1];
    assert_err!(session.write_text(&texts, &EmbedOptions::default()), TURBO_E_CAPACITY);
    let exactly = vec!["x"; max];
    session.write_text(&exactly, &EmbedOptions::default()).expect("max_batch rows are accepted");
}

#[test]
fn contract_session_maxima_above_the_model_are_capacity() {
    let t = embed_target();
    needs!(t, Embedding);
    let model = t.model(BundleKind::Embedding);
    let info = model.info().clone();
    let too_wide = SessionDesc { max_batch: info.max_batch + 1, ..Default::default() };
    assert_err!(model.create_session(&too_wide), TURBO_E_CAPACITY, field = 2);
    let too_long = SessionDesc { max_seq: info.max_seq + 1, ..Default::default() };
    assert_err!(model.create_session(&too_long), TURBO_E_CAPACITY, field = 3);
}

#[test]
fn contract_unknown_enum_values_are_invalid_enum() {
    // Every ABI enumeration rejects a value outside its constant set rather
    // than mapping it onto a default.
    assert_eq!(Task::from_abi(99).unwrap_err().code(), TURBO_E_INVALID_ENUM);
    assert_eq!(Modality::from_abi(0).unwrap_err().code(), TURBO_E_INVALID_ENUM);
    assert_eq!(Truncate::from_abi(77).unwrap_err().code(), TURBO_E_INVALID_ENUM);
    assert_eq!(Pooling::from_abi(u32::MAX).unwrap_err().code(), TURBO_E_INVALID_ENUM);
    assert_eq!(DType::from_abi(0).unwrap_err().code(), TURBO_E_INVALID_ENUM);
    assert_eq!(Placement::from_abi(9).unwrap_err().code(), TURBO_E_INVALID_ENUM);
}

#[test]
fn contract_oversized_buffer_shapes_are_invalid_shape() {
    let t = embed_target();
    let ctx = t.context();
    // Product overflows u64.
    let overflow = BufferDesc::packed(Placement::Host, DType::F32, &[u64::MAX, 2]);
    assert_err!(overflow, TURBO_E_INVALID_SHAPE);
    // Fits in u64 but not in the address space.
    let unaddressable = BufferDesc::packed(Placement::Host, DType::F32, &[1 << 62]);
    assert_err!(unaddressable, TURBO_E_INVALID_SHAPE);
    // Rank above TURBO_MAX_RANK.
    let deep = BufferDesc::packed(Placement::Host, DType::F32, &[1; TURBO_MAX_RANK + 1]);
    assert_err!(deep, TURBO_E_INVALID_SHAPE);
    // A shape the device cannot possibly back is a resource error, never a partial allocation.
    let huge = BufferDesc::packed(Placement::Host, DType::F32, &[1 << 40, 1 << 20]).expect("descriptor");
    let e = match ctx.alloc(&huge) {
        Ok(_) => panic!("an allocation of 2^60 bytes must fail"),
        Err(e) => e,
    };
    assert!(
        e.code() == TURBO_E_OUT_OF_MEMORY || e.code() == TURBO_E_INVALID_SHAPE,
        "expected OUT_OF_MEMORY or INVALID_SHAPE, got {} ({})",
        e.code_name(),
        e.message()
    );
}

#[test]
fn contract_token_batch_short_arrays_are_invalid_shape() {
    let ids = [1, 2, 3, 4];
    let mask = [1, 1, 1, 1];
    let full = TokenBatch { batch: 2, seq: 2, row_stride: 2, ids: &ids, mask: &mask, types: None };
    full.validate(1000).expect("a well formed batch");
    // A short array names itself, the same way a bad value in it does.
    for (field, e) in [
        (TokenBatch::FIELD_IDS, TokenBatch { ids: &ids[..3], ..full }.validate(1000).unwrap_err()),
        (TokenBatch::FIELD_MASK, TokenBatch { mask: &mask[..3], ..full }.validate(1000).unwrap_err()),
        (TokenBatch::FIELD_TYPES, TokenBatch { types: Some(&mask[..3]), ..full }.validate(1000).unwrap_err()),
    ] {
        assert_eq!(e.code(), TURBO_E_INVALID_SHAPE, "{e}");
        assert_eq!(e.field(), field, "a short array must name its field: {e}");
    }
}

#[test]
fn contract_token_batch_zero_dimensions_are_invalid_shape() {
    assert_eq!(TokenBatch::required_len(0, 4, 4).unwrap_err().code(), TURBO_E_INVALID_SHAPE);
    assert_eq!(TokenBatch::required_len(4, 0, 4).unwrap_err().code(), TURBO_E_INVALID_SHAPE);
    // row_stride smaller than seq would make rows overlap.
    assert_eq!(TokenBatch::required_len(2, 4, 3).unwrap_err().code(), TURBO_E_INVALID_SHAPE);
    // Large dimensions are computed with checked arithmetic, never wrapped.
    let big = TokenBatch::required_len(u32::MAX, u32::MAX, u32::MAX);
    match big {
        // 64-bit hosts can represent this; 32-bit hosts must report the overflow.
        Ok(n) => assert_eq!(n as u128, (u32::MAX as u128 - 1) * u32::MAX as u128 + u32::MAX as u128),
        Err(e) => assert_eq!(e.code(), TURBO_E_INVALID_SHAPE),
    }
}

#[test]
fn contract_token_batch_bad_ids_and_mask_are_invalid_argument() {
    let t = embed_target();
    needs!(t, Embedding);
    let (model, session) = t.session(BundleKind::Embedding);
    let vocab = model.info().vocab_size;
    // A model that reports no vocabulary size cannot bound-check an id at
    // all (`TokenBatch::validate` skips the upper bound when it is 0), so the
    // refusal below would be unreachable.
    assert_ne!(vocab, 0, "a loaded model must report its vocabulary size so token ids can be bounded");
    let ids = [1, 2];
    let mask = [1, 1];
    let good = TokenBatch { batch: 1, seq: 2, row_stride: 2, ids: &ids, mask: &mask, types: None };
    session.write_tokens(&good).expect("a valid token batch");

    // Each refusal names the array the bad value is in, by its 1-based
    // `turbo_token_batch` field index, so a caller does not have to re-scan
    // three arrays to find it.
    let negative = [1, -3];
    assert_err!(
        session.write_tokens(&TokenBatch { ids: &negative, ..good }),
        TURBO_E_INVALID_ARGUMENT,
        field = TokenBatch::FIELD_IDS
    );
    let above = [1, vocab as i32];
    assert_err!(
        session.write_tokens(&TokenBatch { ids: &above, ..good }),
        TURBO_E_INVALID_ARGUMENT,
        field = TokenBatch::FIELD_IDS
    );
    let bad_mask = [1, 2];
    assert_err!(
        session.write_tokens(&TokenBatch { mask: &bad_mask, ..good }),
        TURBO_E_INVALID_ARGUMENT,
        field = TokenBatch::FIELD_MASK
    );
    let bad_types = [0, 5];
    assert_err!(
        session.write_tokens(&TokenBatch { types: Some(&bad_types), ..good }),
        TURBO_E_INVALID_ARGUMENT,
        field = TokenBatch::FIELD_TYPES
    );
}

#[test]
fn contract_token_batch_above_session_shape_is_capacity() {
    let t = embed_target();
    needs!(t, Embedding);
    let (model, session) = t.session(BundleKind::Embedding);
    let seq = model.info().max_seq as usize + 1;
    let ids = vec![1i32; seq];
    let mask = vec![1i32; seq];
    let long = TokenBatch { batch: 1, seq: seq as u32, row_stride: seq as u32, ids: &ids, mask: &mask, types: None };
    assert_err!(session.write_tokens(&long), TURBO_E_CAPACITY);

    let rows = model.info().max_batch as usize + 1;
    let ids = vec![1i32; rows * 2];
    let mask = vec![1i32; rows * 2];
    let wide = TokenBatch { batch: rows as u32, seq: 2, row_stride: 2, ids: &ids, mask: &mask, types: None };
    assert_err!(session.write_tokens(&wide), TURBO_E_CAPACITY);
}

#[test]
fn contract_run_without_input_is_invalid_state() {
    let t = embed_target();
    needs!(t, Embedding);
    let (_m, session) = t.session(BundleKind::Embedding);
    assert_err!(session.run(&RunOptions::default()), TURBO_E_INVALID_STATE);
}

#[test]
fn contract_failed_write_leaves_the_session_without_inputs() {
    let t = embed_target();
    needs!(t, Embedding);
    let (_m, session) = t.session(BundleKind::Embedding);
    session.write_text(&["hello"], &EmbedOptions::default()).expect("write");
    // A rejected write must not leave the previous inputs runnable, and must
    // not corrupt the session either.
    assert_err!(session.write_text(&[], &EmbedOptions::default()), TURBO_E_INVALID_ARGUMENT);
    session.write_text(&["hello"], &EmbedOptions::default()).expect("the session is still usable");
    session.run(&RunOptions::default()).expect("run");
}

#[test]
fn contract_unknown_provider_options_are_invalid_argument_with_field() {
    let t = embed_target();
    needs!(t, Embedding);
    let model = t.model(BundleKind::Embedding);
    let desc = SessionDesc {
        options: turbo::handles::options_from_pairs([("definitely_not_an_option", "1")]),
        ..Default::default()
    };
    assert_err!(model.create_session(&desc), TURBO_E_INVALID_ARGUMENT, field = 1);
}

#[test]
fn contract_status_names_cover_every_code() {
    // `status_name` is the contract's own vocabulary: every constant maps to
    // its symbolic name and nothing else claims one.
    let all: &[(i32, &str)] = &[
        (TURBO_OK, "TURBO_OK"),
        (TURBO_E_INVALID_ARGUMENT, "TURBO_E_INVALID_ARGUMENT"),
        (TURBO_E_INVALID_STRUCT_SIZE, "TURBO_E_INVALID_STRUCT_SIZE"),
        (TURBO_E_INVALID_UTF8, "TURBO_E_INVALID_UTF8"),
        (TURBO_E_INVALID_HANDLE, "TURBO_E_INVALID_HANDLE"),
        (TURBO_E_INVALID_SHAPE, "TURBO_E_INVALID_SHAPE"),
        (TURBO_E_INVALID_STATE, "TURBO_E_INVALID_STATE"),
        (TURBO_E_INVALID_ENUM, "TURBO_E_INVALID_ENUM"),
        (TURBO_E_UNSUPPORTED, "TURBO_E_UNSUPPORTED"),
        (TURBO_E_UNSUPPORTED_OPTION, "TURBO_E_UNSUPPORTED_OPTION"),
        (TURBO_E_UNSUPPORTED_TASK, "TURBO_E_UNSUPPORTED_TASK"),
        (TURBO_E_UNSUPPORTED_DTYPE, "TURBO_E_UNSUPPORTED_DTYPE"),
        (TURBO_E_UNSUPPORTED_PLACEMENT, "TURBO_E_UNSUPPORTED_PLACEMENT"),
        (TURBO_E_NOT_IMPLEMENTED, "TURBO_E_NOT_IMPLEMENTED"),
        (TURBO_E_UNSUPPORTED_MODALITY, "TURBO_E_UNSUPPORTED_MODALITY"),
        (TURBO_E_OUT_OF_MEMORY, "TURBO_E_OUT_OF_MEMORY"),
        (TURBO_E_BUSY, "TURBO_E_BUSY"),
        (TURBO_E_OVERLOADED, "TURBO_E_OVERLOADED"),
        (TURBO_E_CAPACITY, "TURBO_E_CAPACITY"),
        (TURBO_E_DEVICE_NOT_FOUND, "TURBO_E_DEVICE_NOT_FOUND"),
        (TURBO_E_DEVICE_UNAVAILABLE, "TURBO_E_DEVICE_UNAVAILABLE"),
        (TURBO_E_RUNTIME, "TURBO_E_RUNTIME"),
        (TURBO_E_PROVIDER_LOAD, "TURBO_E_PROVIDER_LOAD"),
        (TURBO_E_ABI_MISMATCH, "TURBO_E_ABI_MISMATCH"),
        (TURBO_E_CANCELLED, "TURBO_E_CANCELLED"),
        (TURBO_E_BUNDLE_NOT_FOUND, "TURBO_E_BUNDLE_NOT_FOUND"),
        (TURBO_E_BUNDLE_INVALID, "TURBO_E_BUNDLE_INVALID"),
        (TURBO_E_BUNDLE_INTEGRITY, "TURBO_E_BUNDLE_INTEGRITY"),
        (TURBO_E_BUNDLE_NO_ARTIFACT, "TURBO_E_BUNDLE_NO_ARTIFACT"),
        (TURBO_E_INTERNAL, "TURBO_E_INTERNAL"),
        (TURBO_E_PANIC, "TURBO_E_PANIC"),
    ];
    for (code, name) in all {
        assert_eq!(turbo::error::status_name(*code), *name, "status_name({code:#x})");
    }
    // The table above is the whole vocabulary, not a sample: every status
    // constant the committed header defines has to appear in it, so a new
    // code cannot be added without a name and a case.
    let header = turbo_conformance::repo_root().join("include").join("turbo").join("turbo_types.h");
    let text = std::fs::read_to_string(&header).unwrap_or_else(|e| panic!("reading {}: {e}", header.display()));
    for line in text.lines() {
        let Some(rest) = line.trim_start().strip_prefix("#define TURBO_") else { continue };
        let Some(name) = rest.split_whitespace().next() else { continue };
        if !(name.starts_with("E_") || name == "OK") {
            continue;
        }
        let name = format!("TURBO_{name}");
        assert!(
            all.iter().any(|(_, n)| *n == name),
            "{} defines {name}, which contract_status_names_cover_every_code does not check",
            header.display()
        );
    }
    assert_eq!(turbo::error::status_name(-1), "TURBO_E_UNKNOWN");
    assert_eq!(turbo::error::status_name(0x7FFF), "TURBO_E_UNKNOWN");
    // The grade of every code is one of the documented bands.
    for (code, name) in all {
        let grade = code & 0xF00;
        assert!(
            *code == TURBO_OK || (0x100..=0x600).contains(&grade),
            "{name} ({code:#x}) is outside the graded status space"
        );
    }
}

#[test]
fn contract_abi_version_matches_the_runtime() {
    let t = embed_target();
    assert_eq!(t.runtime.abi_version(), TURBO_ABI_VERSION);
}
