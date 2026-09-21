//! Token packing contract at the safe-wrapper surface (`TokenBuffer`,
//! `Engine::pack_text`). CPU-only and always on; WordPiece/vocab-dependent
//! tests early-return unless `weights_present()` (same idiom as
//! `cpu_minilm.rs`). `position_ids` has no safe accessor, so tests that pin
//! positions read the four arrays through `turborerank::ffi` on a throwaway
//! CPU buffer. Expected rows are the documented `[CLS] query [SEP] doc [SEP]`
//! layout from `include/turborerank.h` / `native/turborerank/src/pack.cpp`.

use std::ptr;

use turborerank::ffi;
use turborerank::{
    default_model_dir, weights_present, Device, Engine, Error, TokenBuffer, Truncation,
};

const CLS: i32 = 101;
const SEP: i32 = 102;
const PAD: i32 = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pack one pair through the raw FFI into a 1-row CPU buffer and copy out all
/// four `[seq]` rows (ids, mask, types, positions). Needed because the safe
/// `TokenBuffer` exposes no `position_ids` accessor.
fn pack_row_ffi(
    query: &[i32],
    doc: &[i32],
    seq: u32,
    truncation: ffi::turborerank_truncation,
    max_length: u32,
) -> [Vec<i32>; 4] {
    unsafe {
        let mut raw = ptr::null_mut();
        let st = ffi::turborerank_buffer_alloc(
            ffi::turborerank_device::TURBORERANK_DEVICE_CPU,
            1,
            seq,
            &mut raw,
        );
        assert!(
            st == ffi::turborerank_status::TURBORERANK_OK,
            "buffer_alloc: {st:?}"
        );
        assert!(!raw.is_null());
        let st = ffi::turborerank_pack_ids(
            raw,
            0,
            query.as_ptr(),
            query.len(),
            doc.as_ptr(),
            doc.len(),
            truncation,
            max_length,
        );
        assert!(
            st == ffi::turborerank_status::TURBORERANK_OK,
            "pack_ids: {st:?}"
        );
        let b = &*raw;
        assert_eq!(b.batch, 1);
        assert_eq!(b.seq, seq);
        assert_eq!(b.row_stride, seq, "row_stride must equal seq");
        let n = seq as usize;
        let read = |p: *mut i32| std::slice::from_raw_parts(p, n).to_vec();
        let rows = [
            read(b.input_ids),
            read(b.attention_mask),
            read(b.token_type_ids),
            read(b.position_ids),
        ];
        ffi::turborerank_buffer_free(raw);
        rows
    }
}

fn assert_row(
    buf: &TokenBuffer,
    row: usize,
    want_ids: &[i32],
    want_mask: &[i32],
    want_types: &[i32],
) {
    let seq = buf.seq() as usize;
    let off = row * seq;
    assert_eq!(
        &buf.input_ids()[off..off + want_ids.len()],
        want_ids,
        "row {row} ids"
    );
    assert_eq!(
        &buf.attention_mask()[off..off + want_mask.len()],
        want_mask,
        "row {row} mask"
    );
    assert_eq!(
        &buf.token_type_ids()[off..off + want_types.len()],
        want_types,
        "row {row} types"
    );
}

fn arange(seq: u32) -> Vec<i32> {
    (0..seq as i32).collect()
}

fn load_cpu_engine() -> Option<Engine> {
    if !weights_present() {
        return None;
    }
    let dir = default_model_dir();
    let engine = Engine::create_with_config(Device::Cpu, Some(&dir)).expect("CPU engine");
    engine
        .load_model("ms-marco-minilm-l6")
        .unwrap_or_else(|e| panic!("load MiniLM CE: {e} ({})", engine.last_error()));
    Some(engine)
}

/// Extract the packed token stream that starts at `start` inside `row`,
/// stopping at the first `[SEP]`.
fn packed_stream(buf: &TokenBuffer, row: usize, start: usize) -> Vec<i32> {
    let seq = buf.seq() as usize;
    let off = row * seq + start;
    let mut out = Vec::new();
    for &id in &buf.input_ids()[off..off + seq - start] {
        if id == SEP {
            break;
        }
        out.push(id);
    }
    out
}

// ---------------------------------------------------------------------------
// TokenBuffer allocation contract (safe wrapper)
// ---------------------------------------------------------------------------

#[test]
fn alloc_cpu_shape_stride_alignment_and_independent_storage() {
    let mut a = TokenBuffer::alloc(Device::Cpu, 3, 24).unwrap();
    assert_eq!(a.batch(), 3);
    assert_eq!(a.seq(), 24);
    assert_eq!(a.device(), Device::Cpu);
    assert!(a.ptr_aligned(), "CPU buffers must be 64-byte aligned");
    // Safe accessors span batch * row_stride; row_stride must equal seq.
    assert_eq!(a.input_ids().len(), 3 * 24);
    assert_eq!(a.attention_mask().len(), 3 * 24);
    assert_eq!(a.token_type_ids().len(), 3 * 24);

    // Independent storage: sentinel-fill b, write a, b must be untouched.
    let mut b = TokenBuffer::alloc(Device::Cpu, 3, 24).unwrap();
    for v in b.input_ids_mut() {
        *v = -7;
    }
    for (i, v) in a.input_ids_mut().iter_mut().enumerate() {
        *v = i as i32;
    }
    assert!(
        b.input_ids().iter().all(|&v| v == -7),
        "buffers must not alias"
    );
}

// ---------------------------------------------------------------------------
// Pack layout through the safe wrapper
// ---------------------------------------------------------------------------

#[test]
fn pack_mask_is_ones_then_zeros() {
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 16).unwrap();
    // nq=4, nd=3 -> 4 + 3 + 3 specials = 10 packed slots, mask must be
    // ten 1s then 0s.
    buf.pack_ids(
        0,
        &[10, 11, 12, 13],
        &[20, 21, 22],
        Truncation::LongestFirst,
        16,
    )
    .unwrap();
    let mask = buf.attention_mask();
    assert_eq!(&mask[..10], &[1; 10]);
    assert!(
        mask[10..].iter().all(|&m| m == 0),
        "mask must be 1s then 0s"
    );
}

#[test]
fn pack_layout_positions_arange_and_zeroed_tail() {
    // The safe wrapper has no position accessor, so read all four arrays
    // through the FFI on a throwaway CPU buffer.
    let [ids, mask, types, pos] = pack_row_ffi(
        &[10, 11],
        &[20, 21, 22],
        16,
        ffi::turborerank_truncation::TURBORERANK_TRUNC_LONGEST_FIRST,
        16,
    );
    assert_eq!(&ids[..8], &[CLS, 10, 11, SEP, 20, 21, 22, SEP]);
    assert_eq!(&mask[..8], &[1; 8]);
    assert_eq!(&types[..8], &[0, 0, 0, 0, 1, 1, 1, 1]);
    // positions are monotonic arange over the whole row, pad region included.
    assert_eq!(pos, arange(16), "position_ids must be arange(seq)");
    // Everything past the packed region is zeroed (ids/mask/types).
    assert!(
        ids[8..].iter().all(|&v| v == PAD),
        "ids tail must be zeroed"
    );
    assert!(
        mask[8..].iter().all(|&v| v == 0),
        "mask tail must be zeroed"
    );
    assert!(
        types[8..].iter().all(|&v| v == 0),
        "types tail must be zeroed"
    );
}

#[test]
fn pack_specials_carry_segment_types() {
    // The query-side [SEP] is segment 0, the doc-side [SEP] segment 1, and
    // every doc token (including a truncated one) is segment 1.
    let [_, _, types, _] = pack_row_ffi(
        &[7],
        &[8, 9],
        8,
        ffi::turborerank_truncation::TURBORERANK_TRUNC_QUERY_PRIORITY,
        5,
    );
    assert_eq!(&types[..5], &[0, 0, 0, 1, 1]);
}

// ---------------------------------------------------------------------------
// Truncation boundaries (max_length - 1 / = / + 1)
// ---------------------------------------------------------------------------

#[test]
fn truncation_longest_first_at_max_length_minus_eq_plus() {
    // q = 5 tokens, d = 2 tokens -> total = 10 = 5 + 2 + 3 specials.
    let q = [1, 2, 3, 4, 5];
    let d = [6, 7];
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 12).unwrap();

    // max_length = total - 1: drop exactly one token from the longer (query)
    // side -> q keeps 4.
    buf.pack_ids(0, &q, &d, Truncation::LongestFirst, 9)
        .unwrap();
    assert_row(
        &buf,
        0,
        &[CLS, 1, 2, 3, 4, SEP, 6, 7, SEP],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        &[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    );

    // max_length = total: fits exactly.
    buf.pack_ids(0, &q, &d, Truncation::LongestFirst, 10)
        .unwrap();
    assert_row(
        &buf,
        0,
        &[CLS, 1, 2, 3, 4, 5, SEP, 6, 7, SEP],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        &[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    );

    // max_length = total + 1: same full row, no truncation applied.
    buf.pack_ids(0, &q, &d, Truncation::LongestFirst, 11)
        .unwrap();
    assert_row(
        &buf,
        0,
        &[CLS, 1, 2, 3, 4, 5, SEP, 6, 7, SEP],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        &[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    );
}

#[test]
fn truncation_longest_first_tie_drops_query_first() {
    // With equal-length sides over budget, pack_ids drops from the query on
    // ties (`nq >= nd`), so the query loses one more token than the doc.
    // NOTE: this tie-break differs from HuggingFace `longest_first` (which
    // drops the doc on ties) and from `pack_text`'s split formula — pinned
    // as current behavior, see pack_text divergence test below.
    let q = [1, 2, 3];
    let d = [4, 5, 6];

    let [ids, mask, _, _] = pack_row_ffi(
        &q,
        &d,
        8,
        ffi::turborerank_truncation::TURBORERANK_TRUNC_LONGEST_FIRST,
        6,
    );
    assert_eq!(&ids[..6], &[CLS, 1, SEP, 4, 5, SEP]);
    assert_eq!(&mask[..6], &[1; 6]);

    let [ids, mask, _, _] = pack_row_ffi(
        &q,
        &d,
        8,
        ffi::turborerank_truncation::TURBORERANK_TRUNC_LONGEST_FIRST,
        5,
    );
    assert_eq!(&ids[..5], &[CLS, 1, SEP, 4, SEP]);
    assert_eq!(&mask[..5], &[1; 5]);
}

#[test]
fn truncation_query_priority_keeps_query_and_drops_doc() {
    let q = [1, 2, 3, 4, 5];
    let d = [6, 7];
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 12).unwrap();

    // budget = 6 >= nq = 5: doc is cut to 1 token.
    buf.pack_ids(0, &q, &d, Truncation::QueryPriority, 9)
        .unwrap();
    assert_row(
        &buf,
        0,
        &[CLS, 1, 2, 3, 4, 5, SEP, 6, SEP],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        &[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    );

    // budget = 5 == nq: doc side vanishes entirely (adjacent [SEP]s).
    buf.pack_ids(0, &q, &d, Truncation::QueryPriority, 8)
        .unwrap();
    assert_row(
        &buf,
        0,
        &[CLS, 1, 2, 3, 4, 5, SEP, SEP],
        &[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    );

    // Query alone exceeds the budget: query truncated to budget, doc dropped.
    let q8 = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut small = TokenBuffer::alloc(Device::Cpu, 1, 8).unwrap();
    small
        .pack_ids(0, &q8, &d, Truncation::QueryPriority, 6)
        .unwrap();
    assert_row(
        &small,
        0,
        &[CLS, 1, 2, 3, SEP, SEP],
        &[1, 1, 1, 1, 1, 1, 0, 0],
        &[0, 0, 0, 0, 0, 1, 0, 0],
    );
}

#[test]
fn truncation_error_mode_fails_only_when_over() {
    let q = [1, 2, 3, 4, 5];
    let d = [6, 7];
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 12).unwrap();

    // total = 10: max_length 9 must fail ...
    let err = buf.pack_ids(0, &q, &d, Truncation::Error, 9).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    // ... while max_length 10 (exactly total) succeeds.
    buf.pack_ids(0, &q, &d, Truncation::Error, 10).unwrap();
    assert_row(
        &buf,
        0,
        &[CLS, 1, 2, 3, 4, 5, SEP, 6, 7, SEP],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        &[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    );
}

// ---------------------------------------------------------------------------
// Empty sides and row addressing
// ---------------------------------------------------------------------------

#[test]
fn pack_empty_sides_full_shape() {
    let mut buf = TokenBuffer::alloc(Device::Cpu, 3, 8).unwrap();

    // Empty query: [CLS] [SEP] doc [SEP].
    buf.pack_ids(0, &[], &[9, 8], Truncation::LongestFirst, 8)
        .unwrap();
    assert_row(&buf, 0, &[CLS, SEP, 9, 8, SEP], &[1; 5], &[0, 0, 1, 1, 1]);

    // Empty doc: [CLS] query [SEP] [SEP].
    buf.pack_ids(1, &[5], &[], Truncation::LongestFirst, 8)
        .unwrap();
    assert_row(&buf, 1, &[CLS, 5, SEP, SEP], &[1; 4], &[0, 0, 0, 1]);

    // Both empty: [CLS] [SEP] [SEP] (pack_text rejects this; pack_ids packs it).
    buf.pack_ids(2, &[], &[], Truncation::LongestFirst, 8)
        .unwrap();
    assert_row(&buf, 2, &[CLS, SEP, SEP], &[1; 3], &[0, 0, 1]);
}

#[test]
fn pack_ids_row_out_of_range_is_invalid_argument() {
    let mut buf = TokenBuffer::alloc(Device::Cpu, 2, 8).unwrap();
    let err = buf
        .pack_ids(2, &[1], &[2], Truncation::LongestFirst, 8)
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
}

#[test]
fn packing_row0_leaves_row1_untouched() {
    // Safe-wrapper half: sentinel-fill row 1 ids, pack row 0, row 1 ids
    // must be unchanged.
    let mut buf = TokenBuffer::alloc(Device::Cpu, 2, 8).unwrap();
    let seq = 8usize;
    for v in &mut buf.input_ids_mut()[seq..] {
        *v = -3;
    }
    buf.pack_ids(0, &[10, 11], &[20], Truncation::LongestFirst, 8)
        .unwrap();
    assert!(
        buf.input_ids()[seq..].iter().all(|&v| v == -3),
        "packing row 0 must not touch row 1 ids"
    );
    assert_row(
        &buf,
        0,
        &[CLS, 10, 11, SEP, 20, SEP],
        &[1; 6],
        &[0, 0, 0, 0, 1, 1],
    );

    // FFI half: sentinel-fill row 1 of ALL four arrays (the safe wrapper has
    // no writers for mask/types/positions) and pack row 0 through pack_ids.
    unsafe {
        let mut raw = ptr::null_mut();
        let st = ffi::turborerank_buffer_alloc(
            ffi::turborerank_device::TURBORERANK_DEVICE_CPU,
            2,
            8,
            &mut raw,
        );
        assert!(
            st == ffi::turborerank_status::TURBORERANK_OK,
            "buffer_alloc: {st:?}"
        );
        let b = &*raw;
        for arr in [
            b.input_ids,
            b.attention_mask,
            b.token_type_ids,
            b.position_ids,
        ] {
            std::slice::from_raw_parts_mut(arr.add(seq), seq).fill(-3);
        }
        let st = ffi::turborerank_pack_ids(
            raw,
            0,
            [10, 11].as_ptr(),
            2,
            [20].as_ptr(),
            1,
            ffi::turborerank_truncation::TURBORERANK_TRUNC_LONGEST_FIRST,
            8,
        );
        assert!(
            st == ffi::turborerank_status::TURBORERANK_OK,
            "pack_ids: {st:?}"
        );
        for (name, arr) in [
            ("input_ids", b.input_ids),
            ("attention_mask", b.attention_mask),
            ("token_type_ids", b.token_type_ids),
            ("position_ids", b.position_ids),
        ] {
            assert!(
                std::slice::from_raw_parts(arr.add(seq), seq)
                    .iter()
                    .all(|&v| v == -3),
                "packing row 0 must not touch row 1 {name}"
            );
        }
        ffi::turborerank_buffer_free(raw);
    }
}

#[test]
fn packed_rows_land_in_their_own_slices() {
    let mut two = TokenBuffer::alloc(Device::Cpu, 2, 16).unwrap();
    two.pack_ids(0, &[10, 11], &[20, 21, 22], Truncation::LongestFirst, 16)
        .unwrap();
    two.pack_ids(1, &[7], &[8], Truncation::LongestFirst, 16)
        .unwrap();

    let mut ref0 = TokenBuffer::alloc(Device::Cpu, 1, 16).unwrap();
    ref0.pack_ids(0, &[10, 11], &[20, 21, 22], Truncation::LongestFirst, 16)
        .unwrap();
    let mut ref1 = TokenBuffer::alloc(Device::Cpu, 1, 16).unwrap();
    ref1.pack_ids(0, &[7], &[8], Truncation::LongestFirst, 16)
        .unwrap();

    // Rows are addressed at row * seq with row_stride == seq: a 2-row buffer
    // must equal two independently packed 1-row buffers.
    assert_eq!(two.input_ids()[..16], ref0.input_ids()[..], "row 0 ids");
    assert_eq!(two.input_ids()[16..], ref1.input_ids()[..], "row 1 ids");
    assert_eq!(
        two.attention_mask()[..16],
        ref0.attention_mask()[..],
        "row 0 mask"
    );
    assert_eq!(
        two.attention_mask()[16..],
        ref1.attention_mask()[..],
        "row 1 mask"
    );
    assert_eq!(
        two.token_type_ids()[..16],
        ref0.token_type_ids()[..],
        "row 0 types"
    );
    assert_eq!(
        two.token_type_ids()[16..],
        ref1.token_type_ids()[..],
        "row 1 types"
    );
}

// ---------------------------------------------------------------------------
// pack_text vs pack_ids (vocab-dependent, gated on weights)
// ---------------------------------------------------------------------------

#[test]
fn pack_text_ascii_token_ids_match_vocab() {
    let Some(engine) = load_cpu_engine() else {
        return;
    };
    // bert-base-uncased vocab (models/rerank/ms-marco-minilm-l6/vocab.txt):
    // hello = 7592, world = 2088, tokyo = 5522.
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 16).unwrap();
    engine
        .pack_text(
            &mut buf,
            0,
            "hello world",
            "tokyo",
            Truncation::LongestFirst,
            16,
        )
        .unwrap();
    assert_row(
        &buf,
        0,
        &[CLS, 7592, 2088, SEP, 5522, SEP],
        &[1; 6],
        &[0, 0, 0, 0, 1, 1],
    );
}

#[test]
fn pack_text_unicode_accent_strip_and_cjk_ids() {
    let Some(engine) = load_cpu_engine() else {
        return;
    };
    // "héllo" lowercases + strips accents -> "hello" (7592).
    // "CAFÉ" -> "cafe" (7668); each CJK char is its own word: 日=1864 本=1876
    // 語=1950 (vocab line minus one).
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 16).unwrap();
    engine
        .pack_text(
            &mut buf,
            0,
            "héllo",
            "CAFÉ 日本語",
            Truncation::LongestFirst,
            16,
        )
        .unwrap();
    assert_row(
        &buf,
        0,
        &[CLS, 7592, SEP, 7668, 1864, 1876, 1950, SEP],
        &[1; 8],
        &[0, 0, 0, 1, 1, 1, 1, 1],
    );
}

#[test]
fn pack_text_equals_pack_ids_ascii_roundtrip() {
    let Some(engine) = load_cpu_engine() else {
        return;
    };
    let query = "How many people live in Berlin?";
    let doc = "Berlin has a population of 3,520,031 registered inhabitants.";

    // Tokenize each side alone to recover the raw id streams.
    let mut split = TokenBuffer::alloc(Device::Cpu, 2, 64).unwrap();
    engine
        .pack_text(&mut split, 0, query, "", Truncation::LongestFirst, 64)
        .unwrap();
    engine
        .pack_text(&mut split, 1, "", doc, Truncation::LongestFirst, 64)
        .unwrap();
    let q_ids = packed_stream(&split, 0, 1);
    let d_ids = packed_stream(&split, 1, 2);
    assert!(!q_ids.is_empty() && !d_ids.is_empty());

    let mut via_text = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
    engine
        .pack_text(&mut via_text, 0, query, doc, Truncation::LongestFirst, 64)
        .unwrap();
    let mut via_ids = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
    via_ids
        .pack_ids(0, &q_ids, &d_ids, Truncation::LongestFirst, 64)
        .unwrap();

    // Compare token IDs (and mask/types), not just derived outputs.
    assert_eq!(via_text.input_ids(), via_ids.input_ids(), "token ids");
    assert_eq!(via_text.attention_mask(), via_ids.attention_mask(), "mask");
    assert_eq!(via_text.token_type_ids(), via_ids.token_type_ids(), "types");
}

#[test]
fn pack_text_equals_pack_ids_unicode_roundtrip() {
    let Some(engine) = load_cpu_engine() else {
        return;
    };
    let query = "héllo wörld";
    let doc = "Tokyo: 日本語, CAFÉ — naïve résumé!";

    let mut split = TokenBuffer::alloc(Device::Cpu, 2, 64).unwrap();
    engine
        .pack_text(&mut split, 0, query, "", Truncation::LongestFirst, 64)
        .unwrap();
    engine
        .pack_text(&mut split, 1, "", doc, Truncation::LongestFirst, 64)
        .unwrap();
    let q_ids = packed_stream(&split, 0, 1);
    let d_ids = packed_stream(&split, 1, 2);
    assert!(!q_ids.is_empty() && !d_ids.is_empty());

    let mut via_text = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
    engine
        .pack_text(&mut via_text, 0, query, doc, Truncation::LongestFirst, 64)
        .unwrap();
    let mut via_ids = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
    via_ids
        .pack_ids(0, &q_ids, &d_ids, Truncation::LongestFirst, 64)
        .unwrap();

    assert_eq!(via_text.input_ids(), via_ids.input_ids(), "token ids");
    assert_eq!(via_text.attention_mask(), via_ids.attention_mask(), "mask");
    assert_eq!(via_text.token_type_ids(), via_ids.token_type_ids(), "types");
}

#[test]
fn pack_text_query_priority_truncation_equals_pack_ids() {
    let Some(engine) = load_cpu_engine() else {
        return;
    };
    // Query-priority truncation is implemented identically in both pack
    // paths, so truncated rows must agree token-for-token.
    let cases: [(&str, &str, u32); 2] = [
        ("hello world", "tokyo tokyo tokyo tokyo", 6), // doc cut to 1
        ("hello hello hello", "tokyo tokyo", 6),       // doc cut to 0
    ];
    for (query, doc, max_length) in cases {
        let mut split = TokenBuffer::alloc(Device::Cpu, 2, 64).unwrap();
        engine
            .pack_text(&mut split, 0, query, "", Truncation::LongestFirst, 64)
            .unwrap();
        engine
            .pack_text(&mut split, 1, "", doc, Truncation::LongestFirst, 64)
            .unwrap();
        let q_ids = packed_stream(&split, 0, 1);
        let d_ids = packed_stream(&split, 1, 2);

        let mut via_text = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
        engine
            .pack_text(
                &mut via_text,
                0,
                query,
                doc,
                Truncation::QueryPriority,
                max_length,
            )
            .unwrap();
        let mut via_ids = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
        via_ids
            .pack_ids(0, &q_ids, &d_ids, Truncation::QueryPriority, max_length)
            .unwrap();

        assert_eq!(
            via_text.input_ids(),
            via_ids.input_ids(),
            "QP rows diverge for ({query:?}, {doc:?}, ml={max_length})"
        );
        assert_eq!(via_text.attention_mask(), via_ids.attention_mask());
        assert_eq!(via_text.token_type_ids(), via_ids.token_type_ids());
    }
}

#[test]
fn pack_text_longest_first_split_diverges_from_pack_ids() {
    let Some(engine) = load_cpu_engine() else {
        return;
    };
    // SUSPECTED PRODUCT BUG, pinned as current behavior: with a query-longer
    // pair over budget, `pack_text` (wordpiece_pack_pair split formula) keeps
    // MORE query tokens than `pack_ids` (pack_pair_ids iterative loop), and
    // neither matches HF `longest_first` tie-breaking. q=3 tokens, d=2 tokens,
    // max_length=6 -> budget 3:
    //   pack_text: [CLS] hello hello [SEP] tokyo [SEP]  (2 query, 1 doc)
    //   pack_ids:  [CLS] hello [SEP] tokyo tokyo [SEP]  (1 query, 2 doc)
    let query = "hello hello hello";
    let doc = "tokyo tokyo";

    let mut via_text = TokenBuffer::alloc(Device::Cpu, 1, 8).unwrap();
    engine
        .pack_text(&mut via_text, 0, query, doc, Truncation::LongestFirst, 6)
        .unwrap();
    assert_row(
        &via_text,
        0,
        &[CLS, 7592, 7592, SEP, 5522, SEP],
        &[1; 6],
        &[0, 0, 0, 0, 1, 1],
    );

    let [ids, mask, _, _] = pack_row_ffi(
        &[7592, 7592, 7592],
        &[5522, 5522],
        8,
        ffi::turborerank_truncation::TURBORERANK_TRUNC_LONGEST_FIRST,
        6,
    );
    assert_eq!(&ids[..6], &[CLS, 7592, SEP, 5522, 5522, SEP]);
    assert_eq!(&mask[..6], &[1; 6]);
}

#[test]
fn pack_text_rejects_both_empty_and_out_of_range_row() {
    let Some(engine) = load_cpu_engine() else {
        return;
    };
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 16).unwrap();
    let err = engine
        .pack_text(&mut buf, 0, "", "", Truncation::LongestFirst, 16)
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");

    let err = engine
        .pack_text(&mut buf, 1, "hello", "world", Truncation::LongestFirst, 16)
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
}
