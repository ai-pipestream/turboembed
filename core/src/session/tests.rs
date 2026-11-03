use super::*;

fn zero() -> turbo_embed_options {
    turbo_embed_options {
        struct_size: size_of::<turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: 0,
        normalize: 0,
        pooling: 0,
        output_dim: 0,
        quantization: 0,
        int8_ranges: std::ptr::null(),
        layers: 0,
        reserved: 0,
    }
}

fn desc(cpu_threads: u32, max_ranges: u32) -> turbo_session_desc {
    turbo_session_desc {
        struct_size: size_of::<turbo_session_desc>() as u32,
        max_batch: 0,
        max_seq: 0,
        precision: 0,
        tuning: 0,
        tuning_budget_ms: 0,
        cpu_threads,
        max_ranges,
    }
}

#[test]
fn max_ranges_is_refused_where_no_range_write_is_offered() {
    let resolve = |open, ranges, d: &turbo_session_desc| resolve_threads_and_ranges("x", open, ranges, d, 4, 8);
    // Both functions: 0 is max_batch, anything up to max_batch * max_seq is taken.
    assert_eq!(resolve(true, true, &desc(0, 0)).unwrap(), 4);
    assert_eq!(resolve(true, true, &desc(0, 32)).unwrap(), 32);
    let e = resolve(true, true, &desc(0, 33)).unwrap_err();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 7));
    // session_open alone, or neither: max_batch, given or as 0, and nothing else.
    for open in [true, false] {
        assert_eq!(resolve(open, false, &desc(0, 0)).unwrap(), 4);
        assert_eq!(resolve(open, false, &desc(0, 4)).unwrap(), 4);
        let e = resolve(open, false, &desc(0, 32)).unwrap_err();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 7), "open {open}");
    }
    // cpu_threads goes to a backend with session_open, to check.
    assert_eq!(resolve(true, false, &desc(3, 0)).unwrap(), 4);
    let e = resolve(false, false, &desc(3, 0)).unwrap_err();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6));
}

#[test]
fn an_option_a_device_does_not_honor_is_refused_by_its_number() {
    honored(&zero(), 0, "x").unwrap();
    let o = turbo_embed_options { pooling: TURBO_POOLING_CLS, output_dim: 8, ..zero() };
    // Only the fields the core applies: pooling, field 5, comes first.
    let e = honored(&o, CORE_HONORED, "x").unwrap_err();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 5));
    assert_eq!(e.message, "pooling 2: the x backend does not honor it at this precision");
    let e = honored(&o, CORE_HONORED | 1 << 4, "x").unwrap_err();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6));
    honored(&o, CORE_HONORED | 0b110000, "x").unwrap();
    let o = turbo_embed_options { max_tokens: 3, ..zero() };
    assert_eq!(honored(&o, 0, "x").unwrap_err().field, 2);
}

#[test]
fn a_layer_count_is_judged_against_the_model_before_the_device() {
    use crate::backend::{TURBO_FAMILY_BERT, TURBO_FAMILY_NOMIC_BERT, TURBO_FAMILY_ROBERTA, TURBO_FAMILY_STATIC};
    let enc = |family| Taps { layers: 2, family, span_head: false };
    let code = |taps, model, read| taps_refusal(taps, model, read, "x").map_err(|e| (e.code, e.field));
    for family in [TURBO_FAMILY_BERT, TURBO_FAMILY_ROBERTA, TURBO_FAMILY_NOMIC_BERT] {
        for read in [true, false] {
            // 0 and over the model's layers: the request's fault on every device.
            assert_eq!(code(0, enc(family), read), Err((INVALID_ARGUMENT, 9)), "family {family}, read {read}");
            assert_eq!(code(3, enc(family), read), Err((INVALID_ARGUMENT, 9)), "family {family}, read {read}");
            assert_eq!(code(1, enc(family), read), Ok(()), "the final layer alone, on any device");
        }
        // Within the model's layers: the device's, by its cell's bit.
        assert_eq!(code(2, enc(family), true), Ok(()));
        assert_eq!(code(2, enc(family), false), Err((UNSUPPORTED_OPTION, 9)));
        // A span head on the device reads the final layer, after the count.
        let headed = Taps { span_head: true, ..enc(family) };
        assert_eq!(code(3, headed, true), Err((INVALID_ARGUMENT, 9)));
        assert_eq!(code(2, headed, true), Err((UNSUPPORTED_OPTION, 9)));
    }
    // A static model has no encoder layers to count: above 1 is the family's.
    let table = Taps { layers: 0, family: TURBO_FAMILY_STATIC, span_head: false };
    for read in [true, false] {
        assert_eq!(code(2, table, read), Err((UNSUPPORTED_OPTION, 9)));
        assert_eq!(code(0, table, read), Err((INVALID_ARGUMENT, 9)));
        assert_eq!(code(1, table, read), Ok(()));
    }
    let e = taps_refusal(3, enc(TURBO_FAMILY_BERT), false, "x").unwrap_err();
    assert_eq!(e.message, "layers 3: the model has 1 to 2 to read");
}
