//! Safe-API embed surface on the 8-d FNV mock (`Device::Mock`,
//! alias `mock-embed`).
//!
//! Complements the sibling suites: `abi_smoke.rs` (create/list/load/basic FFI
//! paths), `options.rs` (raw FFI option validation), `ownership.rs` (lifetimes,
//! reentrancy, exact input spans), `mock_kill.rs` (catalog aliases never mock).
//! Here the angle is the *mock's stated policy* for each `EmbedOptions` field,
//! `Embeddings` accessor consistency, determinism, and `embed_stream`
//! row/final-flag agreement with `embed`.

use turboembed::{Device, EmbedOptions, Embeddings, Engine, Error, OutputFormat, Pooling};

const ALIAS: &str = "mock-embed";

fn loaded_mock() -> Engine {
    let engine = Engine::create(Device::Mock).expect("create mock engine");
    engine.load_model(ALIAS).expect("load mock-embed");
    engine
}

fn opts() -> EmbedOptions {
    EmbedOptions::default()
}

/// Mock policy: every option a loaded mock model cannot honor fails with
/// `Error::NotImplemented` whose message names the mock path.
fn assert_mock_option_rejection(result: Result<Embeddings, Error>) {
    let err = result.expect_err("mock must reject options it does not implement");
    assert!(
        matches!(err, Error::NotImplemented(_)),
        "expected NotImplemented, got {err:?}"
    );
    assert!(
        err.to_string().contains("mock path"),
        "rejection should name the mock path, got: {err}"
    );
}

fn assert_unit_norm(row: &[f32], text: &str) {
    let sum_sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    assert!(
        (sum_sq.sqrt() - 1.0).abs() < 1e-4,
        "mock rows are L2-normalized ({text}): norm {}",
        sum_sq.sqrt()
    );
}

#[test]
fn edge_case_texts_are_accepted_with_distinct_rows() {
    let engine = loaded_mock();
    let texts = [
        "",
        " \t\n\r ",
        "a\0b",
        "héllo 🦀",
        "mock-embed",
        "trailing nul\0",
        "🦀🦀🦀",
    ];
    let mut rows = Vec::new();
    for text in texts {
        let one = engine
            .embed_one(ALIAS, text, &opts())
            .unwrap_or_else(|e| panic!("embed_one({text:?}) must succeed: {e}"));
        assert_eq!(one.dim(), 8, "mock-embed dim ({text:?})");
        assert_eq!(one.count(), 1, "single text ({text:?})");
        assert_eq!(one.values().len(), 8);
        assert_eq!(
            one.row(0).unwrap(),
            one.values(),
            "row(0) == values ({text:?})"
        );
        rows.push(one.values().to_vec());
    }
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            assert_ne!(
                rows[i], rows[j],
                "distinct byte content must give distinct mock rows: {:?} vs {:?}",
                texts[i], texts[j]
            );
        }
    }
}

#[test]
fn long_input_embeds_deterministically() {
    let engine = loaded_mock();
    let long = "turbo-embed ".repeat(20_000); // 240k chars, far past any token budget
    let first = engine.embed_one(ALIAS, &long, &opts()).expect("long input");
    let second = engine.embed_one(ALIAS, &long, &opts()).expect("long input");
    assert_eq!(first.dim(), 8);
    assert_eq!(first.values(), second.values(), "same bytes, same row");
}

#[test]
fn truncate_to_policy_on_mock() {
    let engine = loaded_mock();
    let text = "truncate policy probe";

    // None and the 0 sentinel both mean "provider default" and are accepted.
    engine
        .embed_one(
            ALIAS,
            text,
            &EmbedOptions {
                truncate_to: None,
                ..opts()
            },
        )
        .expect("truncate_to: None is the provider default");
    engine
        .embed_one(
            ALIAS,
            text,
            &EmbedOptions {
                truncate_to: Some(0),
                ..opts()
            },
        )
        .expect("Some(0) is the documented provider-default sentinel");

    // Any real bound is unsupported on the mock path — even on long input,
    // the option is rejected before content matters.
    for bound in [1_u32, 8, 256, 2_000_000] {
        assert_mock_option_rejection(engine.embed_one(
            ALIAS,
            text,
            &EmbedOptions {
                truncate_to: Some(bound),
                ..opts()
            },
        ));
    }
    let long = "x".repeat(100_000);
    assert_mock_option_rejection(engine.embed_one(
        ALIAS,
        &long,
        &EmbedOptions {
            truncate_to: Some(256),
            ..opts()
        },
    ));
}

#[test]
fn batch_of_one_and_eight_rows_are_consistent() {
    let engine = loaded_mock();

    let single = engine
        .embed(ALIAS, &["singleton"], &opts())
        .expect("batch of 1");
    assert_eq!(single.count(), 1);
    assert_eq!(single.dim(), 8);
    let one = engine
        .embed_one(ALIAS, "singleton", &opts())
        .expect("embed_one");
    assert_eq!(
        single.row(0).unwrap(),
        one.values(),
        "embed_one row must match the same text in a batch"
    );

    let owned: Vec<String> = (0..8)
        .map(|i| format!("batch row {i} with distinct content"))
        .collect();
    let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
    let batch = engine.embed(ALIAS, &texts, &opts()).expect("batch of 8");

    assert_eq!(batch.count(), 8);
    assert_eq!(batch.dim(), 8);
    assert_eq!(batch.values().len(), batch.count() * batch.dim());

    // Concatenated row views reproduce the flat buffer exactly.
    let mut flat = Vec::with_capacity(batch.values().len());
    for i in 0..batch.count() {
        let row = batch.row(i).expect("row in range");
        assert_eq!(row.len(), batch.dim());
        flat.extend_from_slice(row);
    }
    assert_eq!(flat, batch.values(), "rows() must tile values()");
    assert!(
        batch.row(batch.count()).is_none(),
        "row(count) is out of bounds"
    );

    for i in 0..batch.count() {
        for j in (i + 1)..batch.count() {
            assert_ne!(
                batch.row(i).unwrap(),
                batch.row(j).unwrap(),
                "distinct texts must give distinct rows: {i} vs {j}"
            );
        }
    }
}

#[test]
fn pooling_variants_follow_mock_policy() {
    let engine = loaded_mock();
    let text = "pooling policy probe";
    for pooling in [Pooling::Mean, Pooling::Cls, Pooling::Last] {
        assert_mock_option_rejection(engine.embed_one(
            ALIAS,
            text,
            &EmbedOptions { pooling, ..opts() },
        ));
    }
    engine
        .embed_one(
            ALIAS,
            text,
            &EmbedOptions {
                pooling: Pooling::Default,
                ..opts()
            },
        )
        .expect("Pooling::Default is the only accepted variant on the mock");
}

#[test]
fn normalize_variants_follow_mock_policy() {
    let engine = loaded_mock();
    let text = "normalize policy probe";
    for normalize in [Some(true), Some(false)] {
        assert_mock_option_rejection(engine.embed_one(
            ALIAS,
            text,
            &EmbedOptions {
                normalize,
                ..opts()
            },
        ));
    }
    engine
        .embed_one(
            ALIAS,
            text,
            &EmbedOptions {
                normalize: None,
                ..opts()
            },
        )
        .expect("normalize: None (provider default) is accepted on the mock");
}

#[test]
fn packed_bytes_decodes_to_typed_values_in_both_formats() {
    let engine = loaded_mock();
    let text = "packed vs typed 🦀";
    for output_format in [OutputFormat::Typed, OutputFormat::PackedBytes] {
        let result = engine
            .embed_one(
                ALIAS,
                text,
                &EmbedOptions {
                    output_format,
                    ..opts()
                },
            )
            .unwrap_or_else(|e| panic!("{output_format:?} must be accepted on the mock: {e}"));
        let decoded: Vec<f32> = result
            .packed()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
            .collect();
        assert_eq!(
            decoded,
            result.values(),
            "packed() must be little-endian FP32 of values() ({output_format:?})"
        );
    }
    let typed = engine
        .embed_one(
            ALIAS,
            text,
            &EmbedOptions {
                output_format: OutputFormat::Typed,
                ..opts()
            },
        )
        .expect("typed");
    let packed = engine
        .embed_one(
            ALIAS,
            text,
            &EmbedOptions {
                output_format: OutputFormat::PackedBytes,
                ..opts()
            },
        )
        .expect("packed bytes");
    assert_eq!(
        typed.values(),
        packed.values(),
        "output_format is a presentation hint; the floats must not change"
    );
}

#[test]
fn repeated_calls_are_deterministic() {
    let engine = loaded_mock();
    let text = "determinism probe";
    let baseline = engine.embed_one(ALIAS, text, &opts()).expect("first");
    assert_unit_norm(baseline.values(), text);
    for _ in 0..2 {
        let again = engine.embed_one(ALIAS, text, &opts()).expect("repeat");
        assert_eq!(
            baseline.values(),
            again.values(),
            "embed_one must be deterministic"
        );
    }

    let owned: Vec<String> = (0..8).map(|i| format!("repeat batch row {i}")).collect();
    let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
    let first = engine.embed(ALIAS, &texts, &opts()).expect("batch first");
    let second = engine.embed(ALIAS, &texts, &opts()).expect("batch second");
    assert_eq!(
        first.values(),
        second.values(),
        "embed must be deterministic"
    );
}

#[test]
fn stream_rows_match_embed_with_final_flag_only_on_last() {
    let engine = loaded_mock();
    let owned: Vec<String> = (0..8).map(|i| format!("stream row {i} 🦀")).collect();
    let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
    let reference = engine
        .embed(ALIAS, &texts, &opts())
        .expect("batch reference");

    let mut seen: Vec<(u32, Vec<f32>, bool)> = Vec::new();
    let streamed = engine
        .embed_stream(ALIAS, &texts, &opts(), |index, row, is_final| {
            seen.push((index, row.to_vec(), is_final));
        })
        .expect("embed_stream");

    assert_eq!(seen.len(), texts.len(), "one callback per row");
    for (expected_index, (index, row, _)) in seen.iter().enumerate() {
        assert_eq!(
            *index, expected_index as u32,
            "callback indices must be ascending"
        );
        assert_eq!(
            row,
            reference.row(expected_index).unwrap(),
            "stream row == embed row"
        );
        assert_eq!(
            row,
            streamed.row(expected_index).unwrap(),
            "stream row == returned result row"
        );
    }
    let finals: Vec<u32> = seen
        .iter()
        .filter(|(_, _, is_final)| *is_final)
        .map(|(i, _, _)| *i)
        .collect();
    assert_eq!(finals, [7], "exactly one final flag, on the last row");
    assert_eq!(
        streamed.values(),
        reference.values(),
        "stream result equals embed result"
    );

    // Single-text stream: the only callback is already final.
    let lone = engine
        .embed_one(ALIAS, "lone", &opts())
        .expect("lone reference");
    let mut single = Vec::new();
    engine
        .embed_stream(ALIAS, &["lone"], &opts(), |index, row, is_final| {
            single.push((index, row.to_vec(), is_final));
        })
        .expect("single stream");
    assert_eq!(single, vec![(0_u32, lone.values().to_vec(), true)]);
}

#[test]
fn unknown_alias_fails_and_never_calls_back() {
    let engine = loaded_mock();

    // Loading an unknown alias on a Mock engine is refused before any
    // provider path (feature-independent: the mock-device branch answers).
    let err = engine
        .load_model("no-such-alias")
        .expect_err("unknown alias load");
    assert!(
        matches!(err, Error::NotImplemented(_)),
        "unknown alias load on Mock: {err:?}"
    );
    assert!(
        err.to_string().contains("not served by mock"),
        "mock-device refusal should say why, got: {err}"
    );

    // Embedding an unknown alias: the plain stub answers NOT_IMPLEMENTED;
    // provider builds answer NOT_FOUND ("call load_model first").
    for result in [
        engine
            .embed_one("no-such-alias", "x", &opts())
            .map(|e| e.values().to_vec()),
        engine
            .embed("no-such-alias", &["x", "y"], &opts())
            .map(|e| e.values().to_vec()),
    ] {
        let err = result.expect_err("unknown alias embed");
        #[cfg(not(any(feature = "ort-cuda", feature = "genai")))]
        assert!(
            matches!(err, Error::NotImplemented(_)),
            "plain stub must refuse unknown alias embed, got {err:?}"
        );
        #[cfg(any(feature = "ort-cuda", feature = "genai"))]
        assert!(
            matches!(err, Error::NotFound(_)),
            "provider build must report unknown alias as not loaded, got {err:?}"
        );
    }

    // The same refusal applies to the streaming entry point, and a failed
    // call must not invoke the callback at all.
    let mut called = 0_u32;
    let err = engine
        .embed_stream("no-such-alias", &["x"], &opts(), |_, _, _| called += 1)
        .expect_err("unknown alias stream");
    #[cfg(not(any(feature = "ort-cuda", feature = "genai")))]
    assert!(matches!(err, Error::NotImplemented(_)), "stream: {err:?}");
    #[cfg(any(feature = "ort-cuda", feature = "genai"))]
    assert!(matches!(err, Error::NotFound(_)), "stream: {err:?}");
    assert_eq!(called, 0, "a failed stream must not fire callbacks");
}

#[test]
fn load_model_twice_stays_ready_and_deterministic() {
    let engine = Engine::create(Device::Mock).expect("create mock engine");
    engine.load_model(ALIAS).expect("first load");
    let first = engine
        .embed_one(ALIAS, "after first load", &opts())
        .expect("embed");
    engine.load_model(ALIAS).expect("second load is idempotent");
    let second = engine
        .embed_one(ALIAS, "after first load", &opts())
        .expect("embed");
    assert_eq!(
        first.values(),
        second.values(),
        "reload must not change mock rows"
    );
    let models = engine.list_models().expect("list after double load");
    assert_eq!(models.len(), 1);
    assert!(
        models.get(0).expect("mock row").ready,
        "mock-embed stays ready"
    );
}
