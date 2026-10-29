//! turbo_embed_write_layer_ranges: each range from each of the last
//! layers in one pass, its last block the plain range write's bit for bit
//! on the sealed BERT and ModernBERT, the run's shape, and the refusals
//! turbo.h lists. Each block against a model cut after its layer is the
//! cpu encoder's own test (cpu/encoder/taps_tests.rs).

use crate::common::*;
use crate::support::*;
use turbo::status::*;
use turbo::*;

/// The ranges' vectors from the last `layers` layers, and from the plain
/// range write, on the bundle in testdata named `bundle`, of `deep` layers.
fn agree(bundle: &str, deep: u32) {
    let dir = testdata().join(bundle);
    let l = Loaded::load(&dir).unwrap_or_else(|e| panic!("{e:?}"));
    let (t, lens) = tokens_of(&dir, &RANGES_TEXTS);
    let ranges = [range(0, 0, lens[0] as u32), range(1, 1, 4), range(2, 0, 2)];
    let s = session_of(&l, 3 * deep);
    let plain = run_ranges(&s, &t, &ranges, &embed_options());
    for layers in 1..=deep {
        s.write_layer_ranges(&t.batch(), &ranges, layers, None).unwrap_or_else(|e| panic!("{e:?}"));
        let r = s.run().unwrap();
        assert_eq!(r.info().batch, 3 * layers, "{bundle}: {layers} layers");
        let rows = r.rows();
        let last = &rows[3 * (layers as usize - 1)..];
        assert_eq!(last.iter().map(|v| bits(v)).collect::<Vec<_>>(), plain.iter().map(|v| bits(v)).collect::<Vec<_>>());
        for (k, block) in rows.chunks(3).enumerate().take(layers as usize - 1) {
            assert!(block.iter().flatten().all(|v| v.is_finite()), "{bundle}: tap {k}");
            assert_ne!(block, last, "{bundle}: tap {k} of {layers} is an earlier layer's");
        }
    }
}

#[test]
fn the_last_layer_read_is_the_plain_range_write() {
    agree("tiny-bert-bundle", 2);
    agree("tiny-modernbert-bundle", 3);
}

#[test]
fn a_layer_range_write_refuses_what_it_cannot_take() {
    let l = tiny();
    let s = session_of(&l, 4);
    let (t, _) = tokens_of(&tiny_bundle(), &RANGES_TEXTS);
    let good = [range(0, 0, 2), range(1, 0, 3)];
    let refused = |ranges: &[turbo_token_range], layers: u32| -> Failure {
        s.write_ranges(&t.batch(), &good, None).unwrap();
        let e = s.write_layer_ranges(&t.batch(), ranges, layers, None).expect_err("the write is refused");
        assert_eq!(s.run().err().unwrap().code, INVALID_STATE, "{e:?}");
        e
    };
    let e = refused(&good, 0);
    assert_eq!(e.code, INVALID_ARGUMENT, "{e:?}");
    let e = refused(&good, 3);
    assert_eq!(e.code, INVALID_ARGUMENT, "past the model's 2 layers: {e:?}");
    assert!(e.message.contains("1 to 2"), "{e:?}");
    let e = refused(&[good[0], good[1], good[0]], 2);
    assert_eq!(e.code, CAPACITY, "six vectors over max_ranges 4: {e:?}");
    assert!(e.message.contains("is 6 vectors"), "{e:?}");
    s.write_layer_ranges(&t.batch(), &good, 2, None).unwrap();
    assert_eq!(s.run().unwrap().info().batch, 4);

    // The cpu DeBERTa encoder reads only its final layer.
    let dir = testdata().join("tiny-deberta-bundle");
    let d = Loaded::load(&dir).unwrap_or_else(|e| panic!("{e:?}"));
    let s = session_of(&d, 4);
    let (t, _) = tokens_of(&dir, &RANGES_TEXTS);
    let e = s.write_layer_ranges(&t.batch(), &good, 2, None).expect_err("deberta");
    assert_eq!(e.code, UNSUPPORTED_OPTION, "{e:?}");
    s.write_layer_ranges(&t.batch(), &good, 1, None).unwrap();
}
