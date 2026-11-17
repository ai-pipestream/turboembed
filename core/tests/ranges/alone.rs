//! turbo_embed_options.layer_alone (field 10): a range write of the last
//! K layers that asks for the first of them alone gives block 0 of the
//! K-layer write bit for bit, a vector a range, on every encoder family;
//! it is the plain range write of the model cut after that layer; count
//! alone is held to max_ranges; and the refusals turbo.h lists, with the
//! bit clear on a device that reads every one of the layers.

use std::path::Path;

use super::layers::devices;
use crate::common::*;
use crate::support::*;
use serde_json::json;
use turbo::status::*;
use turbo::*;

/// Backends whose embed cell sets field 10's bit.
const READS_ONE_ALONE: [&str; 2] = ["cpu", "cuda"];

fn alone(layers: u32) -> turbo_embed_options {
    turbo_embed_options { layers, layer_alone: 1, ..embed_options() }
}

fn all_bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter().map(|v| bits(v)).collect()
}

/// For every count of layers on the model at `dir`, of `deep` layers,
/// on the session `s`: one layer alone is block 0 of that layer write,
/// normalized and not, and one vector a range.
fn first_block(s: &Session, dir: &Path, deep: u32, what: &str) {
    let (t, lens) = tokens_of(dir, &RANGES_TEXTS);
    let ranges = [range(0, 0, lens[0] as u32), range(1, 1, 4), range(2, 0, 2)];
    for (layers, normalize) in (1..=deep).flat_map(|k| [0, TURBO_NORMALIZE_NONE].map(|n| (k, n))) {
        let every = run_ranges(s, &t, &ranges, &turbo_embed_options { layers, normalize, ..embed_options() });
        let one = run_ranges(s, &t, &ranges, &turbo_embed_options { normalize, ..alone(layers) });
        assert_eq!(one.len(), 3, "{what}: {layers} layers: a vector a range");
        assert_eq!(all_bits(&one), all_bits(&every[..3]), "{what}: {layers} layers, normalize {normalize}");
    }
}

#[test]
fn one_layer_alone_is_the_first_block_of_the_layer_write() {
    for (bundle, deep) in
        [("tiny-bert-bundle", 2), ("tiny-modernbert-bundle", 3), ("tiny-deberta-bundle", 2), ("tiny-nomic-bundle", 2)]
    {
        let dir = testdata().join(bundle);
        let l = Loaded::load(&dir).unwrap_or_else(|e| panic!("{bundle}: {e:?}"));
        // max_ranges of 3: one layer alone needs no more, whatever K.
        first_block(&session_of(&l, 3 * deep), &dir, deep, bundle);
    }
}

/// The small BERT's first layer alone against the same weights sealed as
/// a model of one layer (architecture.layers as a cut, docs/bundle.md).
#[test]
fn one_layer_alone_is_the_model_cut_after_that_layer() {
    let sealed = |name: &str, layers: u32| {
        let mut m = model_manifest();
        m["architecture"]["layers"] = json!(layers);
        let mut f = Fixture::new(name, m);
        f.weights("weights/model.safetensors", &tiny_weights(0));
        let l = Loaded::load(f.write()).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        (f, l)
    };
    let ((whole_dir, whole), (_cut_dir, cut)) = (sealed("alone-whole", 2), sealed("alone-cut", 1));
    let (t, lens) = tokens_of(whole_dir.write(), &RANGES_TEXTS);
    let ranges = [range(0, 0, lens[0] as u32), range(1, 1, 4), range(2, 0, 2)];
    let one = run_ranges(&session_of(&whole, 3), &t, &ranges, &alone(2));
    let plain = run_ranges(&session_of(&cut, 3), &t, &ranges, &embed_options());
    assert_eq!(all_bits(&one), all_bits(&plain));
}

#[test]
fn layer_alone_is_refused_where_turbo_h_says() {
    let l = tiny();
    let s = session_of(&l, 3);
    let (t, lens) = tokens_of(&tiny_bundle(), &RANGES_TEXTS);
    let ranges = [range(0, 0, lens[0] as u32), range(1, 1, 4), range(2, 0, 2)];
    let field = |e: Failure| (e.code, e.field);

    // 0 or 1 alone.
    let o = turbo_embed_options { layers: 2, layer_alone: 2, ..embed_options() };
    assert_eq!(field(s.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap_err()), (INVALID_ARGUMENT, 10));
    // Row writes are the final layer's.
    assert_eq!(field(s.write_text(&["hello"], Some(&alone(0))).unwrap_err()), (INVALID_ARGUMENT, 10));
    assert_eq!(field(s.write_tokens(&t.batch(), Some(&alone(1))).unwrap_err()), (INVALID_ARGUMENT, 10));
    // Whatever layers is refused for comes first.
    assert_eq!(field(s.write_ranges(&t.batch(), &ranges, Some(&alone(3))).unwrap_err()), (INVALID_ARGUMENT, 9));

    // Three ranges from two layers are six vectors, over max_ranges 3;
    // the first of the two alone is three.
    let o = turbo_embed_options { layers: 2, ..embed_options() };
    assert_eq!(s.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap_err().code, CAPACITY);
    s.write_layer_ranges(&t.batch(), &ranges, 2, Some(&alone(2))).unwrap();
    assert_eq!(s.run().unwrap().info().batch, 3);

    // With layers 0 or 1 it is the final layer, as without it.
    let plain = run_ranges(&s, &t, &ranges, &embed_options());
    for layers in [0, 1] {
        assert_eq!(all_bits(&run_ranges(&s, &t, &ranges, &alone(layers))), all_bits(&plain), "layers {layers}");
    }

    // A static model has its table alone: layers is refused first.
    let static_dir = testdata().join("tiny-static-bundle");
    let st = Loaded::load(&static_dir).unwrap_or_else(|e| panic!("{e:?}"));
    let s = session_of(&st, 4);
    let (t, _) = tokens_of(&static_dir, &RANGES_TEXTS);
    let two = [range(0, 0, 2), range(1, 0, 3)];
    assert_eq!(field(s.write_ranges(&t.batch(), &two, Some(&alone(2))).unwrap_err()), (UNSUPPORTED_OPTION, 9));
}

/// Each device this build lists: its cell sets field 10's bit exactly
/// when it is in READS_ONE_ALONE; where set, one layer alone is the
/// device's own block 0 at MODEL and FASTEST on every family; where
/// clear, the write is TURBO_E_UNSUPPORTED_OPTION naming field 10.
#[test]
fn one_layer_alone_on_each_device() {
    let dir = tiny_bundle();
    let (t, _) = tokens_of(&dir, &RANGES_TEXTS);
    let two = [range(0, 0, 2), range(1, 0, 3)];
    let mut backends = devices();
    backends.push("cpu");
    for backend in backends {
        let l = Loaded::load_on(&dir, |rt| first_of(rt, backend).expect("listed"))
            .unwrap_or_else(|e| panic!("{backend}: {e:?}"));
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        let at = first_of(l.rt, backend).unwrap();
        let rc = unsafe {
            turbo_runtime_capability(l.rt, at, TURBO_TASK_EMBED, TURBO_PRECISION_MODEL, &mut cap, &mut new_error())
        };
        assert_eq!(rc, 0, "{backend}");
        let reads = cap.options_honored & 1 << 9 != 0;
        assert_eq!(reads, READS_ONE_ALONE.contains(&backend), "{backend}: which backends read one layer alone");
        if !reads {
            // A device that reads the layers but not one alone names
            // field 10; one that reads neither names field 9 first.
            let e = session_of(&l, 4).write_ranges(&t.batch(), &two, Some(&alone(2))).unwrap_err();
            let named = if cap.options_honored & 1 << 8 != 0 { 10 } else { 9 };
            assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, named), "{backend}: {e:?}");
            continue;
        }
        if backend == "cpu" {
            continue;
        }
        for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
            for (bundle, deep) in [
                ("tiny-bert-bundle", 2),
                ("tiny-modernbert-bundle", 3),
                ("tiny-deberta-bundle", 2),
                ("tiny-nomic-bundle", 2),
            ] {
                let dir = testdata().join(bundle);
                let what = format!("{backend}, precision {precision}, {bundle}");
                let l = Loaded::load_on(&dir, |rt| first_of(rt, backend).unwrap())
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let mut d = session_desc(0, 0, precision);
                d.max_ranges = 3 * deep;
                // A backend that runs a family in F32 alone refuses
                // FASTEST for it, naming the precision.
                match Session::create(l.m, Some(&d)) {
                    Ok(s) => first_block(&s, &dir, deep, &what),
                    Err(e) if precision == TURBO_PRECISION_FASTEST && (e.code, e.field) == (UNSUPPORTED_OPTION, 3) => {
                        println!("{what}: skipped: {e:?}");
                    }
                    Err(e) => panic!("{what}: {e:?}"),
                }
            }
        }
    }
}
