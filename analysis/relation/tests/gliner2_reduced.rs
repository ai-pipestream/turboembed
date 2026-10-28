//! The GLiNER2 relation heads against the reference's own layer classes,
//! on the reduced fixture `reference/gliner2/build_fixture.py` writes: the
//! encoder and heads of `fastino/gliner2-base-v1` sliced to width 32, fixed
//! token sequences, and the reference's count logits, instance counts,
//! queries (for three instances and for the predicted count) and span
//! logits. The tolerance follows the encoder's own fixture.

use serde_json::Value;
use turbo::deberta::{Config, Encoder};
use turbo::safetensors::File;
use turbo_relation::gliner2::heads::Heads;
use turbo_relation::gliner2::prompt::Prompt;
use turbo_relation::gliner2::run;
use turbo_relation::gliner2::words::Word;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gliner2");

fn config() -> Config {
    Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 16,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    }
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().map_or(f32::NEG_INFINITY, |f| f as f32)).collect()
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w == f32::NEG_INFINITY {
            assert_eq!(g, w, "{what}[{i}]");
        } else {
            assert!((g - w).abs() <= 2e-5 * w.abs().max(1.0), "{what}[{i}]: {g} against {w}");
        }
    }
}

#[test]
fn the_heads_match_the_reference_layers() {
    let bytes = std::fs::read(format!("{DIR}/reduced.safetensors")).unwrap();
    let canonical = turbo_bundle::deberta::convert(&config(), &bytes, "encoder.").unwrap();
    let encoder = Encoder::from_safetensors(config(), &canonical).unwrap();
    let heads = Heads::load(&File::parse("reduced", &bytes).unwrap(), 32, 8).unwrap();
    let reference: Value = serde_json::from_slice(&std::fs::read(format!("{DIR}/reduced.json")).unwrap()).unwrap();
    let fixed = reference["fixed_count"].as_u64().unwrap() as usize;
    let mut instances = 0;
    for (c, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let ids: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let words: Vec<usize> =
            case["words"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let markers: Vec<[usize; 3]> = case["markers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let m: Vec<usize> = m.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                [m[0], m[1], m[2]]
            })
            .collect();
        let hidden = encoder.encode(&ids, &vec![true; ids.len()]).unwrap();
        let row = |p: usize| &hidden.data()[p * 32..(p + 1) * 32];
        let word_rows: Vec<f32> = words.iter().flat_map(|&p| row(p).iter().copied()).collect();
        let spans = heads.spans(&word_rows, words.len());
        for (l, label) in case["labels"].as_array().unwrap().iter().enumerate() {
            let [p, h, t] = markers[l];
            let what = format!("case {c} label {l}");
            close(&heads.count_logits(row(p)), &floats(&label["count_logits"]), &format!("{what} count logits"));
            let count = heads.count(row(p));
            assert_eq!(count as u64, label["count"].as_u64().unwrap(), "{what} count");
            let queries = heads.queries([row(h), row(t)], fixed);
            close(&queries, &floats(&label["queries"]), &format!("{what} queries"));
            let logits: Vec<f32> = queries.as_chunks::<32>().0.iter().flat_map(|q| heads.logits(&spans, q)).collect();
            close(&logits, &floats(&label["logits"]), &format!("{what} logits"));
            close(
                &heads.queries([row(h), row(t)], count),
                &floats(&label["actual_queries"]),
                &format!("{what} actual"),
            );
            // Two unlike fields, so the attention across them shows.
            let crossed = heads.queries([row(p), row(words[0])], fixed);
            close(&crossed, &floats(&label["crossed_queries"]), &format!("{what} crossed queries"));
            instances += count;
        }
        // The same through the pass the scorer runs.
        let text_words =
            words.iter().map(|&p| Word { text: "".into(), start: p, end: p + 1, subword_len: 1 }).collect();
        let text = "x".repeat(ids.len() + 1);
        let prompt = Prompt { ids, text: text.into(), words: text_words, word_first: words.clone(), markers };
        let pass = run(&encoder, &heads, prompt).unwrap();
        for (l, label) in case["labels"].as_array().unwrap().iter().enumerate() {
            assert_eq!(pass.labels[l].count as u64, label["count"].as_u64().unwrap());
            let size = words.len() * 8;
            assert_eq!(pass.labels[l].logits.len(), pass.labels[l].count * 2 * size);
        }
    }
    assert!(instances > 0, "the fixture exercises at least one predicted instance");
}

#[test]
fn heads_from_a_checkpoint_of_another_width_are_refused() {
    let bytes = std::fs::read(format!("{DIR}/reduced.safetensors")).unwrap();
    let file = File::parse("reduced", &bytes).unwrap();
    assert!(Heads::load(&file, 48, 8).is_err());
    assert!(Heads::load(&file, 32, 0).is_err());
}
