//! Malformed checkpoints and inputs at the GLiNER2 scorer's public
//! boundary: each is refused with an error, or for a wrong-length slice
//! a panic before any kernel reads it, never a wrong answer or an
//! out-of-bounds read. Built from the reduced fixture, with one tensor or
//! one field changed.

use serde_json::{Map, Value};
use turbo::deberta::{Config, Encoder};
use turbo::safetensors::File;
use turbo_analysis::Span;
use turbo_relation::gliner2::heads::Heads;
use turbo_relation::gliner2::prompt::Prompt;
use turbo_relation::gliner2::words::{Word, split};
use turbo_relation::gliner2::{LabelScores, Pass, decode, run, span_of};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gliner2");

fn fixture() -> Vec<u8> {
    std::fs::read(format!("{DIR}/reduced.safetensors")).unwrap()
}

/// The fixture with `edit` applied to each tensor's name, shape and data,
/// written back out as safetensors.
fn rewrite(bytes: &[u8], edit: impl Fn(&str, &mut Vec<u64>, &mut Vec<u8>)) -> Vec<u8> {
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Map<String, Value> = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
    let data = &bytes[8 + len..];
    let (mut out_header, mut out_data) = (Map::new(), Vec::new());
    for (name, t) in &header {
        if name == "__metadata__" {
            continue;
        }
        let mut shape: Vec<u64> = t["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        let o: Vec<usize> =
            t["data_offsets"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let mut bytes = data[o[0]..o[1]].to_vec();
        edit(name, &mut shape, &mut bytes);
        let start = out_data.len();
        out_data.extend_from_slice(&bytes);
        let entry = serde_json::json!({"dtype": t["dtype"], "shape": shape, "data_offsets": [start, out_data.len()]});
        out_header.insert(name.clone(), entry);
    }
    let mut head = serde_json::to_vec(&out_header).unwrap();
    // The header is padded with spaces to a multiple of 8 bytes.
    head.resize(head.len().div_ceil(8) * 8, b' ');
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    out.extend_from_slice(&out_data);
    out
}

fn load(bytes: &[u8]) -> Result<Heads, String> {
    Heads::load(&File::parse("edited", bytes).unwrap(), 32, 8)
}

const OUT_PROJ: &str = "count_embed.transformer.transformer.layers.0.self_attn.out_proj.weight";

#[test]
fn the_unedited_rewrite_loads() {
    assert!(load(&rewrite(&fixture(), |_, _, _| {})).is_ok());
}

#[test]
fn an_attention_output_projection_of_another_width_is_refused() {
    // [128, 128] cut to its first 64 columns, the 128 biases kept.
    let bytes = rewrite(&fixture(), |name, shape, data| {
        if name == OUT_PROJ {
            *data = data.as_chunks::<{ 128 * 4 }>().0.iter().flat_map(|row| row[..64 * 4].to_vec()).collect();
            shape[1] = 64;
        }
    });
    assert!(load(&bytes).is_err());
}

#[test]
fn non_finite_norm_and_position_values_are_refused() {
    for target in [
        "count_embed.transformer.transformer.layers.0.norm1.weight",
        "count_embed.transformer.transformer.layers.1.norm2.bias",
        "count_embed.pos_embedding.weight",
    ] {
        let bytes = rewrite(&fixture(), |name, _, data| {
            if name == target {
                data[..4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
        });
        assert!(load(&bytes).is_err(), "{target} with a NaN");
    }
}

#[test]
fn a_norm_of_another_length_is_refused() {
    let bytes = rewrite(&fixture(), |name, shape, data| {
        if name.starts_with("count_embed.transformer.transformer.layers.1.norm1.") {
            data.truncate(64 * 4);
            shape[0] = 64;
        }
    });
    assert!(load(&bytes).is_err());
}

fn heads() -> Heads {
    load(&fixture()).unwrap()
}

/// No word rows for eight words: the length check stops it before the
/// kernels, whatever the processor.
#[test]
#[should_panic(expected = "word rows of the model width")]
fn spans_of_too_few_rows_panic_before_the_kernels() {
    heads().spans(&[], 8);
}

#[test]
#[should_panic(expected = "word rows of the model width")]
fn spans_of_a_partial_row_panic_before_the_kernels() {
    heads().spans(&[0.5; 33], 1);
}

#[test]
#[should_panic(expected = "of the model width")]
fn a_short_count_vector_panics() {
    heads().count_logits(&[0.0; 31]);
}

#[test]
#[should_panic(expected = "queries of the model width")]
fn queries_of_a_partial_row_panic() {
    let h = heads();
    let spans = h.spans(&[0.1; 64], 2);
    h.logits_all(&spans, &[0.0; 40]);
}

fn encoder() -> Encoder {
    let config = Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 16,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let canonical = turbo_bundle::deberta::convert(&config, &fixture(), "encoder.").unwrap();
    Encoder::from_safetensors(config, &canonical).unwrap()
}

/// Reduced case 0: one relation, five words.
fn prompt() -> Prompt<'static> {
    let text: std::borrow::Cow<'static, str> = "a b c d e.".into();
    let words = split("a b c d e.")[..5].to_vec();
    Prompt {
        ids: vec![5, 3, 9, 1, 4, 2, 4, 6, 7, 8, 11, 12, 13, 14, 10],
        text,
        words,
        word_first: vec![10, 11, 12, 13, 14],
        markers: vec![[1, 4, 6]],
    }
}

#[test]
fn inconsistent_prompts_are_errors() {
    let (e, h) = (encoder(), heads());
    assert!(run(&e, &h, prompt()).is_ok());
    let mut extra = prompt();
    extra.words.push(Word { text: ".".into(), start: 9, end: 10, subword_len: 0 });
    assert!(run(&e, &h, extra).is_err(), "a word without a first subword");
    let mut past = prompt();
    past.word_first[4] = 15;
    assert!(run(&e, &h, past).is_err(), "a first subword past the input");
    let mut off = prompt();
    off.words[4].end = 40;
    assert!(run(&e, &h, off).is_err(), "a word past the text");
    let mut order = prompt();
    order.words.swap(0, 1);
    assert!(run(&e, &h, order).is_err(), "words out of order");
}

#[test]
fn passes_that_do_not_fit_together_are_errors() {
    let pass = run(&encoder(), &heads(), prompt()).unwrap();
    assert!(decode(&pass, 0.5).is_ok());
    let rebuild = |width: usize, labels: Vec<LabelScores>| Pass { prompt: prompt(), width, labels };
    let copy = |p: &Pass| p.labels.iter().map(|l| LabelScores { count: l.count, logits: l.logits.clone() }).collect();
    assert!(decode(&rebuild(4, copy(&pass)), 0.5).is_err(), "another span width");
    assert!(decode(&rebuild(0, copy(&pass)), 0.5).is_err(), "no span width");
    let mut more = copy(&pass);
    more[0].count += 1;
    assert!(decode(&rebuild(8, more), 0.5).is_err(), "a count with no logits");
    assert!(decode(&rebuild(8, Vec::new()), 0.5).is_err(), "no scores for a marker");
}

/// Finite logits in the slots of spans past the last word: the lengths
/// fit, but no span is there, so none is decoded.
#[test]
fn logits_past_the_last_word_are_not_spans() {
    let prompt = Prompt {
        ids: vec![1],
        text: "a".into(),
        words: vec![Word { text: "a".into(), start: 0, end: 1, subword_len: 1 }],
        word_first: vec![0],
        markers: vec![[0, 0, 0]],
    };
    let labels = vec![LabelScores { count: 1, logits: vec![0.0, 10.0, 0.0, 10.0] }];
    let found = decode(&Pass { prompt, width: 2, labels }, 0.5).unwrap();
    // Only the one-word span is there, at a score of 0.5.
    assert_eq!(found.len(), 1);
    assert_eq!((found[0].head.start, found[0].head.end, found[0].head.score), (0, 1, 0.5));
}

#[test]
fn only_whole_word_runs_are_spans() {
    let text = "New York, São Paulo";
    let words = split(text);
    let at = |s: u32, e: u32| span_of(&words, Span::new(s, e), 8);
    assert_eq!(at(0, 3), Some(0));
    assert_eq!(at(0, 8), Some(1));
    assert_eq!(at(4, 8), Some(8));
    // Inside "New" through the end of "York".
    assert_eq!(at(1, 8), None);
    // "New Y": part of a word at the end.
    assert_eq!(at(0, 5), None);
    // Leading or trailing whitespace is not a word boundary.
    assert_eq!(at(3, 8), None);
    assert_eq!(at(0, 4), None);
    // "São" is four bytes; ending inside it is not a span.
    assert_eq!(at(10, 14), Some(3 * 8));
    assert_eq!(at(10, 12), None);
    assert_eq!(at(10, 20), Some(3 * 8 + 1));
}
