//! Word features at each of the encoder's last layers (#1616) on the
//! small BERT in testdata (two layers): the last is the plain features,
//! the one below differs, sentences read in windows agree with one read
//! per window, and refusals.

mod common;

use common::*;
use serde_json::json;
use turbo::manifest::WordPooling;
use turbo_analysis::{Document, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::{HeadRunner, PosTagger, WordEncoder};

const LONG: &str = "Ann met Bob in Paris , and the quiet old dog slept by the river while Bob walked the dog .";

#[test]
fn the_last_layer_is_the_plain_features_and_the_one_below_is_not() {
    let words: Vec<&str> = LONG.split(' ').collect();
    let seqs = [words.clone(), TEXT.split(' ').collect()];
    for (pooling, o) in [
        (WordPooling::First, EncoderOptions::default()),
        (WordPooling::Mean, EncoderOptions::default()),
        // Rows of six pieces: the long sentence is read in windows, two a run.
        (WordPooling::Mean, EncoderOptions { max_seq: 8, max_batch: 2, ..EncoderOptions::default() }),
    ] {
        let mut enc = WordEncoder::open(&tiny(), &o).unwrap();
        let plain = enc.features(&seqs, pooling).unwrap();
        let rows = enc.rows();
        let layers = enc.layer_features(&seqs, pooling, 2).unwrap();
        assert_eq!(enc.rows(), 2 * rows, "one read of each window for both layers");
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[1], plain, "{pooling:?} {o:?}");
        assert_eq!(layers[0].iter().map(Vec::len).collect::<Vec<_>>(), plain.iter().map(Vec::len).collect::<Vec<_>>());
        assert_ne!(layers[0], plain);
        let one = enc.layer_features(&seqs, pooling, 1).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0], plain);
        // The encoder goes back to plain reads after.
        assert_eq!(enc.features(&seqs, pooling).unwrap(), plain);

        let mut tapped = WordEncoder::open(&tiny(), &EncoderOptions { layers: 2, ..o.clone() }).unwrap();
        assert_eq!(tapped.taps(), 2);
        let packed = tapped.features(&seqs, pooling).unwrap();
        for (s, seq) in packed.iter().enumerate() {
            let mut want = layers[0][s].clone();
            want.extend_from_slice(&layers[1][s]);
            assert_eq!(seq, &want, "{pooling:?} {o:?} seq {s}");
        }
    }
}

#[test]
fn a_head_with_a_layer_mix_reads_the_tapped_word_features() {
    let dir = bundle("layer-mix", |m| {
        m["heads"][0]["layer_mix"] = json!({"layers": 2});
    });
    let runner = HeadRunner::open(&dir, Some("pos"), "turboembed:pos", &EncoderOptions::default()).unwrap();
    assert_eq!(runner.tagger().taps(), 2);
    let words: Vec<&str> = TEXT.split(' ').collect();
    let decoded = runner.decode(std::slice::from_ref(&words)).unwrap();
    assert_eq!(decoded[0].tags.len(), words.len());
    let pos = PosTagger::open(&dir, Some("pos"), &EncoderOptions::default()).unwrap();
    let doc = Document::new(TEXT);
    let view = doc.get::<View>().unwrap();
    let toks = tokens(TEXT);
    let tags = runner.pos_tags(view, &toks, None, None, pos.tagset()).unwrap();
    assert_eq!(tags.tags.len(), words.len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn layers_outside_the_model_or_a_read_are_refused() {
    let words: Vec<&str> = TEXT.split(' ').collect();
    assert!(WordEncoder::open(&tiny(), &EncoderOptions { layers: 3, ..EncoderOptions::default() }).is_err());
    let mut enc = WordEncoder::open(&tiny(), &EncoderOptions::default()).unwrap();
    let e = enc.layer_features(std::slice::from_ref(&words), WordPooling::First, 0).unwrap_err();
    assert!(e.message.contains("0 layers"), "{e:?}");
    let e = enc.layer_features(std::slice::from_ref(&words), WordPooling::First, 3).unwrap_err();
    assert!(e.message.contains("layer"), "{e:?}");
    let mut small = WordEncoder::open(&tiny(), &EncoderOptions { max_batch: 1, ..EncoderOptions::default() }).unwrap();
    let e = small.layer_features(std::slice::from_ref(&words), WordPooling::First, 2).unwrap_err();
    assert!(e.message.contains("max_batch (1)"), "{e:?}");
    // A refused read leaves the encoder reading plainly.
    assert_eq!(enc.features(std::slice::from_ref(&words), WordPooling::First).unwrap().len(), 1);
}
