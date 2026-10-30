//! `turbo-train-head cache` on the small BERT in testdata (two layers)
//! and a small corpus in MultiCoNER's format: every split's word features
//! at the last layers, read back as the encoder gives them; heads trained
//! on one cached layer and on a mix (`fit-cache`); and refusals.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo::manifest::WordPooling;
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::cache::{CacheRecord, SCHEMA, layer_file, read_layer};
use turbo_trainer_recipes::fit_cache::{FitRecord, SCHEMA as SCHEMA_FIT};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-train-head-cache-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn corpus(dir: &Path) {
    let mut text = String::new();
    for i in 0..20 {
        text += &format!(
            "# id {i}\tdomain=en\nthey _ _ O\nmet _ _ O\nann _ _ B-OtherPER\nin _ _ O\nrome _ _ B-HumanSettlement\n\n"
        );
    }
    fs::create_dir_all(dir.join("EN-English")).unwrap();
    for f in ["en_train", "en_dev", "en_test"] {
        fs::write(dir.join(format!("EN-English/{f}.conll")), &text).unwrap();
    }
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

#[test]
fn a_cache_holds_each_splits_features_at_the_last_layers() {
    let data = scratch("data");
    corpus(&data);
    let out = scratch("out");
    let (d, o, b) = (data.to_str().unwrap(), out.to_str().unwrap(), tiny());
    let args = [
        "cache",
        "--corpus",
        "multiconer2-en",
        "--dir",
        d,
        "--bundle",
        b.to_str().unwrap(),
        "--out",
        o,
        "--unrecorded-corpus",
        "--layers",
        "2",
        "--pooling",
        "mean",
        "--commit",
        "test",
    ];
    let r = run(&args);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let record: CacheRecord = serde_json::from_slice(&fs::read(out.join("cache.json")).unwrap()).unwrap();
    assert_eq!((record.schema.as_str(), record.commit.as_str()), (SCHEMA, "test"));
    assert_eq!((record.layers.clone(), record.pooling), (vec![1, 2], WordPooling::Mean));
    assert_eq!(record.encoder.layers, Some(2));
    assert_eq!(record.splits.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["train", "dev", "test"]);

    let corpus = turbo_trainer_recipes::read_corpus("multiconer2-en", &data, true).unwrap();
    let mut enc = WordEncoder::open(&b, &EncoderOptions::default()).unwrap();
    for (name, sentences) in
        [("train", &corpus.splits.train), ("dev", &corpus.splits.dev), ("test", &corpus.splits.test)]
    {
        let s = record.splits.iter().find(|s| s.name == name).unwrap();
        assert_eq!((s.sentences, s.words), (20, 100));
        assert_eq!(s.files, [layer_file(name, 1), layer_file(name, 2)]);
        for f in &s.files {
            assert_eq!(fs::metadata(out.join(f)).unwrap().len(), s.bytes_per_file);
            assert_eq!(s.bytes_per_file, 100 * 32 * 4);
        }
        // The last layer is what a head on the full encoder reads; the
        // one below is the encoder's other layer.
        let words: Vec<Vec<&str>> = sentences.iter().map(|s| s.words.iter().map(String::as_str).collect()).collect();
        let want = enc.layer_features(&words, WordPooling::Mean, 2).unwrap();
        assert_eq!(read_layer(&out, &record, name, 2, sentences).unwrap(), want[1]);
        assert_eq!(
            read_layer(&out, &record, name, 2, sentences).unwrap(),
            enc.features(&words, WordPooling::Mean).unwrap()
        );
        assert_eq!(read_layer(&out, &record, name, 1, sentences).unwrap(), want[0]);
    }
    // Reads that do not match the cache are refused.
    let train = &corpus.splits.train;
    assert!(read_layer(&out, &record, "train", 3, train).unwrap_err().0.contains("no layer 3"));
    assert!(read_layer(&out, &record, "eval", 2, train).unwrap_err().0.contains("no split"));
    assert!(read_layer(&out, &record, "train", 2, &train[1..]).unwrap_err().0.contains("19 sentences"));

    // A taken --out, and layers the model does not have, are refused.
    let again = run(&args);
    assert_eq!(again.status.code(), Some(2));
    let deep = scratch("deep");
    let mut a = args.to_vec();
    a[8] = deep.to_str().unwrap();
    a[11] = "3";
    let r = run(&a);
    assert_eq!(r.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&r.stderr).contains("--layers 3: 1 to the model's 2"));
    assert!(!deep.exists());
    // --splits reads only the splits named.
    let some = scratch("some");
    let mut a = args.to_vec();
    a[8] = some.to_str().unwrap();
    a.extend(["--splits", "train,dev"]);
    assert!(run(&a).status.success());
    let r: CacheRecord = serde_json::from_slice(&fs::read(some.join("cache.json")).unwrap()).unwrap();
    assert_eq!(r.splits.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["train", "dev"]);
    assert!(!some.join(layer_file("test", 2)).exists());
    a[8] = "unused";
    let n = a.len();
    a[n - 1] = "train,eval";
    assert!(String::from_utf8_lossy(&run(&a).stderr).contains("--splits"));
    fs::remove_dir_all(some).unwrap();
    for p in [data, out] {
        fs::remove_dir_all(p).unwrap();
    }
}

#[test]
fn a_head_on_a_cached_layer_trains_as_on_the_encoder_and_a_mix_trains_too() {
    let data = scratch("fit-data");
    corpus(&data);
    let cache = scratch("fit-cache");
    let (d, c, b) = (data.to_str().unwrap(), cache.to_str().unwrap(), tiny());
    let bundle = b.to_str().unwrap();
    let common = ["--corpus", "multiconer2-en", "--dir", d, "--unrecorded-corpus"];
    let settings = ["--crf", "--epochs", "3", "--batch", "8", "--lr", "0.01", "--commit", "test"];
    let ok = |args: Vec<&str>| {
        let o = run(&args);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    };
    ok([&["cache"][..], &common, &["--bundle", bundle, "--out", c, "--layers", "2", "--splits", "train,dev"]].concat());

    // The last cached layer is what `train` reads from the encoder: the
    // same seed gives the same dev curve.
    let enc_out = scratch("fit-enc");
    ok([&["train"][..], &common, &["--bundle", bundle, "--out", enc_out.to_str().unwrap()], &settings].concat());
    let receipt: turbo_trainer_recipes::receipt::Receipt =
        serde_json::from_slice(&fs::read(enc_out.join("receipt.json")).unwrap()).unwrap();
    let fit = |name: &str, how: &[&str]| -> FitRecord {
        let out = scratch(name);
        ok([&["fit-cache"][..], &common, &["--cache", c, "--out", out.to_str().unwrap()], how, &settings].concat());
        let r = serde_json::from_slice(&fs::read(out.join("fit.json")).unwrap()).unwrap();
        assert!(out.join("head.safetensors").is_file() && out.join("head.json").is_file());
        fs::remove_dir_all(out).unwrap();
        r
    };
    let one = fit("fit-one", &["--layer", "2"]);
    assert_eq!((one.schema.as_str(), one.layers.clone(), one.mix.clone()), (SCHEMA_FIT, vec![2], None));
    assert_eq!(one.dev, receipt.training.as_ref().unwrap().dev);
    assert_eq!(Some(one.head.clone()), receipt.head);
    let below = fit("fit-below", &["--layer", "1"]);
    assert_eq!(below.layers, [1]);

    let mix = fit("fit-mix", &["--mix"]);
    assert_eq!(mix.layers, [1, 2]);
    let m = mix.mix.as_ref().unwrap();
    assert_eq!(m.weights.len(), 2);
    assert!((m.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6 && m.gain.is_finite(), "{m:?}");
    assert_eq!(mix.dev.len(), 3);
    let top = fit("fit-top", &["--mix", "--from-layer", "2"]);
    assert_eq!((top.layers.clone(), top.mix.as_ref().map(|m| m.weights.len())), (vec![2], Some(1)));

    // Refusals: no layer named, both, a layer not cached, another corpus.
    let refused = |name: &str, how: &[&str], data: &str, msg: &str| {
        let out = scratch(name);
        let o = out.to_str().unwrap();
        let args = [
            &["fit-cache", "--corpus", "multiconer2-en", "--dir", data, "--unrecorded-corpus"][..],
            &["--cache", c, "--out", o],
            how,
        ]
        .concat();
        let r = run(&args);
        assert_eq!(r.status.code(), Some(2), "{how:?}");
        assert!(String::from_utf8_lossy(&r.stderr).contains(msg), "{}", String::from_utf8_lossy(&r.stderr));
        assert!(!out.exists());
    };
    refused("r1", &[], d, "one of --layer N or --mix");
    refused("r2", &["--layer", "2", "--mix"], d, "one of --layer N or --mix");
    refused("r3", &["--layer", "3"], d, "the cache holds [1, 2]");
    let other = scratch("fit-other");
    fs::create_dir_all(other.join("EN-English")).unwrap();
    for f in ["en_train", "en_dev", "en_test"] {
        fs::write(other.join(format!("EN-English/{f}.conll")), "# id 0\tdomain=en\nx _ _ O\n\n").unwrap();
    }
    refused("r4", &["--layer", "2"], other.to_str().unwrap(), "another corpus");
    for p in [data, cache, enc_out, other] {
        fs::remove_dir_all(p).unwrap();
    }
}
