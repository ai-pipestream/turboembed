//! `turbo-train-head cache` on the small BERT in testdata (two layers)
//! and a small corpus in MultiCoNER's format: every split's word features
//! at the last layers, read back as the encoder gives them, and refusals.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo::manifest::WordPooling;
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::cache::{CacheRecord, SCHEMA, layer_file, read_layer};

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
