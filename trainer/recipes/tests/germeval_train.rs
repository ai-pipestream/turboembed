//! Actual CPU CRF fitting on authored German-format fixtures, not a quality gate.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use turbo::head::TokenTagger;
use turbo::manifest::{Decode, HeadSpec};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::{eval, germeval, read_corpus, receipt::Receipt, run};

fn reload(out: &Path) -> TokenTagger {
    let spec: HeadSpec = serde_json::from_slice(&fs::read(out.join("head.json")).unwrap()).unwrap();
    assert_eq!(spec.labels.len(), 25);
    assert_eq!(spec.decode, Decode::Crf);
    let bytes = fs::read(out.join("head.safetensors")).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    TokenTagger::from_tensors(&spec, 32, &|name| {
        let offsets = header.get(format!("{}.{name}", spec.tensor_prefix))?.get("data_offsets")?.as_array()?;
        let a = offsets[0].as_u64()? as usize;
        let b = offsets[1].as_u64()? as usize;
        Some(data[a..b].as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect())
    })
    .unwrap()
}

#[test]
fn both_layers_fit_with_rare_dev_labels_and_reload_identical_scores() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("germeval-fit-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    // Neither training column has an ORGderiv label. Both dev columns do.
    fs::write(root.join("NER-de-train.tsv"), "1\tann\tB-PER\tB-LOC\n2\tlee\tI-PER\tI-LOC\n3\t.\tO\tO\n\n".repeat(4))
        .unwrap();
    fs::write(
        root.join("NER-de-dev.tsv"),
        "1\tfoo\tB-ORGderiv\tB-ORGderiv\n2\tbar\tI-ORGderiv\tI-ORGderiv\n3\t.\tO\tO\n\n",
    )
    .unwrap();
    fs::write(root.join("NER-de-test.tsv"), "1\tbob\tB-PER\tB-LOC\n2\tlee\tI-PER\tI-LOC\n3\t.\tO\tO\n\n").unwrap();
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    for id in [germeval::OUTER, germeval::INNER] {
        let out = root.join(id);
        let output = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
            .args(["train", "--corpus", id, "--dir"])
            .arg(&root)
            .arg("--bundle")
            .arg(&bundle)
            .arg("--out")
            .arg(&out)
            .args([
                "--unrecorded-corpus",
                "--crf",
                "--threads",
                "2",
                "--epochs",
                "2",
                "--batch",
                "2",
                "--commit",
                "fixture",
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let receipt: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
        assert!(!receipt.corpus.recorded);
        assert_eq!(receipt.corpus.train.sentences, 4);
        assert_eq!(receipt.config.as_ref().unwrap().dev_left_out, 0);
        assert_eq!(receipt.timing_s.feature_words, 15, "all training and dev words are encoded");
        let tagger = reload(&out);
        let corpus = read_corpus(id, &root, true).unwrap();
        let mut encoder = WordEncoder::open(&bundle, &EncoderOptions::default()).unwrap();
        let predictions = run::predict(&mut encoder, &tagger, &corpus.splits.test).unwrap();
        let score = eval::evaluate(
            corpus.record.task,
            &corpus.splits.test,
            &predictions,
            &eval::Seen::of(&corpus.splits.train),
            None,
        );
        assert_eq!(receipt.eval.model.as_ref().unwrap(), &score);
    }
    fs::remove_dir_all(root).unwrap();
}
