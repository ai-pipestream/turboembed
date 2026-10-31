//! LitBank coreference reader, CoNLL (MUC, B-cubed, CEAF-e) scoring,
//! seeded paired bootstrap, and `train-coref` end to end on
//! `testdata/tiny-bert-bundle` (#1527).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use turbo::head::CorefRanker;
use turbo::manifest::{CorefShape, Decode, HeadKind, LicenseClass, SpanShape};
use turbo_trainer_recipes::corpus::Task;
use turbo_trainer_recipes::eval::Metrics;
use turbo_trainer_recipes::litbank::{self, CorefDoc, ID, LICENCE, SPDX, SPLIT, WORKS, parse, pins, read, read_coref};
use turbo_trainer_recipes::litbank_eval::{CorefOptions, baseline, bootstrap_coref, evaluate_coref, spec};
use turbo_trainer_recipes::receipt::{Receipt, SCHEMA};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn moonstone_fixture() -> &'static [u8] {
    include_bytes!("../../../analysis/train-recipes/tests/data/litbank/155_the_moonstone_brat.conll")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-litbank-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn row(doc: &str, tok: usize, word: &str, coref: &str) -> String {
    format!("{doc}\t0\t{tok}\t{word}\tNNP\t_\t_\t_\t_\t_\t_\t_\t{coref}\n")
}

/// Synthetic document with singletons, a two-sentence coreference chain,
/// and nested mentions (both cross-cluster and same-cluster).
fn sample_conll(doc: &str) -> String {
    let mut s = format!("#begin document ({doc}); part 0\n");
    // Sentence 1: "[Alice_0] met [her_0 friend_1] in [Paris_2] ."
    s += &row(doc, 0, "Alice", "(0)");
    s += &row(doc, 1, "met", "_");
    s += &row(doc, 2, "her", "(1|(0)");
    s += &row(doc, 3, "friend", "1)");
    s += &row(doc, 4, "in", "_");
    s += &row(doc, 5, "Paris", "(2)");
    s += &row(doc, 6, ".", "_");
    s += "\n";
    // Sentence 2: "[Alice_0] loved [Paris_2] and [its_2 [old_2 streets_2]_2] ."
    s += &row(doc, 0, "Alice", "(0)");
    s += &row(doc, 1, "loved", "-");
    s += &row(doc, 2, "Paris", "(2)");
    s += &row(doc, 3, "and", "_");
    s += &row(doc, 4, "its", "(2|(2)");
    s += &row(doc, 5, "old", "(2");
    s += &row(doc, 6, "streets", "2)|2)");
    s += &row(doc, 7, ".", "_");
    s += "\n#end document\n";
    s
}

/// Synthetic training/eval document where identical word forms corefer
/// (`Paris`..`Paris` in cluster 0, `London`..`London` in cluster 1), so a
/// small ranker over `tiny-bert-bundle` learns to link identical mentions
/// and beats the singleton baseline.
fn trainable_conll(doc: &str) -> String {
    let mut s = format!("#begin document ({doc}); part 0\n");
    s += &row(doc, 0, "Paris", "(0)");
    s += &row(doc, 1, "saw", "_");
    s += &row(doc, 2, "London", "(1)");
    s += &row(doc, 3, ".", "_");
    s += "\n";
    s += &row(doc, 0, "Paris", "(0)");
    s += &row(doc, 1, "met", "_");
    s += &row(doc, 2, "London", "(1)");
    s += &row(doc, 3, ".", "_");
    s += "\n";
    s += &row(doc, 0, "Paris", "(0)");
    s += &row(doc, 1, "left", "_");
    s += &row(doc, 2, "London", "(1)");
    s += &row(doc, 3, ".", "_");
    s += "\n#end document\n";
    s
}

fn write_fixture_corpus(dir: &Path, count: usize) {
    let conll_dir = dir.join("coref/conll");
    fs::create_dir_all(&conll_dir).unwrap();
    for i in 0..count {
        let name = format!("doc_{i:02}_brat");
        fs::write(conll_dir.join(format!("{name}.conll")), trainable_conll(&name)).unwrap();
    }
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

fn read_safetensors(bytes: &[u8]) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    header
        .into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .map(|(k, v)| {
            let shape = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
            let o = v["data_offsets"].as_array().unwrap();
            let (a, b) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
            (k, shape, data[a..b].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
        })
        .collect()
}

#[test]
fn parses_singletons_chains_and_nested_mentions() {
    let doc = parse("coref/conll/sample.conll", sample_conll("sample_doc").as_bytes()).unwrap();
    assert_eq!(doc.id, "sample_doc");
    assert_eq!(doc.sentences.len(), 2);
    assert_eq!(doc.words.len(), 15);
    assert_eq!(
        doc.mentions,
        vec![
            (0, 0),   // Alice (cluster 0)
            (2, 2),   // her (cluster 0)
            (2, 3),   // her friend (cluster 1)
            (5, 5),   // Paris (cluster 2)
            (7, 7),   // Alice (cluster 0)
            (9, 9),   // Paris (cluster 2)
            (11, 11), // its (cluster 2)
            (11, 13), // its old streets (cluster 2)
            (12, 13), // old streets (cluster 2)
        ]
    );
    assert_eq!(doc.chains, vec![0, 0, 1, 2, 0, 2, 2, 2, 2]);
    assert_eq!(
        doc.clusters,
        vec![vec![(0, 0), (2, 2), (7, 7)], vec![(2, 3)], vec![(5, 5), (9, 9), (11, 11), (11, 13), (12, 13)],]
    );
}

#[test]
fn parses_pinned_moonstone_fixture_and_matches_pin() {
    let all_pins = pins();
    assert_eq!(all_pins.len(), WORKS);
    let (tr, dv, te) = SPLIT.partition();
    assert_eq!((tr.len(), dv.len(), te.len()), (80, 10, 10));

    let bytes = moonstone_fixture();
    let pin = all_pins.iter().find(|p| p.path == "coref/conll/155_the_moonstone_brat.conll").unwrap();
    assert_eq!(pin.size, bytes.len() as u64);
    assert_eq!(pin.sha256, turbo::bundle::sha256_hex(bytes));

    let doc = parse(pin.path, bytes).unwrap();
    assert_eq!(doc.id, "155_the_moonstone_brat");
    assert_eq!(doc.sentences.len(), 73);
    assert_eq!(doc.words.len(), 2030);
    assert_eq!(doc.mentions.len(), 273);
    assert_eq!(doc.chains.len(), 273);
    assert_eq!(doc.clusters.len(), 73);
}

#[test]
fn refuses_malformed_conll_files() {
    assert!(parse("bad.conll", &[0xff, 0xfe]).unwrap_err().0.contains("UTF-8"));
    assert!(parse("outside.conll", row("d", 0, "Alice", "_").as_bytes()).unwrap_err().0.contains("outside"));
    assert!(parse("unclosed_doc.conll", b"#begin document (d)\n").unwrap_err().0.contains("unclosed #begin document"));
    assert!(
        parse("unmatched_end.conll", b"#end document\n").unwrap_err().0.contains("without matching #begin document")
    );
    assert!(
        parse("nested_begin.conll", b"#begin document (a)\n#begin document (b)\n#end document\n")
            .unwrap_err()
            .0
            .contains("nested #begin document")
    );
    assert!(parse("empty.conll", b"#begin document (d)\n#end document\n").unwrap_err().0.contains("empty document"));
    assert!(
        parse("short_cols.conll", b"#begin document (d)\nd\t0\t0\tAlice\n#end document\n")
            .unwrap_err()
            .0
            .contains("12 tab-separated columns")
    );
    assert!(
        parse("bad_idx.conll", format!("#begin document (d)\n{}#end document\n", row("d", 2, "Alice", "_")).as_bytes())
            .unwrap_err()
            .0
            .contains("sentence token index")
    );
    assert!(
        parse("empty_word.conll", format!("#begin document (d)\n{}#end document\n", row("d", 0, "", "_")).as_bytes())
            .unwrap_err()
            .0
            .contains("empty word")
    );
    assert!(
        parse(
            "unclosed_bracket.conll",
            format!("#begin document (d)\n{}#end document\n", row("d", 0, "Alice", "(0")).as_bytes()
        )
        .unwrap_err()
        .0
        .contains("unclosed coreference bracket")
    );
    assert!(
        parse(
            "cross_sentence.conll",
            format!(
                "#begin document (d)\n{}\n{}#end document\n",
                row("d", 0, "Alice", "(0"),
                row("d", 0, "left", "0)")
            )
            .as_bytes()
        )
        .unwrap_err()
        .0
        .contains("unclosed coreference bracket")
    );
    assert!(
        parse(
            "unmatched_close.conll",
            format!("#begin document (d)\n{}#end document\n", row("d", 0, "Alice", "0)")).as_bytes()
        )
        .unwrap_err()
        .0
        .contains("unmatched closing bracket")
    );
    assert!(
        parse(
            "bad_bracket.conll",
            format!("#begin document (d)\n{}#end document\n", row("d", 0, "Alice", "(abc)")).as_bytes()
        )
        .unwrap_err()
        .0
        .contains("invalid cluster ID")
    );
    assert!(
        parse(
            "conflict.conll",
            format!("#begin document (d)\n{}#end document\n", row("d", 0, "Alice", "(0)|(1)")).as_bytes()
        )
        .unwrap_err()
        .0
        .contains("conflicting cluster IDs")
    );
}

#[test]
fn enforces_pins_and_permissive_licence_class() {
    let dir = scratch("pins");
    write_fixture_corpus(&dir, 4);

    // Without --unrecorded-corpus, missing/unpinned files are refused.
    let err = read_coref(&dir, false).unwrap_err();
    assert!(err.0.contains("coref/conll/") || err.0.contains("not the recorded release"));

    // With --unrecorded-corpus, the fixture is accepted and marked unrecorded,
    // while retaining LitBank's permissive CC BY 4.0 licence class.
    let splits = read_coref(&dir, true).unwrap();
    assert_eq!(splits.record.id, ID);
    assert_eq!(splits.record.task, Task::Coref);
    assert_eq!(splits.record.licence, LICENCE);
    assert_eq!(splits.record.spdx, SPDX);
    assert!(!splits.record.recorded);
    assert_eq!(splits.record.class().unwrap(), LicenseClass::Permissive);
    assert_eq!((splits.train.len(), splits.dev.len(), splits.test.len()), (2, 1, 1));

    let corpus = read(&dir, true).unwrap();
    assert_eq!(corpus.labels(), vec!["coref".to_owned()]);

    let head_spec = spec(&splits.record, &CorefOptions::default()).unwrap();
    assert_eq!(head_spec.kind, HeadKind::CorefRanker);
    assert_eq!(head_spec.decode, Decode::Antecedent);
    assert_eq!(head_spec.license_class, Some(LicenseClass::Permissive));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn scores_muc_b_cubed_ceaf_e_and_paired_bootstrap() {
    let doc1 = parse("d1.conll", trainable_conll("d1").as_bytes()).unwrap();
    let doc2 = parse("d2.conll", sample_conll("d2").as_bytes()).unwrap();
    let gold = vec![doc1.clone(), doc2.clone()];

    // Perfect prediction scores 1.0 on every metric.
    let exact: Vec<Vec<Vec<(u32, u32)>>> = gold.iter().map(|d| d.clusters.clone()).collect();
    let Metrics::Coref { documents, mentions, gold_clusters, predicted_clusters, muc, b_cubed, ceaf_e, conll_f1 } =
        evaluate_coref(&gold, &exact).unwrap()
    else {
        panic!("expected Metrics::Coref");
    };
    assert_eq!(documents, 2);
    assert_eq!(mentions, doc1.mentions.len() + doc2.mentions.len());
    assert_eq!(gold_clusters, doc1.clusters.len() + doc2.clusters.len());
    assert_eq!(predicted_clusters, gold_clusters);
    assert!((muc.f1 - 1.0).abs() < 1e-12);
    assert!((b_cubed.f1 - 1.0).abs() < 1e-12);
    assert!((ceaf_e.f1 - 1.0).abs() < 1e-12);
    assert!((conll_f1 - 1.0).abs() < 1e-12);

    // Deterministic singleton baseline has 0 MUC links, B-cubed precision 1.0,
    // and strictly lower CoNLL F1 than the exact clustering.
    let base = baseline(&gold);
    let Metrics::Coref { muc: b_muc, b_cubed: b_b3, ceaf_e: b_ceaf, conll_f1: b_f1, .. } =
        evaluate_coref(&gold, &base).unwrap()
    else {
        panic!("expected Metrics::Coref");
    };
    assert_eq!(b_muc.f1, 0.0);
    assert!((b_b3.precision - 1.0).abs() < 1e-12);
    assert!(b_b3.recall > 0.0 && b_b3.recall < 1.0);
    assert!(b_ceaf.f1 > 0.0 && b_ceaf.f1 < 1.0);
    assert!(b_f1 > 0.0 && b_f1 < 0.5);

    let cmp = bootstrap_coref(&gold, &exact, &base, 200, 42).unwrap();
    assert_eq!(cmp, bootstrap_coref(&gold, &exact, &base, 200, 42).unwrap());
    assert!((cmp.model - 1.0).abs() < 1e-12);
    assert!((cmp.baseline - b_f1).abs() < 1e-12);
    assert!(cmp.delta > 0.5);
    assert!(cmp.low > 0.4);
}

#[test]
fn train_coref_writes_a_ranker_the_core_loads_and_is_deterministic() {
    let data = scratch("data-coref");
    write_fixture_corpus(&data, 6);

    // Refuse invalid flags and clean up --out.
    let bad_out = scratch("out-bad");
    let bad = run(&[
        "train-coref",
        "--corpus",
        "litbank",
        "--dir",
        data.to_str().unwrap(),
        "--bundle",
        tiny().to_str().unwrap(),
        "--out",
        bad_out.to_str().unwrap(),
        "--unrecorded-corpus",
        "--crf",
    ]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("--crf"));
    assert!(!bad_out.exists());

    // Check and baseline commands work on litbank.
    let chk = run(&["check", "--corpus", "litbank", "--dir", data.to_str().unwrap(), "--unrecorded-corpus"]);
    assert!(chk.status.success(), "{}", String::from_utf8_lossy(&chk.stderr));

    let base_out = scratch("out-base");
    let base_cmd = run(&[
        "baseline",
        "--corpus",
        "litbank",
        "--dir",
        data.to_str().unwrap(),
        "--out",
        base_out.to_str().unwrap(),
        "--unrecorded-corpus",
        "--commit",
        "test",
    ]);
    assert!(base_cmd.status.success(), "{}", String::from_utf8_lossy(&base_cmd.stderr));
    let base_receipt: Receipt = serde_json::from_slice(&fs::read(base_out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(base_receipt.schema, SCHEMA);
    assert_eq!(base_receipt.corpus.id, "litbank");
    assert!(base_receipt.eval.model.is_none());

    // Train two runs with the same seed and verify determinism + CorefRanker reload.
    let out1 = scratch("out-coref-1");
    let out2 = scratch("out-coref-2");
    for out in [&out1, &out2] {
        let cmd = run(&[
            "train-coref",
            "--corpus",
            "litbank",
            "--dir",
            data.to_str().unwrap(),
            "--bundle",
            tiny().to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--unrecorded-corpus",
            "--mlp",
            "16",
            "--max-antecedents",
            "16",
            "--max-words",
            "8",
            "--width-dim",
            "4",
            "--epochs",
            "8",
            "--batch",
            "2",
            "--lr",
            "0.02",
            "--seed",
            "7",
            "--commit",
            "test",
        ]);
        assert!(cmd.status.success(), "{}", String::from_utf8_lossy(&cmd.stderr));
    }

    let bytes1 = fs::read(out1.join("head.safetensors")).unwrap();
    let bytes2 = fs::read(out2.join("head.safetensors")).unwrap();
    assert_eq!(bytes1, bytes2, "identical seed produces identical safetensors bytes");
    assert_eq!(fs::read(out1.join("head.json")).unwrap(), fs::read(out2.join("head.json")).unwrap());

    let r1: Receipt = serde_json::from_slice(&fs::read(out1.join("receipt.json")).unwrap()).unwrap();
    let r2: Receipt = serde_json::from_slice(&fs::read(out2.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(r1.eval, r2.eval);

    let head = r1.head.clone().unwrap();
    assert_eq!(head.kind, HeadKind::CorefRanker);
    assert_eq!(head.layer, "turboembed:coref");
    assert_eq!(head.decode, Decode::Antecedent);
    assert_eq!(head.license_class, Some(LicenseClass::Permissive));
    assert_eq!(head.span, Some(SpanShape { max_words: 8, width_dim: 4 }));
    assert_eq!(head.coref, Some(CorefShape { dim: 16, antecedents: 16 }));

    let training = r1.training.as_ref().unwrap();
    assert!(
        training.train_loss.last().unwrap() < training.train_loss.first().unwrap(),
        "training loss decreases: {:?}",
        training.train_loss
    );
    let model_score = r1.eval.model.as_ref().unwrap().score();
    let base_score = r1.eval.baseline.score();
    assert!(model_score > base_score, "trained coref ranker ({model_score}) beats singleton baseline ({base_score})");

    // Reload the written safetensors through CorefRanker::from_tensors.
    let tensors = read_safetensors(&bytes1);
    let want = turbo::head::coref::tensors(&head, 32);
    let mut got_shapes: Vec<(String, Vec<u64>)> = tensors.iter().map(|(k, s, _)| (k.clone(), s.clone())).collect();
    let mut want_shapes: Vec<(String, Vec<u64>)> =
        want.into_iter().map(|(s, shape)| (format!("heads.coref.{s}"), shape)).collect();
    got_shapes.sort();
    want_shapes.sort();
    assert_eq!(got_shapes, want_shapes);

    let get = |s: &str| tensors.iter().find(|(k, _, _)| *k == format!("heads.coref.{s}")).map(|(_, _, v)| v.clone());
    let reloaded = CorefRanker::from_tensors(&head, 32, &get).unwrap();
    assert_eq!(reloaded.hidden(), 32);

    for d in [data, base_out, out1, out2] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn recorded_litbank_release_counts_and_baseline() {
    let Some(dir) = std::env::var_os("TURBO_TEST_LITBANK_DIR").map(PathBuf::from) else { return };
    let splits = litbank::read_coref(&dir, false).unwrap();
    assert!(splits.record.recorded);
    assert_eq!((splits.train.len(), splits.dev.len(), splits.test.len()), (80, 10, 10));
    assert_eq!((splits.record.train.sentences, splits.record.train.words), (6_997, 166_604));
    assert_eq!((splits.record.dev.sentences, splits.record.dev.words), (806, 20_467));
    assert_eq!((splits.record.test.sentences, splits.record.test.words), (759, 23_461));

    let count_mentions = |docs: &[CorefDoc]| docs.iter().map(|d| d.mentions.len()).sum::<usize>();
    assert_eq!(
        (count_mentions(&splits.train), count_mentions(&splits.dev), count_mentions(&splits.test)),
        (23_182, 2_752, 3_169)
    );

    let base = baseline(&splits.test);
    let score = evaluate_coref(&splits.test, &base).unwrap().score();
    assert!((score - 0.300915886939698).abs() < 1e-12);
}
