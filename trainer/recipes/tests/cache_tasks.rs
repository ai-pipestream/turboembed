//! Integration tests (`#1822`) for `cache`, `fit-cache` / `train-cached`,
//! `train-{lemma,parse,split} --cache`, and `sweep` over cached multi-layer
//! encoder taps (`--layer k` and `--mix`), verifying that heads trained on
//! a lower layer or a learned scalar mix load into `core` and score
//! identically on the encoder's multi-layer output.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use turbo_scratch::Scratch;

use serde_json::Value;
use turbo::head::{BiaffineParser, EditTreeLemmatizer, TokenSplitter, TokenTagger};
use turbo::manifest::{HeadKind, HeadSpec, LayerMix, WordPooling};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::cache::{CacheRecord, stack_layers};
use turbo_trainer_recipes::corpus::Task;
use turbo_trainer_recipes::eval::{Seen, evaluate};
use turbo_trainer_recipes::fit_cache::FitRecord;
use turbo_trainer_recipes::receipt::Receipt;
use turbo_trainer_recipes::sweep::{SCHEMA as SWEEP_SCHEMA, SweepRecord};
use turbo_trainer_recipes::ud::{EN_EWT, read_treebank};
use turbo_trainer_recipes::ud_heads::{UdHead, predict_lemmas, predict_parses, uas};
use turbo_trainer_recipes::ud_split as split;

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn scratch(name: &str) -> Scratch {
    Scratch::at(std::env::temp_dir().join(format!("turbo-cache-tasks-{}-{name}", std::process::id())))
}

/// 24 sentences with multiword contractions (`al` -> `a` + `el`, `del` -> `de` + `el`),
/// inflected verbs (`walked`/`walks`, `painted`/`paints`) and plural nouns (`dogs`, `cars`)
/// so one CoNLL-U fixture exercises `upos`, `lemma`, `parse`, and `split`.
fn treebank() -> String {
    let verbs = [("walked", "walk", "Tense=Past"), ("walks", "walk", "Tense=Pres"), ("painted", "paint", "Tense=Past")];
    let nouns = [("dogs", "dog"), ("cars", "car"), ("walls", "wall")];
    let mut text = String::new();
    for i in 0..24 {
        let (vf, vl, feats) = verbs[i % verbs.len()];
        let (nf, nl) = nouns[i % nouns.len()];
        let (mwt, p1, p2) = if i % 2 == 0 { ("al", "a", "el") } else { ("del", "de", "el") };
        text += &format!(
            "# sent_id = {i}\n\
             1\t{vf}\t{vl}\tVERB\tVBD\t{feats}\t0\troot\t_\t_\n\
             2-3\t{mwt}\t_\t_\t_\t_\t_\t_\t_\t_\n\
             2\t{p1}\t{p1}\tADP\tIN\t_\t4\tcase\t_\t_\n\
             3\t{p2}\t{p2}\tDET\tDT\t_\t4\tdet\t_\t_\n\
             4\t{nf}\t{nl}\tNOUN\tNNS\tNumber=Plur\t1\tobl\t_\t_\n\
             5\t.\t.\tPUNCT\t.\t_\t1\tpunct\t_\t_\n\n"
        );
    }
    text
}

fn write_treebank(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    for pin in [EN_EWT.train, EN_EWT.dev, EN_EWT.test] {
        fs::write(dir.join(pin.path), treebank()).unwrap();
    }
}

fn run_line(cmdline: &str) -> std::process::Output {
    let args: Vec<&str> = cmdline.split_whitespace().collect();
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(&args).output().unwrap()
}

fn run_ok(cmdline: &str) {
    let o = run_line(cmdline);
    assert!(o.status.success(), "cmd={cmdline}\nstderr={}", String::from_utf8_lossy(&o.stderr));
}

fn read_tensors(dir: &Path) -> Vec<(String, Vec<f32>)> {
    let bytes = fs::read(dir.join("head.safetensors")).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    header
        .into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .map(|(k, v)| {
            let o = v["data_offsets"].as_array().unwrap();
            let (a, b) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
            (k, data[a..b].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
        })
        .collect()
}

#[test]
fn cache_fit_cache_and_core_round_trip_lemma_parse_and_split_across_layers_and_mix() {
    let data = scratch("ud-data");
    write_treebank(&data);
    let syn_cache = scratch("syn-cache");
    let split_cache = scratch("split-cache");
    let bundle = tiny();
    let (d, sc, spc, b) = (data.display(), syn_cache.display(), split_cache.display(), bundle.display());

    // 1. Cache syntactic words (`ud-en-ewt`) and surface tokens (`--task split`) across 2 layers.
    run_ok(&format!(
        "cache --corpus ud-en-ewt --dir {d} --bundle {b} --out {sc} --unrecorded-corpus --layers 2 --pooling first --commit test"
    ));
    run_ok(&format!(
        "cache --corpus ud-en-ewt --task split --dir {d} --bundle {b} --out {spc} --unrecorded-corpus --layers 2 --pooling first --commit test"
    ));

    let syn_rec: CacheRecord = serde_json::from_slice(&fs::read(syn_cache.join("cache.json")).unwrap()).unwrap();
    let split_rec: CacheRecord = serde_json::from_slice(&fs::read(split_cache.join("cache.json")).unwrap()).unwrap();
    assert_eq!((syn_rec.layers.as_slice(), split_rec.layers.as_slice()), (&[1, 2][..], &[1, 2][..]));
    assert_eq!((syn_rec.splits[0].words, split_rec.splits[0].words), (24 * 5, 24 * 4));

    let mut enc = WordEncoder::open(&bundle, &EncoderOptions::default()).unwrap();
    let (ud_splits, _, _) = read_treebank(&EN_EWT, &data, true).unwrap();
    let syn_test_words: Vec<Vec<&str>> =
        ud_splits.test.iter().map(|s| s.words.iter().map(String::as_str).collect()).collect();
    let syn_test_taps = stack_layers(&enc.layer_features(&syn_test_words, WordPooling::First, 2).unwrap());

    let (tok_splits, _, _) = split::read_tokens(&EN_EWT, &data, true).unwrap();
    let tok_test_words: Vec<Vec<&str>> =
        tok_splits.test.iter().map(|s| s.tokens.iter().map(String::as_str).collect()).collect();
    let tok_test_taps = stack_layers(&enc.layer_features(&tok_test_words, WordPooling::First, 2).unwrap());

    // 2. Lemma: compare `--layer 2` with uncached `train-lemma`, then test `--layer 1` and `--mix` in `core`.
    let (lemma_enc, lemma_l2, lemma_l1, lemma_mix) =
        (scratch("lemma-enc"), scratch("lemma-l2"), scratch("lemma-l1"), scratch("lemma-mix"));
    let l_opts = "--min-tree-freq 1 --epochs 3 --batch 8 --lr 0.01 --commit test --unrecorded-corpus";
    run_ok(&format!("train-lemma --corpus ud-en-ewt --dir {d} --bundle {b} --out {} {l_opts}", lemma_enc.display()));
    run_ok(&format!(
        "train-lemma --corpus ud-en-ewt --dir {d} --cache {sc} --layer 2 --out {} {l_opts}",
        lemma_l2.display()
    ));
    let r_enc: Receipt = serde_json::from_slice(&fs::read(lemma_enc.join("receipt.json")).unwrap()).unwrap();
    let r_l2: Receipt = serde_json::from_slice(&fs::read(lemma_l2.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(r_l2.training.as_ref().unwrap().dev, r_enc.training.as_ref().unwrap().dev);
    assert_eq!(r_l2.eval.model, r_enc.eval.model);
    assert_eq!(r_l2.head.as_ref().unwrap().layer_mix, None);

    run_ok(&format!(
        "train-lemma --corpus ud-en-ewt --dir {d} --cache {sc} --layer 1 --bilstm 8x1 --out {} {l_opts}",
        lemma_l1.display()
    ));
    run_ok(&format!(
        "train-cached --corpus ud-en-ewt --task lemma --dir {d} --cache {sc} --mix --bilstm 8x1 --out {} {l_opts}",
        lemma_mix.display()
    ));
    for (dir, want_mix) in [
        (&lemma_l1, Some(LayerMix { layers: 2, layer: Some(1) })),
        (&lemma_mix, Some(LayerMix { layers: 2, layer: None })),
    ] {
        let head: HeadSpec = serde_json::from_slice(&fs::read(dir.join("head.json")).unwrap()).unwrap();
        let receipt: Receipt = serde_json::from_slice(&fs::read(dir.join("receipt.json")).unwrap()).unwrap();
        assert_eq!((head.kind, head.layer_mix), (HeadKind::EditTreeLemmatizer, want_mix));
        let tensors = read_tensors(dir);
        let get =
            |s: &str| tensors.iter().find(|(k, _)| *k == format!("{}.{s}", head.tensor_prefix)).map(|(_, v)| v.clone());
        let core = EditTreeLemmatizer::from_tensors(&head, 32, &get).unwrap();
        let pred = predict_lemmas(&core, &ud_splits.test, &syn_test_taps).unwrap();
        let seen = Seen::of(&UdHead::Lemma.sentences(&ud_splits.train));
        let got = evaluate(Task::Tags, &UdHead::Lemma.sentences(&ud_splits.test), &pred, &seen, None);
        assert_eq!(Some(got), receipt.eval.model, "core EditTreeLemmatizer on 2-tap encoder output matches receipt");
    }

    // 3. Parse: test `--layer 1` and `--mix` in `core` (`BiaffineParser` on 2-tap encoder output).
    let (parse_l1, parse_mix) = (scratch("parse-l1"), scratch("parse-mix"));
    let p_opts = "--arc 8 --rel 4 --bilstm 8x1 --epochs 3 --batch 8 --lr 0.01 --commit test --unrecorded-corpus";
    run_ok(&format!(
        "train-parse --corpus ud-en-ewt --dir {d} --cache {sc} --layer 1 --out {} {p_opts}",
        parse_l1.display()
    ));
    run_ok(&format!(
        "fit-cache --corpus ud-en-ewt --task parse --dir {d} --cache {sc} --mix --out {} {p_opts}",
        parse_mix.display()
    ));
    for (dir, want_mix) in [
        (&parse_l1, Some(LayerMix { layers: 2, layer: Some(1) })),
        (&parse_mix, Some(LayerMix { layers: 2, layer: None })),
    ] {
        let head: HeadSpec = serde_json::from_slice(&fs::read(dir.join("head.json")).unwrap()).unwrap();
        let receipt: Receipt = serde_json::from_slice(&fs::read(dir.join("receipt.json")).unwrap()).unwrap();
        assert_eq!((head.kind, head.layer_mix), (HeadKind::BiaffineParser, want_mix));
        let tensors = read_tensors(dir);
        let get =
            |s: &str| tensors.iter().find(|(k, _)| *k == format!("{}.{s}", head.tensor_prefix)).map(|(_, v)| v.clone());
        let core = BiaffineParser::from_tensors(&head, 32, &get).unwrap();
        let pred = predict_parses(&core, &ud_splits.test, &syn_test_taps).unwrap();
        let gold = UdHead::Deps.sentences(&ud_splits.test);
        let seen = Seen::of(&UdHead::Deps.sentences(&ud_splits.train));
        let got = evaluate(Task::Tags, &gold, &pred, &seen, None);
        assert_eq!(Some(got), receipt.eval.model);
        assert_eq!(uas(&gold, &pred), receipt.eval.uas.unwrap().model);
    }

    // 4. Split: test `--layer 1` and `--mix` in `core` (`TokenSplitter` on 2-tap encoder output).
    let (split_l1, split_mix) = (scratch("split-l1"), scratch("split-mix"));
    let s_opts = "--min-tree-freq 1 --bilstm 8x1 --epochs 3 --batch 8 --lr 0.01 --commit test --unrecorded-corpus";
    run_ok(&format!(
        "train-split --corpus ud-en-ewt --dir {d} --cache {spc} --layer 1 --out {} {s_opts}",
        split_l1.display()
    ));
    run_ok(&format!(
        "train-split --corpus ud-en-ewt --dir {d} --cache {spc} --mix --out {} {s_opts}",
        split_mix.display()
    ));
    for (dir, want_mix) in [
        (&split_l1, Some(LayerMix { layers: 2, layer: Some(1) })),
        (&split_mix, Some(LayerMix { layers: 2, layer: None })),
    ] {
        let head: HeadSpec = serde_json::from_slice(&fs::read(dir.join("head.json")).unwrap()).unwrap();
        let receipt: Receipt = serde_json::from_slice(&fs::read(dir.join("receipt.json")).unwrap()).unwrap();
        assert_eq!((head.kind, head.layer_mix), (HeadKind::TokenSplit, want_mix));
        let tensors = read_tensors(dir);
        let get =
            |s: &str| tensors.iter().find(|(k, _)| *k == format!("{}.{s}", head.tensor_prefix)).map(|(_, v)| v.clone());
        let core = TokenSplitter::from_tensors(&head, 32, &get).unwrap();
        assert_eq!(core.taps(), 2);
        let pred = split::predict(&core, &tok_splits.test, &tok_test_taps).unwrap();
        let gold = split::sentences(&tok_splits.test);
        let seen = Seen::of(&split::sentences(&tok_splits.train));
        let got = evaluate(Task::Tags, &gold, &pred, &seen, None);
        assert_eq!(Some(got), receipt.eval.model);
    }

    // 5. Tag (`ud-en-ewt-upos`) on the shared syntactic-word cache (`ud-en-ewt`) with `--layer 1`.
    let tag_l1 = scratch("tag-l1");
    run_ok(&format!(
        "train --corpus ud-en-ewt-upos --dir {d} --cache {sc} --layer 1 --out {} --unrecorded-corpus --epochs 3 --batch 8 --commit test",
        tag_l1.display()
    ));
    let tag_head: HeadSpec = serde_json::from_slice(&fs::read(tag_l1.join("head.json")).unwrap()).unwrap();
    let tag_receipt: Receipt = serde_json::from_slice(&fs::read(tag_l1.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(tag_head.layer_mix, Some(LayerMix { layers: 2, layer: Some(1) }));
    let tag_tensors = read_tensors(&tag_l1);
    let tag_get = |s: &str| {
        tag_tensors.iter().find(|(k, _)| *k == format!("{}.{s}", tag_head.tensor_prefix)).map(|(_, v)| v.clone())
    };
    let tag_core = TokenTagger::from_tensors(&tag_head, 32, &tag_get).unwrap();
    let upos_corpus = turbo_trainer_recipes::read_corpus("ud-en-ewt-upos", &data, true).unwrap();
    let mut tag_pred = Vec::new();
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), turbo::head::tagger::Work::default());
    for (s, f) in upos_corpus.splits.test.iter().zip(&syn_test_taps) {
        let ids: Vec<u32> = (0..s.words.len() as u32).collect();
        tag_core.scores(f, &ids, &mut scores, &mut work).unwrap();
        tag_core.decode(&scores, s.words.len(), &mut tags).unwrap();
        tag_pred.push(tags.iter().map(|&t| tag_core.labels()[t as usize].clone()).collect());
    }
    let tag_seen = Seen::of(&upos_corpus.splits.train);
    let tag_got = evaluate(Task::Tags, &upos_corpus.splits.test, &tag_pred, &tag_seen, None);
    assert_eq!(Some(tag_got), tag_receipt.eval.model);

    // 6. Mismatched cache kind (syntactic words vs surface tokens) is refused.
    let bad = scratch("bad-split");
    let res = run_line(&format!(
        "train-split --corpus ud-en-ewt --dir {d} --cache {sc} --layer 2 --out {} --unrecorded-corpus",
        bad.display()
    ));
    assert_eq!(res.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&res.stderr).contains("another corpus"));

    for p in [
        data,
        syn_cache,
        split_cache,
        lemma_enc,
        lemma_l2,
        lemma_l1,
        lemma_mix,
        parse_l1,
        parse_mix,
        split_l1,
        split_mix,
        tag_l1,
    ] {
        let _ = fs::remove_dir_all(p);
    }
}

#[test]
fn sweep_writes_per_layer_and_mix_receipts_and_comparison_table() {
    let data = scratch("sweep-data");
    write_treebank(&data);
    let syn_cache = scratch("sweep-syn-cache");
    let bundle = tiny();
    let (d, sc, b) = (data.display(), syn_cache.display(), bundle.display());

    run_ok(&format!(
        "cache --corpus ud-en-ewt --dir {d} --bundle {b} --out {sc} --unrecorded-corpus --layers 2 --pooling first --commit test"
    ));

    let peer_dir = scratch("sweep-peer");
    let test_conllu = data.join(EN_EWT.test.path);
    run_ok(&format!(
        "peer --corpus ud-en-ewt --dir {d} --peer stanza-test --predictions {} --column deps --out {} --unrecorded-corpus --commit test",
        test_conllu.display(),
        peer_dir.display()
    ));
    let peer_receipt = peer_dir.join("receipt.json");

    let sweep_out = scratch("sweep-parse");
    let o = run_line(&format!(
        "sweep --corpus ud-en-ewt --task parse --dir {d} --cache {sc} --out {} --peer-receipt {} --unrecorded-corpus --arc 8 --rel 4 --epochs 2 --batch 8 --commit test",
        sweep_out.display(),
        peer_receipt.display()
    ));
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let stdout = String::from_utf8_lossy(&o.stdout);
    assert!(
        stdout.contains("| `layer-1` |") && stdout.contains("| `layer-2` |") && stdout.contains("| `mix` |"),
        "{stdout}"
    );
    assert!(stdout.contains("stanza-test"), "{stdout}");

    let record: SweepRecord = serde_json::from_slice(&fs::read(sweep_out.join("sweep.json")).unwrap()).unwrap();
    assert_eq!(record.schema, SWEEP_SCHEMA);
    assert_eq!(record.reference_layer, 2);
    assert_eq!(record.runs.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), ["layer-1", "layer-2", "mix"]);
    assert_eq!(record.runs[1].vs_last_layer.delta, 0.0);
    assert!(record.runs[2].mix.is_some());
    assert!(sweep_out.join("sweep.md").is_file());

    for sub in ["layer-01", "layer-02", "mix"] {
        let dir = sweep_out.join(sub);
        assert!(dir.join("head.safetensors").is_file() && dir.join("head.json").is_file());
        let fit: FitRecord = serde_json::from_slice(&fs::read(dir.join("fit.json")).unwrap()).unwrap();
        let rec: Receipt = serde_json::from_slice(&fs::read(dir.join("receipt.json")).unwrap()).unwrap();
        assert_eq!(fit.head, rec.head.unwrap());
        assert!(rec.eval.uas.is_some());
    }

    for p in [data, syn_cache, peer_dir, sweep_out] {
        let _ = fs::remove_dir_all(p);
    }
}

#[path = "cache_tasks/regressions.rs"]
mod regressions;
