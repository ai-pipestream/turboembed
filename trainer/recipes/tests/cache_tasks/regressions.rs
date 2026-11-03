use super::*;

#[test]
fn split_cache_respects_source_prefix_before_pruned_labels() {
    let data = scratch("prefix-data");
    write_treebank(&data);
    let cache = scratch("prefix-cache");
    let direct = scratch("prefix-direct");
    let cached = scratch("prefix-cached");
    let bundle = tiny();
    let d = data.display();
    run_ok(&format!(
        "cache --corpus ud-en-ewt --task split --dir {d} --bundle {} --out {} --unrecorded-corpus --layers 2 --pooling first --commit test",
        bundle.display(),
        cache.display()
    ));
    // al occurs twice in the prefix, del once: the middle sentence is
    // pruned. A later al sentence must not backfill that source position.
    let opts = "--unrecorded-corpus --max-train 3 --min-tree-freq 2 --epochs 2 --batch 8 --commit test";
    run_ok(&format!(
        "train-split --corpus ud-en-ewt --dir {d} --bundle {} --out {} {opts}",
        bundle.display(),
        direct.display()
    ));
    run_ok(&format!(
        "train-split --corpus ud-en-ewt --dir {d} --cache {} --layer 2 --out {} {opts}",
        cache.display(),
        cached.display()
    ));
    let a: Receipt = serde_json::from_slice(&fs::read(direct.join("receipt.json")).unwrap()).unwrap();
    let b: Receipt = serde_json::from_slice(&fs::read(cached.join("receipt.json")).unwrap()).unwrap();
    let fit: FitRecord = serde_json::from_slice(&fs::read(cached.join("fit.json")).unwrap()).unwrap();
    assert_eq!(b.training.as_ref().unwrap().train_loss, a.training.as_ref().unwrap().train_loss);
    assert_eq!(b.training.as_ref().unwrap().dev, a.training.as_ref().unwrap().dev);
    assert_eq!(b.timing_s.feature_words, a.timing_s.feature_words);
    assert_eq!(b.eval.model, a.eval.model);
    assert_eq!(fit.train_left_out, 1);
    assert_eq!(read_tensors(&cached), read_tensors(&direct));
    for p in [data, cache, direct, cached] {
        fs::remove_dir_all(p).unwrap();
    }
}

#[test]
fn tag_cache_counts_only_excluded_sentences_inside_source_prefix() {
    let data = scratch("tag-prefix-data");
    write_treebank(&data);
    // This label exists only outside the training prefix. It must not be
    // counted as a rejected training example or enter the trained head.
    let text = treebank()
        .split("\n\n")
        .enumerate()
        .map(|(i, s)| if i >= 3 { s.replace("\tVERB\t", "\tADJ\t") } else { s.to_owned() })
        .collect::<Vec<_>>()
        .join("\n\n");
    fs::write(data.join(EN_EWT.train.path), text).unwrap();
    let cache = scratch("tag-prefix-cache");
    let out = scratch("tag-prefix-fit");
    run_ok(&format!(
        "cache --corpus ud-en-ewt --dir {} --bundle {} --out {} --unrecorded-corpus --layers 2 --commit test",
        data.display(),
        tiny().display(),
        cache.display()
    ));
    run_ok(&format!(
        "fit-cache --corpus ud-en-ewt-upos --dir {} --cache {} --mix --out {} --unrecorded-corpus --max-train 3 --epochs 2 --batch 8 --commit test",
        data.display(),
        cache.display(),
        out.display()
    ));
    let fit: FitRecord = serde_json::from_slice(&fs::read(out.join("fit.json")).unwrap()).unwrap();
    assert_eq!(fit.train_left_out, 0, "outside-prefix labels are not rejected training examples");
    assert!(!fit.head.labels.iter().any(|l| l == "ADJ"));
    for p in [data, cache, out] {
        fs::remove_dir_all(p).unwrap();
    }
}

#[test]
fn sweep_refuses_incompatible_peer_receipts() {
    let data = scratch("peer-refusal-data");
    write_treebank(&data);
    let cache = scratch("peer-refusal-cache");
    let peer = scratch("peer-refusal-source");
    run_ok(&format!(
        "cache --corpus ud-en-ewt --dir {} --bundle {} --out {} --unrecorded-corpus --layers 2 --commit test",
        data.display(),
        tiny().display(),
        cache.display()
    ));
    run_ok(&format!(
        "peer --corpus ud-en-ewt --dir {} --peer stanza-test --predictions {} --column deps --out {} --unrecorded-corpus --commit test",
        data.display(),
        data.join(EN_EWT.test.path).display(),
        peer.display()
    ));
    let path = peer.join("receipt.json");
    let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let cases = [
        ("corpus", "/corpus/id", Value::from("unrelated-corpus")),
        ("split", "/eval/split", Value::from("dev")),
        ("task", "/corpus/task", Value::from("bio")),
        ("schema", "/schema", Value::from("unknown.v1")),
        ("release", "/corpus/files/2/sha256", Value::from("0".repeat(64))),
        ("column", "/peer/column", Value::from("lemma")),
        ("identity", "/peer", Value::Null),
        ("name", "/peer/name", Value::from("")),
        (
            "metric",
            "/eval/model",
            serde_json::json!({"spans":{"all":{"gold":0,"found":0,"correct":0,"precision":0.0,"recall":0.0,"f1":0.0},"per_type":{}}}),
        ),
        ("uas", "/eval/uas", Value::Null),
        ("score", "/eval/model/tags/accuracy", Value::from(2.0)),
        ("count", "/eval/model/tags/words", Value::from(1)),
    ];
    let mut accepted = Vec::new();
    for (name, pointer, value) in cases {
        let mut r = original.clone();
        *r.pointer_mut(pointer).unwrap() = value;
        fs::write(&path, serde_json::to_vec(&r).unwrap()).unwrap();
        let out = scratch(&format!("peer-refusal-{name}"));
        let o = run_line(&format!(
            "sweep --corpus ud-en-ewt --task parse --dir {} --cache {} --out {} --peer-receipt {} --unrecorded-corpus --arc 8 --rel 4 --epochs 2 --batch 8 --max-train 3 --commit test",
            data.display(),
            cache.display(),
            out.display(),
            path.display()
        ));
        if o.status.success() {
            accepted.push(name);
        } else {
            assert_eq!(o.status.code(), Some(2));
            assert!(
                String::from_utf8_lossy(&o.stderr).contains("peer receipt"),
                "{}",
                String::from_utf8_lossy(&o.stderr)
            );
            assert!(!out.exists(), "refusal must leave no comparison output");
        }
        if out.exists() {
            fs::remove_dir_all(out).unwrap();
        }
    }
    for p in [data, cache, peer] {
        fs::remove_dir_all(p).unwrap();
    }
    assert!(accepted.is_empty(), "incompatible peer receipts accepted: {accepted:?}");
}

#[test]
fn sweep_accepts_same_test_peers_with_a_smaller_training_prefix() {
    let data = scratch("peer-compatible-data");
    write_treebank(&data);
    let cache = scratch("peer-compatible-cache");
    run_ok(&format!(
        "cache --corpus ud-en-ewt --dir {} --bundle {} --out {} --unrecorded-corpus --layers 2 --commit test",
        data.display(),
        tiny().display(),
        cache.display()
    ));
    for (task, id, column) in [
        ("parse", "ud-en-ewt", "deps"),
        ("lemma", "ud-en-ewt", "lemma"),
        ("tag", "ud-en-ewt-upos", "upos"),
        ("tag", "ud-en-ewt-xpos", "xpos"),
        ("tag", "ud-en-ewt-feats", "feats"),
    ] {
        let peer = scratch(&format!("compatible-{column}-peer"));
        let out = scratch(&format!("compatible-{column}-sweep"));
        run_ok(&format!(
            "peer --corpus {id} --dir {} --peer stanza-test --predictions {} --column {column} --out {} --unrecorded-corpus --commit test",
            data.display(),
            data.join(EN_EWT.test.path).display(),
            peer.display()
        ));
        run_ok(&format!(
            "sweep --corpus {id} --task {task} --dir {} --cache {} --out {} --peer-receipt {} --unrecorded-corpus --min-tree-freq 1 --arc 8 --rel 4 --epochs 2 --batch 8 --max-train 3 --commit test",
            data.display(),
            cache.display(),
            out.display(),
            peer.join("receipt.json").display()
        ));
        let record: SweepRecord = serde_json::from_slice(&fs::read(out.join("sweep.json")).unwrap()).unwrap();
        assert_eq!(record.peer.unwrap().test, 1.0);
        assert_eq!(record.runs.len(), 3);
        for row in record.runs {
            assert_eq!(row.vs_peer, Some(row.test - 1.0));
            assert_eq!(row.uas_vs_peer.is_some(), task == "parse");
        }
        for p in [peer, out] {
            fs::remove_dir_all(p).unwrap();
        }
    }
    for p in [data, cache] {
        fs::remove_dir_all(p).unwrap();
    }
}
