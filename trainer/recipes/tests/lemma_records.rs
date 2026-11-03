//! The committed #1823 lemma head `--min-tree-freq`, `--seen-lookup`, and
//! `--bilstm` sweep records across `ud-en-ewt`, `ud-it-markit`,
//! `ud-de-gsd`, and `ud-es-ancora`: 48 training receipts, 4 majority
//! baseline receipts, and 4 Stanza peer receipts.

use std::path::{Path, PathBuf};

use turbo::manifest::{HeadKind, LicenseClass};
use turbo_trainer_recipes::eval::Metrics;
use turbo_trainer_recipes::receipt::{Receipt, SCHEMA};
use turbo_trainer_recipes::ud::{DE_GSD, EN_EWT, ES_ANCORA, IT_MARKIT, Treebank};

const VARIANTS: [&str; 12] = [
    "mtf3-nolookup",
    "mtf2-nolookup",
    "mtf1-nolookup",
    "mtf3-lookup2",
    "mtf2-lookup2",
    "mtf1-lookup2",
    "mtf3-lookup1",
    "mtf2-lookup1",
    "mtf1-lookup1",
    "mtf2-bilstm-nolookup",
    "mtf2-bilstm-lookup2",
    "mtf2-bilstm-lookup1",
];

fn records_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("records/lemma-seen-lookup-1823")
}

fn receipt(bank: &str, name: &str) -> Receipt {
    let p = records_dir().join(bank).join(format!("{name}.json"));
    serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn round(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

struct ExpectedBank {
    dir: &'static str,
    bank: Treebank,
    class: LicenseClass,
    encoder: &'static str,
    encoder_sha256: &'static str,
    test_words: usize,
    unseen_words: usize,
    majority: (f64, f64, f64),
    stanza_peer: &'static str,
    stanza: (f64, f64, f64, f64, f64, f64),
    winner: &'static str,
    winner_dev: f64,
    winner_trees: usize,
    lookup2_entries: usize,
    lookup1_entries: usize,
    winner_scores: (f64, f64, f64, f64, f64, f64),
    safetensors_sha256: &'static str,
    head_json_sha256: &'static str,
}

const BANKS: [ExpectedBank; 4] = [
    ExpectedBank {
        dir: "en-ewt",
        bank: EN_EWT,
        class: LicenseClass::ShareAlike,
        encoder: "answerdotai/ModernBERT-base",
        encoder_sha256: "65a0720298fc5fe7f229ee57ffbe1ccf0eb3836a91526840ca23fa30a5937ac8",
        test_words: 25_094,
        unseen_words: 2_292,
        majority: (0.9503, 0.9812, 0.6431),
        stanza_peer: "stanza-1.10.1",
        stanza: (0.9683, 0.9849, 0.8028, 0.0180, 0.0158, 0.0202),
        winner: "mtf3-lookup1",
        winner_dev: 0.9576,
        winner_trees: 337,
        lookup2_entries: 8_896,
        lookup1_entries: 18_697,
        winner_scores: (0.9579, 0.9809, 0.7286, 0.0076, 0.0052, 0.0099),
        safetensors_sha256: "1cba58c78ee009cfe0f866d0e585d54f694e293904e164bf3b25dba72f45b93b",
        head_json_sha256: "be8c81065f89830793fa8cd28aadaddcd8428c1fe475478b0a919423d7625729",
    },
    ExpectedBank {
        dir: "it-markit",
        bank: IT_MARKIT,
        class: LicenseClass::Permissive,
        encoder: "FacebookAI/xlm-roberta-base",
        encoder_sha256: "4c5a35e444094f547df03c0e1745a0d13375f17e06430f1d2dd0c185a643ad52",
        test_words: 10_196,
        unseen_words: 1_306,
        majority: (0.8172, 0.8654, 0.4893),
        stanza_peer: "stanza-1.15.0-it_markit",
        stanza: (0.8695, 0.8751, 0.8308, 0.0523, 0.0475, 0.0573),
        winner: "mtf2-bilstm-lookup1",
        winner_dev: 0.9397,
        winner_trees: 185,
        lookup2_entries: 1_391,
        lookup1_entries: 3_980,
        winner_scores: (0.8530, 0.8732, 0.7152, 0.0358, 0.0317, 0.0403),
        safetensors_sha256: "0c305b865db5e34440e7d7ea20d1f233723c5d2af5711e496921f43bdb38fcc9",
        head_json_sha256: "d8e8f70560ba855a886cc1146979daf35c19812b531332398de7ee9a3e2ce461",
    },
    ExpectedBank {
        dir: "de-gsd",
        bank: DE_GSD,
        class: LicenseClass::ShareAlike,
        encoder: "FacebookAI/xlm-roberta-base",
        encoder_sha256: "4c5a35e444094f547df03c0e1745a0d13375f17e06430f1d2dd0c185a643ad52",
        test_words: 16_499,
        unseen_words: 1_932,
        majority: (0.9424, 0.9901, 0.5823),
        stanza_peer: "stanza-1.15.0-de_gsd",
        stanza: (0.9743, 0.9912, 0.8468, 0.0319, 0.0290, 0.0351),
        winner: "mtf2-bilstm-lookup1",
        winner_dev: 0.9697,
        winner_trees: 616,
        lookup2_entries: 14_634,
        lookup1_entries: 48_968,
        winner_scores: (0.9670, 0.9920, 0.7780, 0.0246, 0.0221, 0.0271),
        safetensors_sha256: "918a79d49ec12466c28f719a7401e36eed8de4749a84dfb39802c5f3a4bb22dc",
        head_json_sha256: "1701aa7bf53e5d6eef5845248151a119215c70a3292be1db4094bb59ac177251",
    },
    ExpectedBank {
        dir: "es-ancora",
        bank: ES_ANCORA,
        class: LicenseClass::Permissive,
        encoder: "FacebookAI/xlm-roberta-base",
        encoder_sha256: "4c5a35e444094f547df03c0e1745a0d13375f17e06430f1d2dd0c185a643ad52",
        test_words: 53_622,
        unseen_words: 2_847,
        majority: (0.9677, 0.9897, 0.5753),
        stanza_peer: "stanza-1.15.0-es_ancora",
        stanza: (0.9900, 0.9922, 0.9505, 0.0224, 0.0207, 0.0240),
        winner: "mtf2-bilstm-lookup1",
        winner_dev: 0.9899,
        winner_trees: 695,
        lookup2_entries: 17_513,
        lookup1_entries: 36_065,
        winner_scores: (0.9883, 0.9949, 0.8711, 0.0206, 0.0193, 0.0220),
        safetensors_sha256: "b1a569fb8d1a25dc29a729e0c7c4ce97cd657657400de5812afa7e8cc43731e8",
        head_json_sha256: "c3530dfe640df8da2cbc60833a782253cd42c3fe2f3b39272e03c23e4a1397b4",
    },
];

#[test]
fn lemma_sweep_winners_baselines_and_stanza_peers_match_readme() {
    for b in &BANKS {
        let corpus_id = format!("{}-lemma", b.bank.id);
        let pinned_shas: Vec<&str> = [b.bank.train.sha256, b.bank.dev.sha256, b.bank.test.sha256].into();

        // Majority baseline receipt.
        let maj = receipt(b.dir, "majority");
        assert_eq!(maj.schema, SCHEMA);
        assert_eq!(maj.corpus.id, corpus_id);
        assert!(maj.corpus.recorded);
        assert_eq!(maj.corpus.files.iter().map(|f| f.sha256.as_str()).collect::<Vec<_>>(), pinned_shas);
        assert!(maj.head.is_none() && maj.encoder.is_none() && maj.peer.is_none() && maj.eval.model.is_none());
        let mb = &maj.eval.baseline;
        assert_eq!(
            (round(mb.score()), round(mb.seen_accuracy().unwrap()), round(mb.unseen_accuracy().unwrap())),
            b.majority,
            "{} majority",
            b.dir
        );

        // Stanza peer receipt.
        let st = receipt(b.dir, "stanza");
        assert_eq!(st.schema, SCHEMA);
        assert_eq!(st.corpus.id, corpus_id);
        assert!(st.corpus.recorded);
        assert_eq!(st.peer.as_ref().unwrap().name, b.stanza_peer);
        let sm = st.eval.model.as_ref().unwrap();
        let sc = st.eval.comparison.as_ref().unwrap();
        assert_eq!(
            (
                round(sm.score()),
                round(sm.seen_accuracy().unwrap()),
                round(sm.unseen_accuracy().unwrap()),
                round(sc.delta),
                round(sc.low),
                round(sc.high),
            ),
            b.stanza,
            "{} stanza",
            b.dir
        );

        // Sweep variants and dev-selected winner.
        let mut best_dev = ("", -1.0_f64);
        for &v in &VARIANTS {
            let r = receipt(b.dir, v);
            assert_eq!(r.schema, SCHEMA);
            assert_eq!(r.corpus.id, corpus_id);
            assert!(r.corpus.recorded);
            assert_eq!(r.corpus.files.iter().map(|f| f.sha256.as_str()).collect::<Vec<_>>(), pinned_shas);
            let enc = r.encoder.as_ref().unwrap();
            assert_eq!((enc.model.as_str(), enc.manifest_sha256.as_str()), (b.encoder, b.encoder_sha256));
            let spec = r.head.as_ref().unwrap();
            spec.validate("head").unwrap_or_else(|e| panic!("{}/{v}: {}", b.dir, e.message));
            assert_eq!(spec.kind, HeadKind::EditTreeLemmatizer);
            assert_eq!(spec.license_class, Some(b.class));
            if v.ends_with("-nolookup") {
                assert!(spec.seen_lookup.is_none(), "{}/{v}", b.dir);
            } else if v.ends_with("-lookup2") {
                let sl = spec.seen_lookup.as_ref().unwrap();
                assert_eq!((sl.min_freq, sl.entries.len()), (2, b.lookup2_entries), "{}/{v}", b.dir);
            } else if v.ends_with("-lookup1") {
                let sl = spec.seen_lookup.as_ref().unwrap();
                assert_eq!((sl.min_freq, sl.entries.len()), (1, b.lookup1_entries), "{}/{v}", b.dir);
                assert!(r.eval.comparison.as_ref().unwrap().low > 0.0, "{}/{v} clears baseline", b.dir);
            }
            let tr = r.training.as_ref().unwrap();
            let dev = tr.dev[tr.best_epoch];
            if dev > best_dev.1 {
                best_dev = (v, dev);
            }
            let Metrics::Tags { words, unseen_words, .. } = r.eval.model.as_ref().unwrap() else {
                panic!("{}/{v}: expected tag metrics", b.dir);
            };
            assert_eq!((*words, *unseen_words), (b.test_words, b.unseen_words));
        }
        assert_eq!(best_dev.0, b.winner, "{} dev-best variant", b.dir);
        assert_eq!(round(best_dev.1), b.winner_dev, "{} winner dev score", b.dir);

        let win = receipt(b.dir, b.winner);
        let wm = win.eval.model.as_ref().unwrap();
        let wc = win.eval.comparison.as_ref().unwrap();
        assert!(wc.low > 0.0, "{} winner clears gate", b.dir);
        assert_eq!(win.head.as_ref().unwrap().labels.len(), b.winner_trees);
        assert_eq!(
            (
                round(wm.score()),
                round(wm.seen_accuracy().unwrap()),
                round(wm.unseen_accuracy().unwrap()),
                round(wc.delta),
                round(wc.low),
                round(wc.high),
            ),
            b.winner_scores,
            "{} winner scores",
            b.dir
        );
        let st_out = win.outputs.iter().find(|o| o.path == "head.safetensors").unwrap();
        let hj_out = win.outputs.iter().find(|o| o.path == "head.json").unwrap();
        assert_eq!(st_out.sha256, b.safetensors_sha256);
        assert_eq!(hj_out.sha256, b.head_json_sha256);
    }
}

#[test]
fn lemma_sweep_receipts_name_no_machine_or_host_path() {
    for b in &BANKS {
        let dir = records_dir().join(b.dir);
        let mut count = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            count += 1;
            let name = path.file_stem().unwrap().to_str().unwrap();
            let r = receipt(b.dir, name);
            assert!(
                ["rtx-4080-machine", "cpu-32-core-machine"].contains(&r.host.as_str()),
                "{}/{name}: host {}",
                b.dir,
                r.host
            );
            assert!(
                r.commit.len() == 40 && r.commit.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{}/{name}: commit {}",
                b.dir,
                r.commit
            );
            let want_commit = match (b.dir, name) {
                (_, "majority") => "4873eb828a13345e1a41ab6a23c0b71103d89c63",
                ("en-ewt", "stanza") => "befdc5b370379308049008ca008f12f98b3dd1b0",
                ("de-gsd", "stanza") => "056f2854b4e015fb51e4708480d2b30635aa8f90",
                (_, "stanza") => "5c57fdd764d6bf5779c081fc8f18f8cb34178afd",
                _ => "2a05cf523424b06981f8281c819f4e54189a0c66",
            };
            assert_eq!(r.commit, want_commit, "{}/{name}: commit", b.dir);
            assert!(
                r.command.iter().all(|c| !c.starts_with('/') && !c.contains("/work/") && !c.contains("/home/")),
                "{}/{name}: path in command {:?}",
                b.dir,
                r.command
            );
            assert!(r.corpus.files.iter().all(|f| !f.path.contains('/')), "{}/{name}: corpus path", b.dir);
            assert!(r.outputs.iter().all(|o| !o.path.contains('/')), "{}/{name}: output path", b.dir);
            let raw = std::fs::read_to_string(&path).unwrap().to_lowercase();
            assert!(!raw.contains("opennlp") && !raw.contains("/work/"), "{}/{name}: host path or OpenNLP", b.dir);
        }
        assert_eq!(count, 14, "{} has 12 sweep + majority + stanza receipts", b.dir);
    }
}

#[test]
fn lemma_baseline_command_replays_through_cli() {
    let tmp = std::env::temp_dir().join(format!("turbo-lemma-base-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let corpus = tmp.join("corpus");
    let out = tmp.join("out");
    std::fs::create_dir_all(&corpus).unwrap();
    let sample = "1\tCats\tcat\tNOUN\tNNS\tNumber=Plur\t2\tnsubj\t_\t_\n\
                  2\trun\trun\tVERB\tVBP\tMood=Ind\t0\troot\t_\tSpaceAfter=No\n\
                  3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n\n";
    for f in [EN_EWT.train, EN_EWT.dev, EN_EWT.test] {
        std::fs::write(corpus.join(f.path), sample).unwrap();
    }
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args([
            "baseline",
            "--corpus",
            "ud-en-ewt-lemma",
            "--dir",
            corpus.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--unrecorded-corpus",
            "--commit",
            "4873eb828a13345e1a41ab6a23c0b71103d89c63",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "turbo-train-head baseline ud-en-ewt-lemma exited with {status}");
    let r: Receipt = serde_json::from_slice(&std::fs::read(out.join("receipt.json")).unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&tmp);
    assert_eq!(r.corpus.id, "ud-en-ewt-lemma");
    assert!(r.eval.model.is_none());
    assert_eq!(r.eval.baseline.score(), 1.0);
}
