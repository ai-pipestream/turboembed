//! `train-sentences` (#1725): multi-sentence document construction from
//! CoNLL-U (`# newdoc`, `# newpar`, `SpaceAfter=No`, `SpacesAfter`,
//! multiword tokens, non-ASCII text), newline and rule-splitter
//! baselines, exact sentence span scoring, and end-to-end training on
//! the tiny BERT bundle with loading into `SentenceBoundary` and
//! `SentenceTagger`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};
use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::{SentenceBoundary, boundary};
use turbo::manifest::{HeadKind, HeadSpec, LicenseClass};
use turbo_analysis::{Annotator, Document, Key, Span, Spans, View, ViewId};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::Tokens;
use turbo_tagger::SentenceTagger;
use turbo_trainer_recipes::receipt::{Receipt, SCHEMA};
use turbo_trainer_recipes::sentences::{
    SENTENCES_PER_DOCUMENT, bootstrap_spans, evaluate_spans, newline_baseline, predict, read, rule_baseline,
};
use turbo_trainer_recipes::ud::EN_EWT;

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-sentences-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

/// Non-ASCII multi-sentence documents with `# newdoc`, `# newpar`,
/// multiword tokens (`3-4 al`), empty nodes (`4.1`), and `SpaceAfter=No`.
fn fixture_treebank() -> String {
    let mut out = String::new();
    for doc in 0..12 {
        out += &format!("# newdoc id = doc-{doc}\n");
        for s in 0..4 {
            if s == 2 {
                out += "# newpar id = p-2\n";
            }
            out += &format!(
                "# sent_id = {doc}-{s}\n\
                 1\t¿\t¿\tPUNCT\t.\t_\t2\tpunct\t_\tSpaceAfter=No\n\
                 2\tVamos\tir\tVERB\tVBP\t_\t0\troot\t_\t_\n\
                 3-4\tal\t_\t_\t_\t_\t_\t_\t_\t_\n\
                 3\ta\ta\tADP\tIN\t_\t5\tcase\t_\t_\n\
                 4\tel\tel\tDET\tDT\t_\t5\tdet\t_\t_\n\
                 4.1\t_\t_\tX\t_\t_\t_\t_\t2:dep\t_\n\
                 5\tcafé\tcafé\tNOUN\tNN\t_\t2\tobl\t_\tSpaceAfter=No\n\
                 6\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n\n"
            );
        }
    }
    out
}

fn write_treebank(dir: &Path, text: &str) {
    fs::create_dir_all(dir).unwrap();
    for pin in [EN_EWT.train, EN_EWT.dev, EN_EWT.test] {
        fs::write(dir.join(pin.path), text).unwrap();
    }
}

#[test]
fn ud_documents_preserve_newdoc_newpar_multiwords_and_non_ascii_offsets() {
    let dir = scratch("ud-doc");
    write_treebank(&dir, &fixture_treebank());
    let corpus = read("ud-en-ewt", &dir, true).unwrap();
    assert_eq!(corpus.record.id, "ud-en-ewt-sentences");
    assert_eq!(corpus.language, "en");
    assert_eq!(corpus.test.len(), 12, "12 # newdoc documents");
    assert_eq!((corpus.record.test.sentences, corpus.record.test.words), (48, 240));

    let d = &corpus.test[0];
    assert_eq!(
        d.text, "¿Vamos al café. ¿Vamos al café.\n¿Vamos al café. ¿Vamos al café.",
        "SpaceAfter=No after ¿ and café, multiword al kept whole, # newpar becomes newline"
    );
    assert_eq!(d.sentences.len(), 4);
    for (i, &sp) in d.sentences.iter().enumerate() {
        assert_eq!(&d.text[sp.range()], "¿Vamos al café.", "sentence {i} slice");
    }
    for (w, &sp) in d.words.iter().zip(&d.word_spans) {
        assert_eq!(&d.text[sp.range()], w.as_str());
    }

    // Newline baseline gets 2 spans per document (because of 1 # newpar newline),
    // neither matching the 4 gold sentences because each line holds 2 sentences.
    let nl = newline_baseline(&corpus.test);
    assert_eq!(nl[0].len(), 2);
    assert_eq!(evaluate_spans(&corpus.test, &nl).score(), 0.0);

    // Rule splitter splits all 4 sentences per document accurately.
    let rules = rule_baseline("es", &corpus.test);
    assert_eq!(rules[0], d.sentences);
    assert_eq!(evaluate_spans(&corpus.test, &rules).score(), 1.0);
    let cmp = bootstrap_spans(&corpus.test, &rules, &nl, 100, 1);
    assert_eq!((cmp.model, cmp.baseline, cmp.delta), (1.0, 0.0, 1.0));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ud_without_newdoc_groups_by_sentences_per_document() {
    let dir = scratch("ud-no-newdoc");
    let mut text = String::new();
    for i in 0..25 {
        text += &format!(
            "# sent_id = s-{i}\n1\tHello\thello\tINTJ\tUH\t_\t0\troot\t_\tSpaceAfter=No\n2\t.\t.\tPUNCT\t.\t_\t1\tpunct\t_\t_\n\n"
        );
    }
    write_treebank(&dir, &text);
    let corpus = read("ud-en-ewt", &dir, true).unwrap();
    let lens: Vec<usize> = corpus.test.iter().map(|d| d.sentences.len()).collect();
    assert_eq!(lens, [SENTENCES_PER_DOCUMENT, SENTENCES_PER_DOCUMENT, 5]);
    fs::remove_dir_all(dir).unwrap();
}

/// The tensors of a safetensors file, by name.
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

fn train_cmd(data: &Path, out: &Path) -> Receipt {
    let bundle = tiny();
    let args = [
        "train-sentences",
        "--corpus",
        "ud-en-ewt",
        "--dir",
        data.to_str().unwrap(),
        "--bundle",
        bundle.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--unrecorded-corpus",
        "--epochs",
        "6",
        "--batch",
        "4",
        "--lr",
        "0.05",
        "--commit",
        "test",
    ];
    let o = run(&args);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap()
}

/// Copies the tiny BERT bundle and appends the trained `sentences` head
/// tensors into `weights/model.safetensors` so `SentenceTagger::open`
/// loads the bundle directly.
fn bundle_with_head(head_dir: &Path, dst: &Path) {
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dst.join(f).parent().unwrap()).unwrap();
        fs::copy(tiny().join(f), dst.join(f)).unwrap();
    }
    let src = fs::read(tiny().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    header.remove("__metadata__");
    let mut data = src[8 + n..].to_vec();

    let head_tensors = read_safetensors(&fs::read(head_dir.join("head.safetensors")).unwrap());
    for (name, shape, values) in head_tensors {
        let start = data.len();
        data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(name, json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
    }

    let mut head_bytes = serde_json::to_vec(&header).unwrap();
    head_bytes.resize(head_bytes.len().next_multiple_of(8), b' ');
    let mut weights = (head_bytes.len() as u64).to_le_bytes().to_vec();
    weights.extend(head_bytes);
    weights.extend(data);
    fs::create_dir_all(dst.join("weights")).unwrap();
    fs::write(dst.join("weights/model.safetensors"), &weights).unwrap();

    let spec: Value = serde_json::from_slice(&fs::read(head_dir.join("head.json")).unwrap()).unwrap();
    let mut m: Value = serde_json::from_slice(&fs::read(tiny().join("manifest.json")).unwrap()).unwrap();
    m["model"]["license_class"] = json!("CLASS_PERMISSIVE");
    m["heads"] = json!([spec]);
    let files = m["files"].as_array_mut().unwrap();
    let w = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(dst.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
}

#[test]
fn train_sentences_decreases_loss_is_deterministic_and_loads_into_sentence_tagger() {
    let data = scratch("data");
    write_treebank(&data, &fixture_treebank());
    let out = scratch("out-a");
    let r = train_cmd(&data, &out);

    assert_eq!(r.corpus.id, "ud-en-ewt-sentences");
    let head = r.head.clone().unwrap();
    assert_eq!(
        (head.kind, head.name.as_str(), head.layer.as_str(), head.labels.as_slice()),
        (
            HeadKind::SentenceBoundary,
            "sentences",
            "turboembed:sentences",
            &["inside".to_owned(), "last".to_owned()][..]
        )
    );
    assert_eq!(head.license_class, Some(LicenseClass::ShareAlike));
    let stored: HeadSpec = serde_json::from_slice(&fs::read(out.join("head.json")).unwrap()).unwrap();
    assert_eq!(stored, head);

    let tr = r.training.as_ref().unwrap();
    assert!(tr.train_loss.iter().all(|x| x.is_finite()), "non-finite loss: {:?}", tr.train_loss);
    assert!(tr.train_loss.last().unwrap() < tr.train_loss.first().unwrap(), "loss did not fall: {:?}", tr.train_loss);
    assert_eq!(r.eval.model.as_ref().unwrap().score(), 1.0, "learns sentence ends after `.`");
    assert!(r.eval.comparison.as_ref().unwrap().delta > 0.0, "beats newline baseline");
    let rules = r.eval.rules.as_ref().unwrap();
    assert_eq!(rules.name, "sentdetect-rules-en");
    assert_eq!(rules.baseline.score(), 1.0);

    // Load directly into core's SentenceBoundary and into SentenceTagger.
    let t = read_safetensors(&fs::read(out.join("head.safetensors")).unwrap());
    let get = |s: &str| t.iter().find(|(k, _, _)| *k == format!("heads.sentences.{s}")).map(|(_, _, v)| v.clone());
    let sb = SentenceBoundary::from_tensors(&head, 32, &get).unwrap();
    let corpus = read("ud-en-ewt", &data, true).unwrap();
    let bundle_dir = scratch("bundle");
    bundle_with_head(&out, &bundle_dir);
    let bundle = Bundle::open(&bundle_dir).unwrap();
    let loaded = SentenceBoundary::load(&bundle, "sentences").unwrap();
    assert_eq!(loaded.spec().name, sb.spec().name);

    let tagger = SentenceTagger::open(&bundle_dir, Some("sentences"), &EncoderOptions::default()).unwrap();
    let d0 = &corpus.test[0];
    let doc = Document::new(d0.text.as_str());
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    for &sp in &d0.word_spans {
        tok_spans.push(sp);
    }
    let toks = Tokens { spans: tok_spans, sentence: None };
    let (sents,) = tagger.annotate((view, &toks, None)).unwrap();
    let got: Vec<Span> = sents.spans.iter().map(|(_, s)| s).collect();
    assert_eq!(got, d0.sentences, "exact byte offsets on non-ASCII text through SentenceTagger");
    let _ = predict(&sb, std::slice::from_ref(d0), &[vec![0.0; d0.words.len() * 32]]).unwrap();
    let _ = boundary::sentences(&[false, true]).unwrap();

    // Determinism for a fixed seed.
    let again = scratch("out-b");
    let r2 = train_cmd(&data, &again);
    assert_eq!(fs::read(again.join("head.safetensors")).unwrap(), fs::read(out.join("head.safetensors")).unwrap());
    assert_eq!(r2.eval, r.eval);

    for d in [data, out, again, bundle_dir] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn train_sentences_refuses_unknown_corpus_and_leaves_no_out_dir() {
    let out = scratch("refused");
    let o = run(&[
        "train-sentences",
        "--corpus",
        "unknown-corpus",
        "--dir",
        "x",
        "--bundle",
        "x",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown sentence corpus"));
    assert!(!out.exists());
}

fn round(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

#[test]
fn committed_sentence_receipts_verify_and_name_no_machine() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("records/sentences-1725");
    let expected = [
        ("masc-pos.json", "masc-sentences", LicenseClass::Permissive, 0.7781, 0.2558, "sentdetect-rules-en", 0.6520),
        (
            "ud-en-ewt.json",
            "ud-en-ewt-sentences",
            LicenseClass::ShareAlike,
            0.7681,
            0.2900,
            "sentdetect-rules-en",
            0.5970,
        ),
        (
            "ud-es-ancora.json",
            "ud-es-ancora-sentences",
            LicenseClass::Permissive,
            0.9755,
            0.0021,
            "sentdetect-rules-es",
            0.9414,
        ),
        (
            "ud-ca-ancora.json",
            "ud-ca-ancora-sentences",
            LicenseClass::Permissive,
            0.9908,
            0.0020,
            "sentdetect-rules-ca",
            0.9943,
        ),
        (
            "ud-fi-ftb.json",
            "ud-fi-ftb-sentences",
            LicenseClass::Permissive,
            0.9224,
            0.0000,
            "sentdetect-rules-fi",
            0.6720,
        ),
        (
            "ud-it-markit.json",
            "ud-it-markit-sentences",
            LicenseClass::Permissive,
            0.9912,
            0.0000,
            "sentdetect-rules-it",
            0.7136,
        ),
    ];
    for (file, corpus_id, class, model_f1, nl_f1, rules_name, rules_f1) in expected {
        let p = dir.join(file);
        let raw = fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let r: Receipt = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        assert_eq!(r.schema, SCHEMA);
        assert_eq!(r.corpus.id, corpus_id);
        assert!(r.corpus.recorded, "{file}: unrecorded corpus");
        assert_eq!(r.corpus.class().unwrap(), class);
        assert!(["cloud-container", "rtx-4080-machine"].contains(&r.host.as_str()), "{file}: host {}", r.host);
        assert!(r.command.iter().all(|c| !c.starts_with('/')), "{file}: absolute path in command");
        let lower = raw.to_lowercase();
        assert!(!lower.contains("opennlp") && !lower.contains("/work/"), "{file}: forbidden text");
        let head = r.head.as_ref().unwrap();
        assert_eq!(head.kind, HeadKind::SentenceBoundary);
        assert_eq!(head.license_class, Some(class));
        assert_eq!(round(r.eval.model.as_ref().unwrap().score()), model_f1, "{file} model F1");
        assert_eq!(round(r.eval.baseline.score()), nl_f1, "{file} newline F1");
        assert!(r.eval.comparison.as_ref().unwrap().low > 0.0, "{file} must beat newline baseline");
        let rules = r.eval.rules.as_ref().unwrap();
        assert_eq!(rules.name, rules_name);
        assert_eq!(round(rules.baseline.score()), rules_f1, "{file} rules F1");
    }
}
