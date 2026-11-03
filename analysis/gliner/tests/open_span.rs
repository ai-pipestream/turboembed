//! Running a `HEAD_OPEN_SPAN` bundle through [`OpenSpanModel`] and
//! [`GlinerAnnotator`]: zero-shot caller label queries, exact agreement
//! with `turbo::head::OpenSpanMatcher`, sentence bounds, and evidence.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo::bundle::sha256_hex;
use turbo::head::open_span::{OpenSpanWork, tensors};
use turbo::head::tagger::Work;
use turbo::manifest::HeadSpec;
use turbo_analysis::{Analyzer, Annotator, Bound, Document, Ix, Key, Layer, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_gliner::{Candidate, Decisions, Decode, ExtractionSchema, GlinerAnnotator, OpenSpanModel, decode};

const TEXT: &str = "Ann met Bob in Paris . The dog slept .";

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn values(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            (((s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn open_span_bundle(name: &str) -> PathBuf {
    open_span_bundle_with_bias(name, [0.5, 1.0, 1.0])
}

fn open_span_bundle_with_bias(name: &str, bias: [f32; 3]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-gliner-open-span-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(tiny().join(f), dir.join(f)).unwrap();
    }
    let src = fs::read(tiny().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    header.remove("__metadata__");
    let mut data = src[8 + n..].to_vec();
    let mut m: Value = serde_json::from_slice(&fs::read(tiny().join("manifest.json")).unwrap()).unwrap();
    let head_json = json!({
        "name": "open_ner",
        "layer": "turboembed:entities",
        "kind": "HEAD_OPEN_SPAN",
        "labels": ["person", "location"],
        "scheme": "SCHEME_TAGS",
        "word_pooling": "WORD_POOLING_FIRST",
        "decode": "DECODE_SPANS_FLAT",
        "span": {"max_words": 3, "width_dim": 4},
        "artifact": "weights-f32",
        "tensor_prefix": "heads.open_ner"
    });
    m["heads"] = json!([head_json]);
    let spec: HeadSpec = serde_json::from_value(head_json).unwrap();
    for (seed, (suffix, shape)) in (1u64..).zip(tensors(&spec, 32)) {
        let count = shape.iter().product::<u64>() as usize;
        let start = data.len();
        let vals = if suffix == "out.bias" { bias.to_vec() } else { values(seed, count, 0.8) };
        data.extend(vals.iter().flat_map(|v| v.to_le_bytes()));
        let key = format!("{}.{suffix}", spec.tensor_prefix);
        header.insert(key, json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
    }
    let mut head = serde_json::to_vec(&header).unwrap();
    head.resize(head.len().next_multiple_of(8), b' ');
    let mut weights = (head.len() as u64).to_le_bytes().to_vec();
    weights.extend(head);
    weights.extend(data);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();
    let files = m["files"].as_array_mut().unwrap();
    let w = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

fn tokens_and_sentences(text: &str) -> (Tokens, Sentences) {
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    let mut sent_idx = Vec::new();
    let mut start = 0usize;
    let mut sent_start = 0u32;
    let mut cur_sent = 0u32;
    for word in text.split(' ') {
        let end = start + word.len();
        tok_spans.push(Span::new(start as u32, end as u32));
        sent_idx.push(Ix::new(cur_sent));
        if word == "." {
            sent_spans.push(Span::new(sent_start, end as u32));
            cur_sent += 1;
            sent_start = (end + 1) as u32;
        }
        start = end + 1;
    }
    (Tokens { spans: tok_spans, sentence: Some(sent_idx) }, Sentences { spans: sent_spans })
}

#[test]
fn open_span_bundle_annotates_zero_shot_labels_and_builds_evidence() {
    let dir = open_span_bundle("annotate");
    let opts = EncoderOptions::default();
    let schema =
        ExtractionSchema::new(vec!["person".into(), "location".into(), "spacecraft".into()], 0.5, Decode::Flat)
            .unwrap();
    let model = OpenSpanModel::open(&dir, None, &opts).unwrap();
    let annotator = GlinerAnnotator::open_bundle(&dir, Some("open_ner"), &opts, schema.clone()).unwrap();
    assert!(OpenSpanModel::open(&dir, Some("missing"), &opts).is_err());

    let mut doc = Document::new(TEXT);
    let (toks, sents) = tokens_and_sentences(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let (entities, decisions) = annotator.annotate((view, &toks, Some(&sents))).unwrap();
    entities.check().unwrap();
    decisions.check().unwrap();
    assert!(!entities.spans.is_empty(), "seeded head with positive bias emits spans");
    assert_eq!(entities.spans.len(), decisions.logits.len());

    // Direct predict on the first sentence's words matches the first sentence's entities.
    let first_words = ["Ann", "met", "Bob", "in", "Paris", "."];
    let direct = model.predict(&first_words, &schema).unwrap();
    let first_entities: Vec<(usize, usize, u16)> = entities
        .tokens
        .iter()
        .zip(&entities.kind)
        .filter(|(r, _)| r.end.get() <= first_words.len())
        .map(|(r, k)| (r.start.get(), r.end.get(), k.0))
        .collect();
    let want: Vec<(usize, usize, u16)> = direct.iter().map(|c| (c.start, c.end, c.label as u16)).collect();
    assert_eq!(first_entities, want);

    // Core matcher decode at threshold 0.0 agrees with Flat decode at 0.5.
    let label_states = model.embed_labels(schema.labels()).unwrap();
    let word_feats = model.encode_words(&first_words).unwrap();
    let word_map: Vec<u32> = (0..first_words.len() as u32).collect();
    let (mut raw_scores, mut core_spans, mut work, mut sw) =
        (Vec::new(), Vec::new(), Work::default(), OpenSpanWork::default());
    model.head().scores(&word_feats, &word_map, None, &label_states, &mut raw_scores, &mut work, &mut sw).unwrap();
    model.head().decode(&raw_scores, first_words.len(), schema.labels().len(), &mut core_spans).unwrap();
    let core_triples: Vec<(usize, usize, u16)> =
        core_spans.iter().map(|s| (s.first as usize, s.last as usize + 1, s.label as u16)).collect();
    assert_eq!(want, core_triples);
    for (cand, core_sp) in direct.iter().zip(&core_spans) {
        assert_eq!(cand.logit.to_bits(), core_sp.margin.to_bits());
    }

    doc.add(&Key::<Sentences>::DEFAULT, sents, Provenance::by("fixture:sentences")).unwrap();
    doc.add(&Key::<Tokens>::DEFAULT, toks, Provenance::by("fixture:tokens")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .step(Bound::new(annotator))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    for r in pipeline.run(&mut docs) {
        r.unwrap();
    }
    let doc = docs.pop().unwrap();
    let ev = GlinerAnnotator::evidence(&doc, Key::<Entities>::DEFAULT, Key::<Decisions>::DEFAULT).unwrap();
    ev.verify(&doc).unwrap();
    assert_eq!(ev.raw.len(), doc.require(&Key::<Entities>::DEFAULT).unwrap().len());
}

#[test]
fn open_span_loads_from_zoo_by_id_and_language_and_builtin_index_pins_permissive_head() {
    use turbo_zoo::{
        Acceptances, Artifact, Component, Directory, File, Format, Index, Kind, Licence, LicenceClass, Policy, Sha256,
        Sha512, TrainingData, UserId, Zoo,
    };
    let builtin = Index::builtin();
    let entry = builtin.get("turbo-en-wnut17-modernbert-base-l12-open-span-1").unwrap();
    assert_eq!(entry.class, LicenceClass::Permissive);
    assert_eq!(entry.component, Some(Component::EntityExtractor));
    assert_eq!(
        builtin
            .find_within("en", Component::EntityExtractor, LicenceClass::Permissive)
            .into_iter()
            .map(|a| a.id.as_str())
            .collect::<Vec<_>>(),
        ["turbo-en-wnut17-modernbert-base-l12-open-span-1"]
    );

    let dir = open_span_bundle("zoo");
    let mut files = Vec::new();
    for rel in ["manifest.json", "tokenizer.json", "weights/model.safetensors", "reference/reference.safetensors"] {
        let bytes = fs::read(dir.join(rel)).unwrap();
        files.push(File {
            path: rel.parse().unwrap(),
            url: format!("https://example.org/{rel}").parse().unwrap(),
            size: bytes.len() as u64,
            sha256: Sha256::of(&bytes),
            sha512: Sha512::of(&bytes),
            download: None,
        });
    }
    let artifact = Artifact {
        id: "turbo-en-test-open-span-1".parse().unwrap(),
        kind: Kind::Head,
        component: Some(Component::EntityExtractor),
        language: Some("en".parse().unwrap()),
        format: Format::Safetensors,
        version: "1".into(),
        license: Licence::Apache2,
        license_terms: "https://www.apache.org/licenses/LICENSE-2.0".parse().unwrap(),
        trained_on: vec![TrainingData {
            name: "WNUT-17".into(),
            license: Licence::CcBy40,
            license_terms: "https://creativecommons.org/licenses/by/4.0/".parse().unwrap(),
        }],
        class: LicenceClass::Permissive,
        attribution: Some("Trained on WNUT-17 (CC BY 4.0)".into()),
        origin: None,
        files,
    };
    let index = Index::parse(Index::to_canonical_json(&[artifact]).as_bytes()).unwrap();
    let cache = std::env::temp_dir().join(format!("turbo-gliner-open-span-zoo-cache-{}", std::process::id()));
    let _ = fs::remove_dir_all(&cache);
    let transport = Directory::new(&dir);
    let zoo = Zoo::new(index, cache, Policy::default(), &transport);
    let user: UserId = "alice".parse().unwrap();
    let acc = Acceptances::default();
    let schema = ExtractionSchema::new(vec!["person".into(), "location".into()], 0.5, Decode::Flat).unwrap();
    let by_id = GlinerAnnotator::from_zoo(
        &zoo,
        "turbo-en-test-open-span-1",
        None,
        &EncoderOptions::default(),
        schema.clone(),
        &user,
        &acc,
    )
    .unwrap();
    let by_lang =
        GlinerAnnotator::from_zoo_language(&zoo, "en", None, &EncoderOptions::default(), schema, &user, &acc).unwrap();
    assert_eq!(by_id.config_hash(), by_lang.config_hash());
}

#[test]
fn open_span_ranks_by_raw_logit_and_agrees_with_core_under_saturated_scores_and_competing_spans() {
    let dir = open_span_bundle_with_bias("saturated", [100.0, 1.0, 1.0]);
    let opts = EncoderOptions::default();
    let schema = ExtractionSchema::new(vec!["person".into(), "location".into()], 0.5, Decode::Flat).unwrap();
    let model = OpenSpanModel::open(&dir, None, &opts).unwrap();
    let annotator = GlinerAnnotator::open_bundle(&dir, None, &opts, schema.clone()).unwrap();

    let first_words = ["Ann", "met", "Bob", "in", "Paris", "."];
    let direct = model.predict(&first_words, &schema).unwrap();
    assert!(!direct.is_empty());
    assert!(direct.iter().all(|c| c.score == 1.0 && c.logit > 80.0));

    let label_states = model.embed_labels(schema.labels()).unwrap();
    let word_feats = model.encode_words(&first_words).unwrap();
    let word_map: Vec<u32> = (0..first_words.len() as u32).collect();
    let (mut raw_scores, mut core_spans, mut work, mut sw) =
        (Vec::new(), Vec::new(), Work::default(), OpenSpanWork::default());
    model.head().scores(&word_feats, &word_map, None, &label_states, &mut raw_scores, &mut work, &mut sw).unwrap();
    model.head().decode(&raw_scores, first_words.len(), schema.labels().len(), &mut core_spans).unwrap();
    let got_triples: Vec<(usize, usize, u32)> = direct.iter().map(|c| (c.start, c.end, c.label as u32)).collect();
    let core_triples: Vec<(usize, usize, u32)> =
        core_spans.iter().map(|s| (s.first as usize, s.last as usize + 1, s.label)).collect();
    assert_eq!(got_triples, core_triples);
    for (c, s) in direct.iter().zip(&core_spans) {
        assert_eq!(c.logit.to_bits(), s.margin.to_bits());
    }

    let doc = Document::new(TEXT);
    let (toks, sents) = tokens_and_sentences(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let (entities, decisions) = annotator.annotate((view, &toks, Some(&sents))).unwrap();
    let first_annot: Vec<(usize, usize, u32)> = entities
        .tokens
        .iter()
        .zip(&entities.kind)
        .filter(|(r, _)| r.end.get() <= first_words.len())
        .map(|(r, k)| (r.start.get(), r.end.get(), k.0 as u32))
        .collect();
    assert_eq!(first_annot, core_triples);
    assert_eq!(decisions.logits[0].to_bits(), core_spans[0].margin.to_bits());

    let same_span = vec![
        Candidate { start: 0, end: 1, label: 0, logit: 20.0, score: turbo::head::sigmoid(20.0) },
        Candidate { start: 0, end: 1, label: 1, logit: 21.0, score: turbo::head::sigmoid(21.0) },
    ];
    assert_eq!(same_span[0].score, 1.0);
    assert_eq!(same_span[1].score, 1.0);
    let kept_same = decode(same_span, &schema);
    assert_eq!(kept_same.len(), 1);
    assert_eq!((kept_same[0].start, kept_same[0].end, kept_same[0].label), (0, 1, 1));

    let overlapping = vec![
        Candidate { start: 0, end: 2, label: 0, logit: 20.0, score: turbo::head::sigmoid(20.0) },
        Candidate { start: 1, end: 3, label: 1, logit: 21.0, score: turbo::head::sigmoid(21.0) },
    ];
    let kept_overlap = decode(overlapping, &schema);
    assert_eq!(kept_overlap.len(), 1);
    assert_eq!((kept_overlap[0].start, kept_overlap[0].end, kept_overlap[0].label), (1, 3, 1));
}
