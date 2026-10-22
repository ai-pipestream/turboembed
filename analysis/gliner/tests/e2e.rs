//! End-to-end validation and timing on multi-sentence documents, against the
//! reference written by bundle/reference/gliner/build_e2e_reference.py.
//! Needs GLINER_PACKAGE and GLINER_E2E_REFERENCE; see docs/gliner.md.
use serde_json::Value;
use std::{collections::BTreeSet, path::Path, sync::Arc, time::Instant};
use turbo_analysis::{Analyzer, Bound, Document, Key, Provenance, Span, Spans, View};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_gliner::{Decisions, Decode, ExtractionSchema, GlinerAnnotator, Model, package::Manifest};

/// (start, end, label) in document tokens, with score and logit.
type Entity = ((usize, usize, usize), f32, f32);

fn setup() -> (Arc<Model>, Value) {
    let root = std::env::var("GLINER_PACKAGE").expect("GLINER_PACKAGE");
    let reference = std::env::var("GLINER_E2E_REFERENCE").expect("GLINER_E2E_REFERENCE");
    let model = Arc::new(Manifest::load(Path::new(&root)).unwrap());
    (model, serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap())
}
fn schema(r: &Value, mode: Decode) -> ExtractionSchema {
    let labels = r["labels"].as_array().unwrap().iter().map(|l| l.as_str().unwrap().to_string()).collect();
    ExtractionSchema::new(labels, r["threshold"].as_f64().unwrap() as f32, mode).unwrap()
}
fn sentences(d: &Value) -> Vec<String> {
    d["sentences"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}
/// Words separated by single spaces, one Sentences span per sentence.
fn document(sentences: &[String]) -> Document {
    let mut doc = Document::new(sentences.join(" "));
    let view = doc.get::<View>().unwrap().id();
    let (mut tokens, mut spans, mut at) = (Spans::new(view), Spans::new(view), 0u32);
    for s in sentences {
        let start = at;
        for w in s.split(' ') {
            tokens.push(Span::new(at, at + w.len() as u32));
            at += w.len() as u32 + 1;
        }
        spans.push(Span::new(start, at - 1));
    }
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("fixture:sentences")).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("fixture:tokens")).unwrap();
    doc
}
fn annotate(model: &Arc<Model>, schema: &ExtractionSchema, doc: Document) -> Vec<Entity> {
    let pipeline = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .step(Bound::new(GlinerAnnotator::new(model.clone(), schema.clone())))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    for r in pipeline.run(&mut docs) {
        r.unwrap();
    }
    let e = docs[0].require(&Key::<Entities>::DEFAULT).unwrap();
    let d = docs[0].require(&Key::<Decisions>::DEFAULT).unwrap();
    (0..e.tokens.len())
        .map(|i| {
            let r = e.tokens[i];
            ((r.start.get(), r.end.get(), e.kind[i].0 as usize), e.score[i] as f32, d.logits[i])
        })
        .collect()
}
fn reference(decoded: &Value, offset: usize) -> Vec<Entity> {
    let n = |v: &Value| v.as_u64().unwrap() as usize;
    let f = |v: &Value| v.as_f64().unwrap() as f32;
    decoded
        .as_array()
        .unwrap()
        .iter()
        .map(|e| ((n(&e[0]) + offset, n(&e[1]) + offset, n(&e[2])), f(&e[3]), 0.0))
        .collect()
}
/// Matched keys, max score difference over them, and the union size.
fn agree(native: &[Entity], reference: &[Entity]) -> (usize, f32, usize) {
    let keys = |x: &[Entity]| x.iter().map(|e| e.0).collect::<BTreeSet<_>>();
    let (a, b) = (keys(native), keys(reference));
    let mut diff = 0f32;
    for n in native {
        if let Some(r) = reference.iter().find(|r| r.0 == n.0) {
            diff = diff.max((n.1 - r.1).abs());
        }
    }
    (a.intersection(&b).count(), diff, a.union(&b).count())
}

#[test]
#[ignore = "requires converted GLiNER checkpoint and e2e reference; see docs/gliner.md"]
fn multi_sentence_documents_match_python() {
    let (model, r) = setup();
    println!(
        "| document | words | windows | max logit error | mode | window agree | annotator agree | max score diff |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for d in r["documents"].as_array().unwrap() {
        let s = sentences(d);
        let words: Vec<&str> = s.iter().flat_map(|x| x.split(' ')).collect();
        let windows = d["windows"].as_array().unwrap();
        let mut error = 0f64;
        for mode in [Decode::Flat, Decode::Nested] {
            let schema = schema(&r, mode);
            let name = if mode == Decode::Flat { "flat" } else { "nested" };
            let (mut joined, mut windowed, mut matched, mut union, mut diff) = (Vec::new(), Vec::new(), 0, 0, 0f32);
            for w in windows {
                let (a, b) = (w["start"].as_u64().unwrap() as usize, w["end"].as_u64().unwrap() as usize);
                let p = model.pack(&words[a..b], &schema).unwrap();
                assert_eq!(serde_json::to_value(&p.ids).unwrap(), w["ids"], "{} window {a}..{b} ids", d["name"]);
                let scores = model.score_packed(&p, &schema).unwrap();
                let logits = w["logits"].as_array().unwrap();
                assert_eq!(scores.len(), logits.len());
                for (c, l) in scores.iter().zip(logits) {
                    error = error.max((f64::from(c.logit) - l.as_f64().unwrap()).abs());
                }
                let native: Vec<Entity> = turbo_gliner::decode(scores, &schema)
                    .iter()
                    .map(|c| ((c.start + a, c.end + a, c.label), c.score, c.logit))
                    .collect();
                let expected = reference(&w["decoded"][name], a);
                let (m, s, u) = agree(&native, &expected);
                (matched, union, diff) = (matched + m, union + u, diff.max(s));
                windowed.extend(native);
                joined.extend(expected);
            }
            let annotated = annotate(&model, &schema, document(&s));
            let (m, s2, u) = agree(&annotated, &joined);
            println!(
                "| {} | {} | {} | {error:.2e} | {name} | {matched}/{union} | {m}/{u} | {:.1e} |",
                d["name"].as_str().unwrap(),
                words.len(),
                windows.len(),
                diff.max(s2)
            );
            assert!(error < 2e-3, "logit error {error}");
            assert_eq!((matched, m), (union, u), "{} {name}: entity disagreement", d["name"]);
            assert!(diff.max(s2) < 1e-4);
            // The annotator's windows score exactly as the standalone windows.
            assert_eq!(annotated, windowed);
            if windows.len() == 1 {
                let single: Vec<Entity> = model
                    .predict(&words, &schema)
                    .unwrap()
                    .iter()
                    .map(|c| ((c.start, c.end, c.label), c.score, c.logit))
                    .collect();
                assert_eq!(annotated, single, "single-window document differs from predict");
            }
        }
    }
}

fn median(mut f: impl FnMut(), reps: usize) -> f64 {
    f();
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[reps / 2]
}

#[test]
#[ignore = "requires converted GLiNER checkpoint and e2e reference; see docs/gliner.md"]
fn multi_sentence_timing() {
    let (model, r) = setup();
    let reps: usize = std::env::var("GLINER_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let root = std::env::var("GLINER_PACKAGE").unwrap();
    let (_, encoder) = turbo::deberta::package::PackageManifest::load(&Path::new(&root).join("encoder")).unwrap();
    let schema = schema(&r, Decode::Flat);
    for b in r["bench"].as_array().unwrap() {
        let s = sentences(b);
        let words: Vec<&str> = s.iter().flat_map(|x| x.split(' ')).collect();
        let ranges: Vec<(usize, usize)> = serde_json::from_value(b["ranges"].clone()).unwrap();
        let packed = median(|| drop(annotate(&model, &schema, document(&s))), reps);
        let sentence = median(
            || {
                let mut at = 0;
                for x in &s {
                    let n = x.split(' ').count();
                    drop(model.predict(&words[at..at + n], &schema).unwrap());
                    at += n;
                }
            },
            reps,
        );
        let ids: Vec<Vec<u32>> = ranges.iter().map(|&(a, z)| model.pack(&words[a..z], &schema).unwrap().ids).collect();
        let full = encoder.config().max_sequence;
        let encode = |pad: bool| {
            for x in &ids {
                let n = if pad { full } else { x.len() };
                let mut v = x.clone();
                v.resize(n, 0);
                let mask: Vec<bool> = (0..n).map(|i| i < x.len()).collect();
                drop(encoder.encode(&v, &mask).unwrap());
            }
        };
        let (tight, padded) = (median(|| encode(false), reps), median(|| encode(true), reps));
        println!(
            "timing {} words={} windows={} packed_ms={packed:.1} per_sentence_ms={sentence:.1} encoder_packed_ms={tight:.1} encoder_padded_ms={padded:.1} python_ms={}",
            b["name"].as_str().unwrap(),
            words.len(),
            ranges.len(),
            b["python_median_ms"].as_f64().map_or("-".into(), |v| format!("{v:.1}"))
        );
    }
}
