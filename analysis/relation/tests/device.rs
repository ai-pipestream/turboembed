//! GLiNER2 relations and GLiREL with the encoder through the backend table
//! against the CPU encoder (#920): the reduced heads fixtures (width 32)
//! over the tiny DeBERTa-v2 bundle, every count equal and every logit
//! within BOUND, on the device the environment names (device_support).

mod device_support;

use device_support::*;
use turbo::safetensors::File;
use turbo_analysis::ModelRef;
use turbo_relation::encoder::{DebertaEncoder, EncoderStates};
use turbo_relation::{gliner2, glirel};

fn model() -> ModelRef {
    ModelRef { id: "turboembed/tiny-deberta".into(), revision: "1".into(), sha256: String::new() }
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/fixtures/{name}/reduced.safetensors", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn gliner2_on(encoder: DebertaEncoder) -> gliner2::Gliner2Relations {
    let heads = gliner2::heads::Heads::load(&File::parse("reduced", &fixture("gliner2")).unwrap(), 32, 8).unwrap();
    let tokenizer = turbo::sentencepiece::SentencePiece::parse(&spm()).unwrap();
    let specials = gliner2::prompt::Specials { sep_struct: 30000, sep_text: 30001, p: 30002, r: 30003 };
    gliner2::Gliner2Relations::new(encoder, heads, tokenizer, specials, model()).unwrap()
}

fn glirel_on(encoder: DebertaEncoder) -> glirel::GlirelRelations {
    let heads = glirel::heads::Heads::load(&File::parse("reduced", &fixture("glirel")).unwrap(), 32).unwrap();
    let tokenizer = turbo::sentencepiece::SentencePiece::parse(&spm()).unwrap();
    let specials = glirel::prompt::Specials { cls: 101, sep: 102, rel: 30000 };
    glirel::GlirelRelations::new(encoder, heads, tokenizer, specials, model(), 512).unwrap()
}

/// Times each tiny-bundle test runs its passes, so a receipt counts runs:
/// DEVICE_REPS, default 20.
fn reps() -> usize {
    std::env::var("DEVICE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20)
}

#[test]
fn gliner2_relations_score_as_on_the_cpu_encoder() {
    let (cpu, on) = (gliner2_on(cpu()), gliner2_on(device()));
    let labels = ["founded", "located in", "works for"];
    let (mut largest, mut held, mut found) = (0f32, Vec::new(), 0);
    for _ in 0..reps() {
        for text in TEXTS {
            let (want, got) = (cpu.pass(text, &labels).unwrap(), on.pass(text, &labels).unwrap());
            assert_eq!(want.prompt.ids, got.prompt.ids, "{text}: ids");
            let size = want.prompt.words.len() * want.width;
            for (l, (a, b)) in want.labels.iter().zip(&got.labels).enumerate() {
                assert_eq!(a.count, b.count, "{text}: label {l}: instance count");
                let d = worst(&a.logits, &b.logits, text);
                assert!(d <= BOUND, "{text}: label {l}: span logits {d}");
                largest = largest.max(d);
            }
            // Each query's best span: the decisions the heads make.
            let rows = |p: &gliner2::Pass| -> Vec<Vec<f32>> {
                p.labels.iter().flat_map(|l| l.logits.chunks(size).map(<[f32]>::to_vec)).collect()
            };
            let (a, b) = (rows(&want), rows(&got));
            let (a, b): (Vec<&[f32]>, Vec<&[f32]>) =
                (a.iter().map(Vec::as_slice).collect(), b.iter().map(Vec::as_slice).collect());
            if a.len() > 1 {
                held.extend(same_decisions(&a, &b, BOUND, text));
            }
            // Extraction at a threshold half of the spans pass, none at it.
            let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
            let t = midpoint(a.iter().flat_map(|r| r.iter().map(|&x| sig(x))).collect());
            let (x, y) = (cpu.extract(text, &labels, t).unwrap(), on.extract(text, &labels, t).unwrap());
            let ends = |v: &[gliner2::Extracted]| {
                v.iter().map(|x| (x.label, x.head.start, x.head.end, x.tail.start, x.tail.end)).collect::<Vec<_>>()
            };
            assert_eq!(ends(&x), ends(&y), "{text}: extracted relations at {t}");
            found += x.len();
        }
    }
    assert!(found > 0, "no relation extracted on either side");
    varied(&held, "gliner2 best spans");
    let held = held.len();
    let x = on.execution().unwrap();
    println!(
        "gliner2 relations on {} {}: {} runs, largest logit difference from the cpu encoder {largest:e}, {held} best \
         spans held, {found} relations extracted alike",
        x.backend,
        x.device,
        reps()
    );
}

#[test]
fn glirel_scores_as_on_the_cpu_encoder() {
    let (cpu, on) = (glirel_on(cpu()), glirel_on(device()));
    let labels = ["founded", "located in", "works for", "born in", "married to"];
    let (mut largest, mut held) = (0f32, Vec::new());
    for _ in 0..reps() {
        for text in TEXTS {
            let (want, got) = (cpu.pass(text, &labels).unwrap(), on.pass(text, &labels).unwrap());
            let n = want.prompt().words.len();
            if n < 2 {
                continue;
            }
            // Every single word an entity, every ordered pair of distinct ones.
            let entities: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
            let pairs: Vec<(usize, usize)> =
                (0..n).flat_map(|a| (0..n).filter(move |&b| b != a).map(move |b| (a, b))).collect();
            let (a, b) = (cpu.logits(&want, &entities, &pairs).unwrap(), on.logits(&got, &entities, &pairs).unwrap());
            let d = worst(&a, &b, text);
            assert!(d <= BOUND, "{text}: relation logits {d}");
            largest = largest.max(d);
            // Each label's best pair: which relation the scorer ranks first
            // for it. (Each pair's best label is the same label everywhere
            // over the tiny fixture, so that decision would vary nothing.)
            let m = labels.len();
            let column = |v: &[f32], l: usize| -> Vec<f32> { v.chunks(m).map(|r| r[l]).collect() };
            let (x, y): (Vec<Vec<f32>>, Vec<Vec<f32>>) =
                ((0..m).map(|l| column(&a, l)).collect(), (0..m).map(|l| column(&b, l)).collect());
            let (x, y): (Vec<&[f32]>, Vec<&[f32]>) =
                (x.iter().map(Vec::as_slice).collect(), y.iter().map(Vec::as_slice).collect());
            held.extend(same_decisions(&x, &y, BOUND, text));
        }
    }
    varied(&held, "glirel best pairs");
    let held = held.len();
    let x = on.execution().unwrap();
    println!(
        "glirel on {} {}: {} runs, largest logit difference from the cpu encoder {largest:e}, {held} best pairs held",
        x.backend,
        x.device,
        reps()
    );
}

/// A bundle of another family is refused by name.
#[test]
fn a_bert_bundle_is_not_a_deberta_encoder() {
    let o = turbo_relation::encoder::EncoderOptions::default();
    let e = DebertaEncoder::open(&root().join("testdata/tiny-bert-bundle"), &o).err().unwrap();
    assert!(e.contains("not FAMILY_DEBERTA_V2"), "{e}");
}

/// A package written from the bundle's own weights binds; one of other
/// weights is refused before any session is opened.
#[test]
fn a_bundle_is_bound_to_the_package_its_heads_were_converted_with() {
    let dir = std::env::temp_dir().join(format!("turbo-relation-bound-{}", std::process::id()));
    let (same, other) = (dir.join("same"), dir.join("other"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let weights = std::fs::read(bundle().join("weights/model.safetensors")).unwrap();
    let source = ("turboembed/tiny-deberta", "f040d8d433e742f26023553192e8a29aa066358a", "Apache-2.0");
    turbo_bundle::deberta::write_package(cpu().config().clone(), &weights, "", source, &same).unwrap();
    let o = turbo_relation::encoder::EncoderOptions::default();
    let bound = DebertaEncoder::open_bound(&same, &bundle(), &o);
    assert!(bound.is_ok_and(|e| e.execution().is_some()), "the package of the bundle's own weights binds");
    let reduced = fixture("gliner2");
    let shape = turbo::deberta::Config {
        vocabulary: 16,
        intermediate: 48,
        heads: 2,
        max_sequence: 32,
        ..cpu().config().clone()
    };
    let shape = turbo::deberta::Config { max_relative_positions: 32, ..shape };
    turbo_bundle::deberta::write_package(shape, &reduced, "encoder.", source, &other).unwrap();
    let e = DebertaEncoder::open_bound(&other, &bundle(), &o).err().expect("a package of other weights");
    assert!(e.contains("encoder bundle") || e.contains("shape"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A GLiREL package laid out as `turbo-bundle glirel-convert` writes one,
/// from the tiny fixtures: glirel.json, the reduced heads, and encoder/ the
/// DeBERTa package of the tiny bundle's own weights. The real device test's
/// paths (glirel_pair) open it on the CPU and the device, so a path the
/// licensed checkpoint alone would reach is traced here.
#[test]
fn a_glirel_package_opens_by_the_real_tests_paths() {
    let dir = std::env::temp_dir().join(format!("turbo-relation-glirel-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (package, spm_path) = (dir.join("package"), dir.join("spm.model"));
    std::fs::create_dir_all(&package).unwrap();
    let weights = std::fs::read(bundle().join("weights/model.safetensors")).unwrap();
    let source = ("turboembed/tiny-deberta", "f040d8d433e742f26023553192e8a29aa066358a", "Apache-2.0");
    turbo_bundle::deberta::write_package(cpu().config().clone(), &weights, "", source, &package.join("encoder"))
        .unwrap();
    let heads = fixture("glirel");
    std::fs::write(package.join("heads.safetensors"), &heads).unwrap();
    std::fs::write(&spm_path, spm()).unwrap();
    let digest = turbo::deberta::package::digest;
    let meta = serde_json::json!({
        "version": 1, "cls": 101, "sep": 102, "rel": 30000, "max_words": 512,
        "repository": "turboembed/tiny-glirel", "revision": "1", "sha256": digest(&heads), "license": "Apache-2.0",
        "heads_sha256": digest(&heads), "spm_sha256": digest(&spm()),
    });
    std::fs::write(package.join("glirel.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
    let [cpu, on] = glirel_pair(&package, &bundle(), &spm_path);
    assert!(cpu.execution().is_none() && on.execution().is_some(), "the CPU encoder and the backend table");
    let labels = ["founded", "works for"];
    let text = TEXTS[0];
    let (a, b) = (cpu.pass(text, &labels).unwrap(), on.pass(text, &labels).unwrap());
    let n = a.prompt().words.len();
    let pairs: Vec<(usize, usize)> = (1..n).map(|i| (0, i)).collect();
    let entities: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    let (x, y) = (cpu.logits(&a, &entities, &pairs).unwrap(), on.logits(&b, &entities, &pairs).unwrap());
    assert!(worst(&x, &y, text) <= BOUND);
    let _ = std::fs::remove_dir_all(&dir);
}
