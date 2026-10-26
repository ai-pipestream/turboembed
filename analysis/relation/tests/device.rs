//! GLiNER2 relations and GLiREL with the encoder through the backend table
//! against the CPU encoder (#920): the reduced heads fixtures (width 32)
//! over the tiny DeBERTa-v2 bundle, every count equal and every logit
//! within BOUND, on the device the environment names (device_support).

mod device_support;

use device_support::*;
use turbo::safetensors::File;
use turbo_analysis::ModelRef;
use turbo_relation::encoder::DebertaEncoder;
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

#[test]
fn gliner2_relations_score_as_on_the_cpu_encoder() {
    let (cpu, on) = (gliner2_on(cpu().into()), gliner2_on(device()));
    let labels = ["founded", "located in", "works for"];
    let mut largest = 0f32;
    for text in TEXTS {
        let (want, got) = (cpu.pass(text, &labels).unwrap(), on.pass(text, &labels).unwrap());
        assert_eq!(want.prompt.ids, got.prompt.ids, "{text}: ids");
        for (l, (a, b)) in want.labels.iter().zip(&got.labels).enumerate() {
            assert_eq!(a.count, b.count, "{text}: label {l}: instance count");
            let d = worst(&a.logits, &b.logits, text);
            assert!(d <= BOUND, "{text}: label {l}: span logits {d}");
            largest = largest.max(d);
        }
        let (a, b) = (cpu.extract(text, &labels, 0.5).unwrap(), on.extract(text, &labels, 0.5).unwrap());
        let ends = |v: &[gliner2::Extracted]| {
            v.iter().map(|x| (x.label, x.head.start, x.head.end, x.tail.start, x.tail.end)).collect::<Vec<_>>()
        };
        assert_eq!(ends(&a), ends(&b), "{text}: extracted relations");
    }
    let x = on.execution().unwrap();
    println!(
        "gliner2 relations on {} {}: largest logit difference from the cpu encoder {largest:e}",
        x.backend, x.device
    );
}

#[test]
fn glirel_scores_as_on_the_cpu_encoder() {
    let (cpu, on) = (glirel_on(cpu().into()), glirel_on(device()));
    let labels = ["founded", "located in", "works for"];
    let mut largest = 0f32;
    for text in TEXTS {
        let (want, got) = (cpu.pass(text, &labels).unwrap(), on.pass(text, &labels).unwrap());
        let n = want.prompt().words.len();
        // Every single word an entity, every ordered pair of distinct ones.
        let entities: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        let pairs: Vec<(usize, usize)> =
            (0..n).flat_map(|a| (0..n).filter(move |&b| b != a).map(move |b| (a, b))).collect();
        let (a, b) = (cpu.logits(&want, &entities, &pairs).unwrap(), on.logits(&got, &entities, &pairs).unwrap());
        let d = worst(&a, &b, text);
        assert!(d <= BOUND, "{text}: relation logits {d}");
        largest = largest.max(d);
    }
    let x = on.execution().unwrap();
    println!("glirel on {} {}: largest logit difference from the cpu encoder {largest:e}", x.backend, x.device);
}

/// A bundle of another family is refused by name.
#[test]
fn a_bert_bundle_is_not_a_deberta_encoder() {
    let o = turbo_relation::encoder::EncoderOptions::default();
    let e = DebertaEncoder::open(&root().join("testdata/tiny-bert-bundle"), &o).err().unwrap();
    assert!(e.contains("not FAMILY_DEBERTA_V2"), "{e}");
}
