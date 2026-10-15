//! The UD recipes end to end on a treebank written for the tests: the
//! samples, training, scoring on the `test` file, the gate or the record
//! of a model scored alone, the tags left out of a tagger's gate, the
//! card, the model's directory and its zoo entry; the published reference
//! models where they are present; and the binary's refusals. The sentences
//! are written for the project and are not treebank text.

use std::path::{Path, PathBuf};

use turbo_eval::model_gate::{Measure, ModelGateRecord, Task, UngatedRecord};
use turbo_train_components::ud::Component;
use turbo_train_recipes::Error;
use turbo_train_recipes::card::{CARD_FILE, GATE_FILE, MODEL_VERSION, TagFormat};
use turbo_train_recipes::treebank::{Grant, Treebank, TreebankFile};
use turbo_train_recipes::ud::{
    MAX_TOKENS_LEFT_OUT, UD_RECIPES, UdGate, UdRecipe, UdReference, UdSamples, lemma_samples, pos_samples, run_ud,
    tags_left_out,
};
use turbo_train_recipes::ud_card::{EXPERIMENTAL_VERSION, UdGateCard, UdModelCard, write_ud_model_dir};
use turbo_zoo::{LicenceClass, Sha256, Sha512};

/// Sentences written for the tests, in CoNLL-U, with a multiword token
/// (`del`, `al`), an empty node (`1.1`), `SpaceAfter=No`, an abbreviation
/// and an interjection (`INTJ`, in the third sentence only).
const POOL: &[&str] = &[
    "# text = El gato del vecino duerme.
1\tEl\tel\tDET\t_\t_\t2\tdet\t_\t_
2\tgato\tgato\tNOUN\t_\t_\t6\tnsubj\t_\t_
3-4\tdel\t_\t_\t_\t_\t_\t_\t_\t_
3\tde\tde\tADP\t_\t_\t5\tcase\t_\t_
4\tel\tel\tDET\t_\t_\t5\tdet\t_\t_
5\tvecino\tvecino\tNOUN\t_\t_\t2\tnmod\t_\t_
6\tduerme\tdormir\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
7\t.\t.\tPUNCT\t_\t_\t6\tpunct\t_\t_",
    "# text = Vamos al mercado mañana.
1\tVamos\tir\tVERB\t_\t_\t0\troot\t_\t_
2-3\tal\t_\t_\t_\t_\t_\t_\t_\t_
2\ta\ta\tADP\t_\t_\t4\tcase\t_\t_
3\tel\tel\tDET\t_\t_\t4\tdet\t_\t_
4\tmercado\tmercado\tNOUN\t_\t_\t1\tobl\t_\t_
5\tmañana\tmañana\tADV\t_\t_\t1\tadvmod\t_\tSpaceAfter=No
6\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_",
    "# text = ¡Hola, amigos!
1\t¡\t¡\tPUNCT\t_\t_\t2\tpunct\t_\tSpaceAfter=No
2\tHola\thola\tINTJ\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t,\t,\tPUNCT\t_\t_\t4\tpunct\t_\t_
4\tamigos\tamigo\tNOUN\t_\t_\t2\tvocative\t_\tSpaceAfter=No
5\t!\t!\tPUNCT\t_\t_\t2\tpunct\t_\t_",
    "# text = Los niños comen manzanas rojas.
1\tLos\tel\tDET\t_\t_\t2\tdet\t_\t_
2\tniños\tniño\tNOUN\t_\t_\t3\tnsubj\t_\t_
3\tcomen\tcomer\tVERB\t_\t_\t0\troot\t_\t_
4\tmanzanas\tmanzana\tNOUN\t_\t_\t3\tobj\t_\t_
5\trojas\trojo\tADJ\t_\t_\t4\tamod\t_\tSpaceAfter=No
6\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_",
    "# text = El Sr. López llegó a las 10.30 de la noche.
1\tEl\tel\tDET\t_\t_\t2\tdet\t_\t_
2\tSr.\tseñor\tNOUN\t_\t_\t4\tnsubj\t_\t_
3\tLópez\tLópez\tPROPN\t_\t_\t2\tflat\t_\t_
4\tllegó\tllegar\tVERB\t_\t_\t0\troot\t_\t_
5\ta\ta\tADP\t_\t_\t7\tcase\t_\t_
6\tlas\tel\tDET\t_\t_\t7\tdet\t_\t_
7\t10.30\t10.30\tNUM\t_\t_\t4\tobl\t_\t_
8\tde\tde\tADP\t_\t_\t10\tcase\t_\t_
9\tla\tel\tDET\t_\t_\t10\tdet\t_\t_
10\tnoche\tnoche\tNOUN\t_\t_\t7\tnmod\t_\tSpaceAfter=No
11\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_",
    "# text = Ella dijo que no vendría, pero vino.
1\tElla\tél\tPRON\t_\t_\t2\tnsubj\t_\t_
2\tdijo\tdecir\tVERB\t_\t_\t0\troot\t_\t_
3\tque\tque\tSCONJ\t_\t_\t5\tmark\t_\t_
4\tno\tno\tADV\t_\t_\t5\tadvmod\t_\t_
5\tvendría\tvenir\tVERB\t_\t_\t2\tccomp\t_\tSpaceAfter=No
6\t,\t,\tPUNCT\t_\t_\t8\tpunct\t_\t_
7\tpero\tpero\tCCONJ\t_\t_\t8\tcc\t_\t_
8\tvino\tvenir\tVERB\t_\t_\t2\tconj\t_\tSpaceAfter=No
9\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_",
    "# text = Compramos pan y leche.
1\tCompramos\tcomprar\tVERB\t_\t_\t0\troot\t_\t_
1.1\t_\t_\tPRON\t_\t_\t_\t_\t1:nsubj\t_
2\tpan\tpan\tNOUN\t_\t_\t1\tobj\t_\t_
3\ty\ty\tCCONJ\t_\t_\t4\tcc\t_\t_
4\tleche\tleche\tNOUN\t_\t_\t2\tconj\t_\tSpaceAfter=No
5\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_",
    "# text = ¿Dónde está la estación?
1\t¿\t¿\tPUNCT\t_\t_\t3\tpunct\t_\tSpaceAfter=No
2\tDónde\tdónde\tADV\t_\t_\t3\tadvmod\t_\t_
3\testá\testar\tAUX\t_\t_\t0\troot\t_\t_
4\tla\tel\tDET\t_\t_\t5\tdet\t_\t_
5\testación\testación\tNOUN\t_\t_\t3\tnsubj\t_\tSpaceAfter=No
6\t?\t?\tPUNCT\t_\t_\t3\tpunct\t_\t_",
];

/// The indices of the pool sentences without the interjection.
const NO_INTJ: &[usize] = &[0, 1, 3, 4, 5, 6, 7];

/// Enough `test` sentences for the gate: 320, 32 sentence detector samples
/// of 10, more than `turbo_eval::model_gate::MIN_UNITS`.
const TEST_SENTENCES: usize = 320;

/// `n` sentences of `pool` (indices into [`POOL`]) in an order that shifts,
/// as one CoNLL-U text, each with a `sent_id`.
fn conllu(pool: &[usize], n: usize, name: &str) -> String {
    let mut out = String::new();
    for k in 0..n {
        let i = pool[(k * 3 + k / pool.len()) % pool.len()];
        out.push_str(&format!("# sent_id = {name}-{k}\n{}\n\n", POOL[i]));
    }
    out
}

/// UD's CC BY 4.0 statement, as the treebanks' `LICENSE.txt` has it.
const LICENCE: &str = "The treebank is licensed under the Creative Commons License Attribution 4.0 International.\n\n\
                       The complete license text is available at:\n\
                       https://creativecommons.org/licenses/by/4.0/legalcode\n";
const README: &str = "# Summary\n\nWritten for the tests.\n\n<pre>\nLicense: CC BY 4.0\n</pre>\n";

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-train-recipes-ud-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes a treebank of `train` and `test` under `dir` and records its
/// files as they are.
fn write_treebank(dir: &Path, train: &str, test: &str) -> Treebank {
    let file = |name: &'static str, text: &str| {
        std::fs::write(dir.join(name), text).unwrap();
        TreebankFile {
            name,
            size: text.len() as u64,
            sha256: leak(Sha256::of(text.as_bytes()).to_string()),
            sha512: leak(Sha512::of(text.as_bytes()).to_string()),
        }
    };
    Treebank {
        repository: "UD_Spanish-Test",
        language: "es",
        name: "test",
        grant: Grant::CcBy40,
        licence_file: file("LICENSE.txt", LICENCE),
        readme: file("README.md", README),
        train: file("es_test-ud-train.conllu", train),
        dev: file("es_test-ud-dev.conllu", &conllu(&[0, 1], 4, "dev")),
        test: file("es_test-ud-test.conllu", test),
        contributors: "the turboembed tests",
        citation: "Written for the turboembed tests.",
    }
}

fn recipe(treebank: Treebank, component: Component, reference: UdReference) -> UdRecipe {
    let model_id = leak(format!("turbo-es-ud-test-2.16-{}-1", component.name()));
    UdRecipe { model_id, treebank, component, reference }
}

/// The OpenNLP Spanish model of `component`, which the test recipes name
/// as their reference although the test gives them other bytes.
fn spanish(component: Component) -> UdReference {
    UdReference::Zoo(leak(format!("opennlp-es-ud-gsd-{}-1.3-2.5.4", component.name())))
}

const COMPONENTS: [Component; 4] = [Component::Sentence, Component::Tokens, Component::Pos, Component::Lemmas];

#[test]
fn the_tagger_s_samples_leave_out_empty_nodes_and_join_multiword_tokens() {
    let text = format!("{}\n\n{}\n\n# text = _\n1.1\t_\t_\tPRON\t_\t_\t_\t_\t_\t_\n\n", POOL[0], POOL[6]);
    let (samples, skipped) = pos_samples(&text).unwrap();
    assert_eq!(skipped, 1, "a sentence of an empty node alone has no word");
    let tokens = |i: usize| -> Vec<&str> { samples[i].tokens().iter().map(|t| t.as_str()).collect() };
    let tags = |i: usize| -> Vec<&str> { samples[i].tags().iter().map(|t| t.as_str()).collect() };
    assert_eq!(tokens(0), ["El", "gato", "del", "vecino", "duerme", "."]);
    assert_eq!(tags(0), ["DET", "NOUN", "ADP+DET", "NOUN", "VERB", "PUNCT"]);
    assert_eq!(tokens(1), ["Compramos", "pan", "y", "leche", "."], "the empty node is not a token");
    let (lemmas, skipped) = lemma_samples(&text).unwrap();
    assert_eq!(skipped, 1);
    let lemmas: Vec<&str> = lemmas[0].lemmas().iter().map(|l| l.as_str()).collect();
    assert_eq!(lemmas, ["el", "gato", "de+el", "vecino", "dormir", "."]);
}

#[test]
fn the_held_out_digest_covers_tags_and_lemmas() {
    let one = pos_samples(POOL[3]).unwrap().0;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&6u64.to_le_bytes());
    for s in ["Los", "DET", "niños", "NOUN", "comen", "VERB", "manzanas", "NOUN", "rojas", "ADJ", ".", "PUNCT"] {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
    }
    assert_eq!(UdSamples::Tags(one).sha256(), Sha256::of(&bytes).to_string());
    let tags = UdSamples::Tags(pos_samples(POOL[3]).unwrap().0).sha256();
    let lemmas = UdSamples::Lemmas(lemma_samples(POOL[3]).unwrap().0).sha256();
    assert_ne!(tags, lemmas, "the lemmas are in the digest");
}

#[test]
fn the_ud_recipes_train_score_gate_and_write_their_cards() {
    let dir = temp_dir("recipes");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let treebank = write_treebank(&dir, &conllu(&all, 400, "train"), &conllu(&all, TEST_SENTENCES, "test"));
    let text = treebank.read(&dir).unwrap();
    // The reference is the same recipe trained on two sentences only. The
    // candidate has seen every held-out sentence, so the scores say
    // nothing about the recipes; this checks the plumbing.
    let weak = conllu(&[0, 4], 100, "weak");
    let out = temp_dir("recipes-out");
    for component in COMPONENTS {
        let r = recipe(treebank, component, spanish(component));
        let weak_model = r.train(&weak).unwrap().bytes;
        // The weak tagger has no outcome for most held-out tags, so its gate
        // refuses to compare the two.
        if component == Component::Pos {
            assert!(matches!(
                run_ud(&r, &text, Some(&weak_model)),
                Err(Error::TagsOutsideReference { left_out, of, limit, .. })
                    if limit == MAX_TOKENS_LEFT_OUT && !limit.allows(left_out.iter().map(|t| t.tokens).sum(), of)
            ));
            continue;
        }
        let outcome = run_ud(&r, &text, Some(&weak_model)).unwrap();
        println!("{}: {} samples, {} events", r.model_id, outcome.trained.samples, outcome.trained.events);
        let UdGate::Gated { record, tags_left_out } = &outcome.gate else { panic!("{:?}", outcome.gate) };
        println!("{record}");
        record.check().unwrap();
        assert!(tags_left_out.is_empty());
        assert_eq!(record.task, r.task());
        let measure = if component == Component::Sentence || component == Component::Tokens {
            Measure::FMeasure
        } else {
            Measure::Accuracy
        };
        assert_eq!(record.measure, measure);
        assert_eq!(record.set.corpus_sha256, treebank.test.sha256);
        assert_eq!(record.set.name, "UD_Spanish-Test r2.16 test (es_test-ud-test.conllu)");
        let units = if component == Component::Sentence { TEST_SENTENCES / 10 } else { TEST_SENTENCES };
        assert_eq!(record.units.len(), units);
        assert!(record.result.pass, "the candidate has seen the held-out sentences: {record}");
        assert!(record.result.candidate.score >= record.result.reference.score, "{record}");

        let card = &outcome.card;
        assert_eq!(card.id, r.model_id);
        assert_eq!(card.component, r.zoo_component());
        assert_eq!(card.class, LicenceClass::Permissive);
        assert_eq!(card.trained_on, [treebank.training_data()]);
        assert_eq!(card.citation, "Written for the turboembed tests.");
        assert!(card.attribution.contains("UD_Spanish-Test r2.16"));
        assert_eq!(card.treebank.test.sha256, treebank.test.sha256);
        assert_eq!(card.treebank.dev.name, "es_test-ud-dev.conllu");
        assert_eq!(card.treebank.other_licence_offered, None);
        assert_eq!(card.treebank.held_out_samples, units);
        assert_eq!(card.treebank.held_out_skipped_sentences, 0);
        assert_eq!(card.training.parameters.iterations, 300);
        assert_eq!(card.training.sentences_per_sample.is_some(), component == Component::Sentence);
        assert_eq!(card.training.whitespace.is_some(), component != Component::Lemmas);
        assert_eq!(card.training.seed, None);
        let UdGateCard::Gated { reference, reference_sha256, pass, tag_format, .. } = &card.gate else {
            panic!("{:?}", card.gate)
        };
        assert_eq!(*reference, record.reference.id);
        assert_eq!(*reference_sha256, Sha256::of(&weak_model).to_string());
        assert!(*pass);
        assert_eq!(*tag_format, None);
        // The gate's reference is the test's model, not the zoo's file the
        // recipe names, so the card is not shippable.
        assert!(matches!(
            card.artifact(&r, "https://example.org/m.bin"),
            Err(Error::NotShippable { reason: "its gate's reference is not the recipe's", .. })
        ));

        write_ud_model_dir(&out, card, &outcome.trained.bytes, &outcome.gate).unwrap();
        let model_dir = out.join(r.model_id);
        assert_eq!(UdModelCard::read(&model_dir.join(CARD_FILE)).unwrap(), *card);
        assert_eq!(ModelGateRecord::read(&model_dir.join(GATE_FILE)).unwrap(), *record);
        assert!(write_ud_model_dir(&out, card, &outcome.trained.bytes, &outcome.gate).is_err(), "never replaced");
    }
    for d in [dir, out] {
        std::fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn a_tagger_s_gate_leaves_out_the_tags_the_reference_cannot_give() {
    let dir = temp_dir("left-out");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let treebank = write_treebank(&dir, &conllu(&all, 400, "train"), &conllu(&all, TEST_SENTENCES, "test"));
    let text = treebank.read(&dir).unwrap();
    let r = recipe(treebank, Component::Pos, spanish(Component::Pos));
    // A reference trained without the interjection has no INTJ outcome.
    let reference = r.train(&conllu(NO_INTJ, 200, "no-intj")).unwrap();
    let held_out = r.samples(&text.test, treebank.test.name).unwrap();
    let UdSamples::Tags(samples) = &held_out.samples else { panic!() };
    let (left_out, tokens) = tags_left_out(&reference.bytes, samples).unwrap();
    assert_eq!(left_out.len(), 1);
    assert_eq!(left_out[0].tag, "INTJ");
    assert_eq!(left_out[0].tokens, 40, "one in each of the 40 copies of the third sentence");
    let all_tokens: usize = samples.iter().map(|s| s.len()).sum();
    assert_eq!(tokens, all_tokens as u64);

    let outcome = run_ud(&r, &text, Some(&reference.bytes)).unwrap();
    let UdGate::Gated { record, tags_left_out } = &outcome.gate else { panic!() };
    assert_eq!(*tags_left_out, left_out);
    // Both models are counted on the same tokens: every one but the
    // interjections.
    let counted: u64 = record.units.iter().map(|u| u.reference.target).sum();
    assert_eq!(counted, tokens - 40);
    assert!(record.units.iter().all(|u| u.candidate.target == u.reference.target));
    let UdGateCard::Gated { tags_left_out: on_card, tag_format, .. } = &outcome.card.gate else { panic!() };
    assert_eq!(*on_card, left_out);
    assert_eq!(*tag_format, Some(TagFormat::Custom));
    assert!(outcome.gate.to_string().contains("INTJ (40)"), "{}", outcome.gate);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_model_without_a_reference_is_scored_alone_and_listed_as_experimental() {
    let dir = temp_dir("ungated");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let treebank = write_treebank(&dir, &conllu(&all, 200, "train"), &conllu(&all, 40, "test"));
    let text = treebank.read(&dir).unwrap();
    let r = recipe(treebank, Component::Lemmas, UdReference::NoReference("none for the tests"));
    let outcome = run_ud(&r, &text, None).unwrap();
    let UdGate::Ungated(record) = &outcome.gate else { panic!("{:?}", outcome.gate) };
    record.check().unwrap();
    assert_eq!(record.task, Task::Lemmatizer);
    assert_eq!(record.no_reference, "none for the tests");
    assert!(!outcome.gate.passed());
    assert!(outcome.gate.to_string().contains("UNGATED: "));
    let UdGateCard::Ungated { no_reference, candidate_score, .. } = &outcome.card.gate else { panic!() };
    assert_eq!(no_reference, "none for the tests");
    assert_eq!(*candidate_score, record.result.score);
    let artifact = outcome.card.artifact(&r, "https://example.org/m.bin").unwrap();
    assert_eq!(artifact.version, EXPERIMENTAL_VERSION);
    assert_ne!(artifact.version, MODEL_VERSION);

    let out = temp_dir("ungated-out");
    write_ud_model_dir(&out, &outcome.card, &outcome.trained.bytes, &outcome.gate).unwrap();
    assert_eq!(UngatedRecord::read(&out.join(r.model_id).join(GATE_FILE)).unwrap(), *record);

    // The card is the recipe's alone, and a reference is given only to a
    // recipe that has one.
    let gated = UdRecipe { reference: spanish(Component::Lemmas), ..r };
    assert!(matches!(outcome.card.artifact(&gated, "https://example.org/m.bin"), Err(Error::NotShippable { .. })));
    let other = UdRecipe { model_id: "turbo-es-ud-test-2.16-other-1", ..r };
    assert!(matches!(outcome.card.artifact(&other, "https://example.org/m.bin"), Err(Error::NotShippable { .. })));
    assert!(run_ud(&gated, &text, None).is_err());
    assert!(run_ud(&r, &text, Some(&outcome.trained.bytes)).is_err());
    for d in [dir, out] {
        std::fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn a_treebank_whose_files_or_licence_changed_is_refused() {
    let dir = temp_dir("changed");
    let treebank = write_treebank(&dir, &conllu(&[0], 3, "train"), &conllu(&[1], 3, "test"));
    treebank.read(&dir).unwrap();
    std::fs::write(dir.join("es_test-ud-test.conllu"), conllu(&[2], 3, "test")).unwrap();
    assert!(matches!(treebank.read(&dir), Err(Error::TreebankFile { file: "es_test-ud-test.conllu", .. })));
    std::fs::write(dir.join("LICENSE.txt"), LICENCE.replace("Attribution 4.0", "Attribution-ShareAlike 4.0")).unwrap();
    assert!(matches!(treebank.read(&dir), Err(Error::Licence { .. })));
    std::fs::remove_dir_all(dir).unwrap();
}

/// Whether CI asks for the published models to be present.
fn models_required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1")
}

/// The directory of the OpenNLP UD 1.3 models the UD recipes are gated
/// against. CI fetches only some of them, so they are read from
/// `TURBO_TEST_OPENNLP_MODELS`, as the other local-only references are:
/// without it the tests that need them are skipped, and with it a missing
/// file fails when the models are required.
fn references_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("TURBO_TEST_OPENNLP_MODELS").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("TURBO_TEST_OPENNLP_MODELS is not set: skipping the UD references");
    }
    dir
}

/// Each UD recipe's reference, where present: it matches the zoo's digest
/// and is scored on the test treebank's held-out samples of its component.
/// The text is the tests' Spanish for every language, so the scores say
/// nothing; this checks that every reference loads and is scored.
#[test]
fn the_published_references_are_checked_and_scored_when_present() {
    let Some(refs) = references_dir() else { return };
    let dir = temp_dir("published");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let treebank = write_treebank(&dir, &conllu(&all, 10, "train"), &conllu(&all, TEST_SENTENCES, "test"));
    let text = treebank.read(&dir).unwrap();
    for r in UD_RECIPES {
        let Some(reference) = r.reference_model() else {
            assert_eq!(r.treebank.language, "it");
            continue;
        };
        let path = refs.join(reference.file().unwrap());
        if !path.is_file() {
            assert!(!models_required(), "{} not found and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1", path.display());
            eprintln!("{} not found: skipping", path.display());
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        reference.check(&bytes).unwrap();
        let test = recipe(treebank, r.component, r.reference);
        let held_out = test.samples(&text.test, treebank.test.name).unwrap();
        let counts = test.score(&bytes, &held_out.samples).unwrap();
        assert_eq!(counts.len(), held_out.samples.len());
        let right: u64 = counts.iter().map(|c| c.true_positives).sum();
        let target: u64 = counts.iter().map(|c| c.target).sum();
        println!("{}: {right} of {target} on the test sentences", reference.id());
        assert!(right > 0, "{}", reference.id());
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_ud_command_checks_every_reference_and_treebank_before_training() {
    let out = temp_dir("binary");
    let bin = env!("CARGO_BIN_EXE_turbo-train-recipes");
    let run = |references: &Path, extra: &[&str]| {
        let mut c = std::process::Command::new(bin);
        c.arg("ud").arg("--treebanks").arg(out.join("treebanks")).arg("--references").arg(references);
        c.arg("--out").arg(out.join("models")).args(extra);
        c.output().unwrap()
    };
    let unknown = run(&out, &["--recipe", "turbo-en-ud-ewt-2.16-pos-1"]);
    assert_eq!(unknown.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("no recipe trains"), "{unknown:?}");
    // No reference in the directory: refused before any treebank is read.
    let no_reference = run(&out, &[]);
    assert_eq!(no_reference.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&no_reference.stderr).contains("opennlp-es-ud-gsd-sentence"), "{no_reference:?}");
    // A wrong file under a reference's name is refused by its digest.
    let refs = out.join("references");
    std::fs::create_dir_all(&refs).unwrap();
    std::fs::write(refs.join("opennlp-fi-ud-tdt-pos-1.3-2.5.4.bin"), "not the model").unwrap();
    let wrong = run(&refs, &["--recipe", "turbo-fi-ud-ftb-2.16-pos-1"]);
    assert_eq!(wrong.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&wrong.stderr).contains("SHA-256"), "{wrong:?}");
    // A recipe without a reference needs none, and the missing treebank is
    // refused before training.
    let no_treebank = run(&refs, &["--recipe", "turbo-it-ud-markit-2.16-pos-1"]);
    assert_eq!(no_treebank.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&no_treebank.stderr);
    assert!(stderr.contains("ud-italian-markit-r2.16") && stderr.contains("LICENSE.txt"), "{no_treebank:?}");
    assert!(!out.join("models").exists(), "nothing is written before every check passes");
    std::fs::remove_dir_all(out).unwrap();
}
