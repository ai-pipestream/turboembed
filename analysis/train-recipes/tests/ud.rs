//! The UD recipes end to end on a treebank written for the tests: the
//! samples, training, scoring on the `test` file, the gate, the tokens
//! left out of a tagger's and a lemmatizer's gate, the data checks run
//! before training, the card, the model's directory and its zoo entry,
//! the release's zoo entries made from its models' directories;
//! the published reference models where they are present; and the
//! binary's refusals. The sentences are written for the project and are
//! not treebank text.

use std::path::{Path, PathBuf};

use turbo_eval::model_gate::{Measure, ModelGateRecord, ModelId, Scoring, TagScoring};
use turbo_train_components::lemmatize::lemma_class;
use turbo_train_components::ud::Component;
use turbo_train_recipes::card::{CARD_FILE, GATE_FILE, MODEL_VERSION, TagCount, TagFormat, TokenCoverage};
use turbo_train_recipes::postag::{HeldOutTags, ReferenceTags};
use turbo_train_recipes::release::{self, UD_MODELS};
use turbo_train_recipes::treebank::{Grant, SPANISH_ANCORA, TextRights, Treebank, TreebankFile};
use turbo_train_recipes::ud::{
    IT_UD_VIT_POS, MAX_TOKENS_LEFT_OUT, ReferenceClasses, UD_RECIPES, UdRecipe, UdSamples, lemma_samples,
    lemma_samples_sha256, pos_samples, run_ud, score_lemmas,
};
use turbo_train_recipes::ud_card::{TreebankCard, UD_CARD_VERSION, UdModelCard, write_ud_model_dir};
use turbo_train_recipes::{Error, LeftOut, Reference};
use turbo_zoo::{Index, LicenceClass, Sha256, Sha512};

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
        text_rights: TextRights::Granted,
        copyright: None,
        licence_file: file("LICENSE.txt", LICENCE),
        readme: file("README.md", README),
        train: file("es_test-ud-train.conllu", train),
        dev: file("es_test-ud-dev.conllu", &conllu(&[0, 1], 4, "dev")),
        test: file("es_test-ud-test.conllu", test),
        contributors: "the turboembed tests",
        citation: "Written for the turboembed tests.",
    }
}

fn recipe(treebank: Treebank, component: Component, reference: Reference) -> UdRecipe {
    let model_id = leak(format!("turbo-es-ud-test-2.16-{}-1", component.name()));
    UdRecipe { model_id, treebank, component, reference }
}

/// The OpenNLP Spanish model of `component`, which the test recipes name
/// as their reference although the test gives them other bytes.
fn spanish(component: Component) -> Reference {
    Reference::Zoo(leak(format!("opennlp-es-ud-gsd-{}-1.3-2.5.4", component.name())))
}

const COMPONENTS: [Component; 4] = [Component::Sentence, Component::Tokens, Component::Pos, Component::Lemmas];

/// The sentences a weaker lemmatizer than the candidate is trained on,
/// from [`NO_INTJ`].
const WEAK_LEMMAS: usize = 28;

/// The tokens of held-out lemma samples.
fn held_out_tokens(held_out: &turbo_train_recipes::ud::ReadSamples<'_>) -> u64 {
    let UdSamples::Lemmas(samples) = &held_out.samples else { panic!("not lemma samples") };
    samples.iter().map(|s| s.len() as u64).sum()
}

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

fn put(bytes: &mut Vec<u8>, s: &str) {
    bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
    bytes.extend_from_slice(s.as_bytes());
}

/// A tagger's held-out digest is the MASC tagger's, with no mapping: each
/// token's text, tag and whether it is scored, then a count of 0
/// mappings. A lemmatizer's covers each token's text, tag and lemma and
/// whether it is scored.
#[test]
fn the_held_out_digests_cover_tags_what_is_scored_and_lemmas() {
    let tags = HeldOutTags::new(pos_samples(POOL[3]).unwrap().0, 0);
    let reference = ReferenceTags::of(&turbo_analysis::TagSet::new("ud", &["DET", "NOUN", "VERB", "PUNCT"]), &[]);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&6u64.to_le_bytes());
    let words = [("Los", "DET"), ("niños", "NOUN"), ("comen", "VERB"), ("manzanas", "NOUN"), ("rojas", "ADJ")];
    for (token, tag) in words.into_iter().chain([(".", "PUNCT")]) {
        put(&mut bytes, token);
        put(&mut bytes, tag);
        bytes.push(u8::from(tag != "ADJ"));
    }
    bytes.extend_from_slice(&0u64.to_le_bytes());
    assert_eq!(tags.sha256(&reference), Sha256::of(&bytes).to_string());

    let lemmas = lemma_samples(POOL[3]).unwrap().0;
    let lemma = ["el", "niño", "comer", "manzana", "rojo", "."];
    let tokens = words.into_iter().chain([(".", "PUNCT")]).zip(lemma);
    // A reference with every class of the sentence but that of "rojas" to
    // "rojo".
    let classes = ReferenceClasses::new(
        tokens.clone().filter(|&((token, _), _)| token != "rojas").map(|((token, _), lemma)| lemma_class(token, lemma)),
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&6u64.to_le_bytes());
    for ((token, tag), lemma) in tokens.clone() {
        put(&mut bytes, token);
        put(&mut bytes, tag);
        put(&mut bytes, lemma);
        bytes.push(u8::from(token != "rojas"));
    }
    assert_eq!(lemma_samples_sha256(&lemmas, &classes), Sha256::of(&bytes).to_string());
    assert_eq!(classes.left_out(&lemmas), [TagCount { tag: "ADJ".to_owned(), tokens: 1 }]);
    // "niños" to "niño" and "manzanas" to "manzana" are one class, and
    // "Los" to "el" and the identity of "." others.
    assert!(classes.scores("manzanas", "manzana") && classes.scores("Los", "el") && !classes.scores("rojas", "rojo"));
    let all = ReferenceClasses::new(tokens.map(|((token, _), lemma)| lemma_class(token, lemma)));
    assert_ne!(lemma_samples_sha256(&lemmas, &classes), lemma_samples_sha256(&lemmas, &all));
    assert!(all.left_out(&lemmas).is_empty());
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
        let mut weak_model = r.train(&weak).unwrap().bytes;
        // The weak tagger has no outcome for most held-out tags, and the
        // weak lemmatizer none for most held-out lemma classes, so the check
        // before training and the gate both refuse to compare the two.
        if component == Component::Pos || component == Component::Lemmas {
            let refused = |result: Result<(), Error>| match result {
                Err(Error::TagsOutsideReference { left_out, of, limit, .. }) if component == Component::Pos => {
                    limit == MAX_TOKENS_LEFT_OUT && !limit.allows(left_out.iter().map(|t| t.tokens).sum(), of)
                }
                Err(Error::ClassesOutsideReference { left_out, of, limit, .. }) if component == Component::Lemmas => {
                    limit == MAX_TOKENS_LEFT_OUT && !limit.allows(left_out.iter().map(|t| t.tokens).sum(), of)
                }
                _ => false,
            };
            assert!(refused(r.check_data(&text, &weak_model)));
            assert!(refused(run_ud(&r, &text, &weak_model).map(|_| ())));
            if component == Component::Pos {
                continue;
            }
            // A lemmatizer trained on few copies of the sentences without the
            // interjection has every held-out class but gets fewer lemmas
            // right.
            weak_model = r.train(&conllu(NO_INTJ, WEAK_LEMMAS, "no-intj")).unwrap().bytes;
        }
        r.check_data(&text, &weak_model).unwrap();
        let outcome = run_ud(&r, &text, &weak_model).unwrap();
        assert_eq!(outcome.card.training.parameters.threads, 4);
        let record = &outcome.gate;
        println!("{}\n{record}", outcome.card.summary());
        record.check().unwrap();
        assert_eq!(record.task, r.task());
        let segmentation = component == Component::Sentence || component == Component::Tokens;
        assert_eq!(record.measure, if segmentation { Measure::FMeasure } else { Measure::Accuracy });
        assert_eq!(record.set.corpus_sha256, treebank.test.sha256);
        assert_eq!(record.set.name, "UD_Spanish-Test r2.16 test (es_test-ud-test.conllu)");
        assert_eq!(record.set.skipped_sentences, 0);
        let units = if component == Component::Sentence { TEST_SENTENCES / 10 } else { TEST_SENTENCES };
        assert_eq!(record.units.len(), units);
        let held_out = r.held_out(&text).unwrap();
        match &held_out.samples {
            UdSamples::Segments(samples) => {
                assert_eq!(record.set.scoring, Scoring::Spans);
                assert_eq!(record.set.samples_sha256, samples.sha256());
            }
            UdSamples::Lemmas(samples) => {
                let tokens = samples.iter().map(|s| s.len() as u64).sum();
                let scoring = TagScoring { tokens, left_out: 0, reference_mappings: Vec::new() };
                assert_eq!(record.set.scoring, Scoring::Tags(scoring));
                let classes = ReferenceClasses::of_model(&weak_model).unwrap();
                assert_eq!(record.set.samples_sha256, lemma_samples_sha256(samples, &classes));
                // Each model's units are its own lemmas' counts, the
                // reference's from the reference's file.
                let units = |bytes: &[u8]| score_lemmas(bytes, samples, &classes).unwrap();
                assert_eq!(units(&outcome.trained.bytes), record.units.iter().map(|u| u.candidate).collect::<Vec<_>>());
                assert_eq!(units(&weak_model), record.units.iter().map(|u| u.reference).collect::<Vec<_>>());
                assert!(record.result.candidate.score > record.result.reference.score, "{record}");
            }
            UdSamples::Tags(_) => unreachable!(),
        }
        assert!(record.result.pass, "the candidate has seen the held-out sentences: {record}");
        assert!(record.result.candidate.score >= record.result.reference.score, "{record}");
        assert_eq!(
            r.score(&outcome.trained.bytes, &held_out).unwrap(),
            record.units.iter().map(|u| u.candidate).collect::<Vec<_>>()
        );
        assert_eq!(
            r.score(&weak_model, &held_out).unwrap(),
            record.units.iter().map(|u| u.reference).collect::<Vec<_>>()
        );

        let card = &outcome.card;
        assert_eq!(card.card_version, UD_CARD_VERSION);
        assert_eq!(card.id, r.model_id);
        assert_eq!(card.component, r.zoo_component());
        assert_eq!(card.class, LicenceClass::Permissive);
        assert_eq!(card.trained_on, [treebank.training_data()]);
        assert_eq!(card.citation, "Written for the turboembed tests.");
        assert!(card.attribution.contains("UD_Spanish-Test r2.16"));
        assert_eq!(card.treebank.test.sha256, treebank.test.sha256);
        assert_eq!(card.treebank.dev.name, "es_test-ud-dev.conllu");
        assert_eq!(card.treebank.other_licence_offered, None);
        assert_eq!((card.held_out.samples, card.held_out.skipped_sentences), (units, 0));
        match &card.held_out.tokens {
            None => assert!(segmentation),
            Some(TokenCoverage { tokens, reference_mappings, left_out, reference_mismatches }) => {
                assert_eq!(component, Component::Lemmas);
                assert_eq!(*tokens, held_out_tokens(&held_out));
                assert!(reference_mappings.is_empty() && left_out.is_empty());
                assert!(!reference_mismatches.is_empty(), "the reference errs: {reference_mismatches:?}");
            }
        }
        assert_eq!(card.training.parameters.iterations, 300);
        assert_eq!(card.training.sentences_per_sample.is_some(), component == Component::Sentence);
        assert_eq!(card.training.whitespace.is_some(), segmentation);
        assert_eq!(card.training.seed, None);
        assert_eq!(card.gate.reference, record.reference.id);
        assert_eq!(card.gate.reference_sha256, Sha256::of(&weak_model).to_string());
        assert!(card.gate.pass);
        assert_eq!((card.gate.candidate_tag_format, card.gate.reference_tag_format), (None, None));
        let summary = card.summary().to_string();
        assert!(
            summary.starts_with(&format!(
                "{}: trained on {} samples, 0 sentences skipped",
                r.model_id, outcome.trained.samples
            )),
            "{summary}"
        );
        let held_out_line = format!("held out {units} samples, 0 sentences skipped");
        if segmentation {
            assert!(summary.ends_with(&held_out_line), "{summary}");
        } else {
            let tokens = held_out_tokens(&held_out);
            let left_out = format!(
                "{held_out_line}\n{}: 0 of {tokens} held-out tokens left out, their lemmas not the reference's, \
                 by tag: none",
                r.model_id
            );
            assert!(summary.ends_with(&left_out), "{summary}");
        }
        // The test recipe is not one of the recipes, so its card is not
        // shippable.
        assert!(matches!(
            card.artifact("https://example.org/m.bin"),
            Err(Error::NotShippable { reason: "no recipe trains it", .. })
        ));

        write_ud_model_dir(&out, card, &outcome.trained.bytes, record).unwrap();
        let model_dir = out.join(r.model_id);
        assert_eq!(UdModelCard::read(&model_dir.join(CARD_FILE)).unwrap(), *card);
        assert_eq!(ModelGateRecord::read(&model_dir.join(GATE_FILE)).unwrap(), *record);
        assert!(write_ud_model_dir(&out, card, &outcome.trained.bytes, record).is_err(), "never replaced");
    }
    for d in [dir, out] {
        std::fs::remove_dir_all(d).unwrap();
    }
}

#[path = "ud/gates.rs"]
mod gates;
