//! The UD recipes end to end on a treebank written for the tests: the
//! samples, training, scoring on the `test` file, the gate, the tokens
//! left out of a tagger's and a lemmatizer's gate, the data checks run
//! before training, the card, the model's directory and its zoo entry,
//! the release's zoo entries made from its models' directories;
//! the published reference models where they are present; and the
//! binary's refusals. The sentences are written for the project and are
//! not treebank text.

use std::path::{Path, PathBuf};

use turbo_train_components::lemmatize::lemma_class;
use turbo_train_recipes::card::TagCount;
use turbo_train_recipes::postag::{HeldOutTags, ReferenceTags};
use turbo_train_recipes::treebank::{Grant, TextRights, Treebank, TreebankFile};
use turbo_train_recipes::ud::{
    Component, ReferenceClasses, UdRecipe, lemma_samples, lemma_samples_sha256, pos_samples,
};
use turbo_train_recipes::{Error, LeftOut, Reference};
use turbo_zoo::{Sha256, Sha512};

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
    UdRecipe { model_id, treebank, component, reference: Some(reference) }
}

/// The OpenNLP Spanish model of `component`, which the test recipes name
/// as their reference although the test gives them other bytes.
fn spanish(component: Component) -> Reference {
    if component == Component::Sentence {
        return Reference::GoldSentences;
    }
    Reference::Zoo(leak(format!("opennlp-es-ud-gsd-{}-1.3-2.5.4", component.name())))
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

#[path = "ud/gates.rs"]
mod gates;
