//! Published recipe ids retain their original on-disk class format.
use turbo_formats::native::LemmaReader;
use turbo_lemmatizer::{LemmaEncoding, LemmatizerModel};
use turbo_maxent::ModelFile;
use turbo_train::TrainingConfiguration;
use turbo_train_components::{LemmatizerFactory, Stamp, train_lemmatizer_model};
use turbo_train_recipes::segmentation::TIMESTAMP_MS;
use turbo_train_recipes::ud::{UdRecipe, UdSamples};

fn training_text() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/inputs/tagged/lemmas.txt");
    let tagged = std::fs::read_to_string(path).unwrap();
    let mut conllu = String::new();
    for _ in 0..8 {
        for s in LemmaReader::new(&tagged).unwrap().map(Result::unwrap) {
            for (i, ((word, tag), lemma)) in s.tokens().iter().zip(s.tags()).zip(s.lemmas()).enumerate() {
                conllu.push_str(&format!("{}\t{}\t{}\t{}\t_\t_\t0\tdep\t_\t_\n", i + 1, word, lemma, tag));
            }
            conllu.push('\n');
        }
    }
    conllu
}

#[test]
fn published_ud_lemma_recipe_keeps_opennlp_bytes() {
    let conllu = training_text();
    let recipe = UdRecipe::trains("turbo-es-ud-ancora-2.16-lemmas-1").unwrap();
    let trained = recipe.train(&conllu).unwrap();
    let model = LemmatizerModel::open(ModelFile::open(trained.bytes.clone()).unwrap()).unwrap();
    assert_eq!(model.encoding(), LemmaEncoding::OpenNlp, "an existing release id changed its class format");
    let read = recipe.samples(&conllu, recipe.treebank.train.name).unwrap();
    let UdSamples::Lemmas(samples) = read.samples else { panic!("lemma recipe") };
    // The released UD 2.16 recipe retains four partitions. The shared
    // UD 1.3 reproduction parameters now use twelve, a separate protocol.
    let mut released = turbo_train_components::ud::training_parameters();
    released.put_str(turbo_train::Key::THREADS, "4");
    let expected = train_lemmatizer_model(
        &samples,
        &LemmatizerFactory::opennlp(recipe.treebank.language),
        &released,
        &mut TrainingConfiguration::default(),
        Stamp::at(TIMESTAMP_MS),
    )
    .unwrap();
    assert_eq!(trained.bytes, expected.package.to_bytes(), "published recipe bytes changed");
}

#[test]
fn the_explicit_native_route_keeps_case_without_changing_the_published_route() {
    use turbo_lemmatizer::{MaxentLemmatizer, TokenTag};
    use turbo_train_recipes::ud::native;

    let text = training_text();
    let recipe = UdRecipe::trains("turbo-es-ud-ancora-2.16-lemmas-1").unwrap();
    let before = recipe.train(&text).unwrap();
    let trained = native::train(&recipe, &text).unwrap();
    let model = LemmatizerModel::open(ModelFile::open(trained.bytes).unwrap()).unwrap();
    assert_eq!(model.encoding(), LemmaEncoding::CaseSensitiveV1);
    let finder = MaxentLemmatizer::new(model);
    let words = [TokenTag { token: "Smith", tag: "NNP" }, TokenTag { token: "Paris", tag: "NNP" }];
    let (mut classes, mut probabilities) = (Vec::new(), Vec::new());
    assert!(finder.classes(&words, &mut classes, &mut probabilities));
    for (word, class) in words.iter().zip(classes) {
        assert_eq!(LemmaEncoding::CaseSensitiveV1.decode(word.token, class).unwrap(), word.token);
    }
    assert_eq!(recipe.train(&text).unwrap().bytes, before.bytes);
    assert!(UdRecipe::trains(native::model_id(&recipe).unwrap()).is_none(), "a local id became a released recipe");
}

#[test]
fn native_training_refuses_non_lemma_or_modified_sources_before_reading_samples() {
    use turbo_train_recipes::ud::{UD_RECIPES, native};
    let pos = UdRecipe::trains("turbo-es-ud-ancora-2.16-pos-1").unwrap();
    assert!(matches!(native::train(&pos, "invalid CoNLL-U"), Err(turbo_train_recipes::Error::Model(_))));
    let mut modified = UdRecipe::trains("turbo-es-ud-ancora-2.16-lemmas-1").unwrap();
    modified.treebank = UD_RECIPES[7].treebank;
    assert!(matches!(native::train(&modified, "invalid CoNLL-U"), Err(turbo_train_recipes::Error::Model(_))));
}
