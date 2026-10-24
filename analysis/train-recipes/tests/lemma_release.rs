//! Published recipe ids retain their original on-disk class format.
use turbo_formats::native::LemmaReader;
use turbo_lemmatizer::{LemmaEncoding, LemmatizerModel};
use turbo_maxent::ModelFile;
use turbo_train::TrainingConfiguration;
use turbo_train_components::{LemmatizerFactory, Stamp, train_lemmatizer_model};
use turbo_train_recipes::segmentation::TIMESTAMP_MS;
use turbo_train_recipes::ud::{UdRecipe, UdSamples};

#[test]
fn published_ud_lemma_recipe_keeps_opennlp_bytes() {
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
    let recipe = UdRecipe::trains("turbo-es-ud-ancora-2.16-lemmas-1").unwrap();
    let trained = recipe.train(&conllu).unwrap();
    let model = LemmatizerModel::open(ModelFile::open(trained.bytes.clone()).unwrap()).unwrap();
    assert_eq!(model.encoding(), LemmaEncoding::OpenNlp, "an existing release id changed its class format");
    let read = recipe.samples(&conllu, recipe.treebank.train.name).unwrap();
    let UdSamples::Lemmas(samples) = read.samples else { panic!("lemma recipe") };
    let expected = train_lemmatizer_model(
        &samples,
        &LemmatizerFactory::opennlp(recipe.treebank.language),
        &turbo_train_components::ud::training_parameters(),
        &mut TrainingConfiguration::default(),
        Stamp::at(TIMESTAMP_MS),
    )
    .unwrap();
    assert_eq!(trained.bytes, expected.package.to_bytes(), "published recipe bytes changed");
}
