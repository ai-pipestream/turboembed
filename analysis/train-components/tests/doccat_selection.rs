use turbo_doccat::{DoccatFactory, DocumentCategorizerME};
use turbo_formats::native::DocumentReader;
use turbo_train::{TrainingConfiguration, TrainingParameters};
use turbo_train_components::{Stamp, doccat::train_doccat};
#[test]
fn selected_models_keep_accuracy_and_reduce_vocabulary() {
    let corpus =
        "yes\tgood common noiseone\nyes\tgood common noisetwo\nno\tbad common noiseone\nno\tbad common noisetwo\n";
    let samples: Vec<_> = DocumentReader::new(corpus, Default::default()).unwrap().map(Result::unwrap).collect();
    let factory = DoccatFactory::new();
    let mut p = TrainingParameters::new();
    p.put("Iterations", "100");
    p.put("Cutoff", "0");
    let fit = |p: &TrainingParameters| {
        train_doccat("en", &samples, p, &factory, &mut TrainingConfiguration::default(), Stamp::at(0))
            .unwrap()
            .doccat_model(&factory)
            .unwrap()
    };
    let full = fit(&p);
    assert!(full.maxent().num_predicates() > 2);
    for metric in ["MI", "Chi2"] {
        p.put("FeatureSelection", metric);
        p.put("FeatureSelectionTopK", "2");
        let reduced = fit(&p);
        assert_eq!(reduced.maxent().num_predicates(), 2);
        let cat = DocumentCategorizerME::new(reduced);
        for (text, label) in [(vec!["good", "common", "unseen"], "yes"), (vec!["bad", "noiseone"], "no")] {
            let scores = cat.categorize(&text);
            assert_eq!(cat.best_category(&scores), label);
        }
    }
}
