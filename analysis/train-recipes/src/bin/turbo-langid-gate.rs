//! Train on Tatoeba and evaluate on FLORES-200.

use turbo_analysis::ModelRef;
use turbo_langdetect::NgramModel;
use turbo_train_recipes::flores::{Example, evaluate};
use turbo_train_recipes::langid::train_langid;

fn main() {
    let langs = ["eng", "deu", "fra", "spa", "ita", "por", "nld", "rus", "pol", "tur"];
    let mut samples = Vec::new();
    for lang in langs {
        let path = format!("/tmp/tatoeba_{lang}.txt");
        let text = std::fs::read_to_string(&path).expect("read sample");
        for line in text.lines().take(1000) {
            let mut fields = line.split('\t');
            let _id = fields.next();
            let lang_code = fields.next().expect("lang");
            let sentence = fields.next().expect("text");
            samples.push((sentence.to_owned(), lang_code.to_owned()));
        }
    }
    let languages: Vec<String> = langs.iter().map(|s| (*s).to_owned()).collect();
    let (weights, bias) = train_langid(&samples, &languages).expect("training");
    let model = NgramModel::new(
        languages,
        weights,
        bias,
        (1, 3),
        65536,
        ModelRef { id: "tatoeba".into(), revision: "1".into(), sha256: "0".into() },
    )
    .expect("model");

    let mut texts = Vec::new();
    for lang in langs {
        let path = format!("/tmp/flores200_dataset/dev/{lang}_Latn.dev");
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines().take(100) {
                texts.push((line.to_owned(), lang));
            }
        }
    }
    let examples: Vec<Example> = texts.iter().map(|(text, lang)| Example { text, language: lang }).collect();
    let accuracy = evaluate(&model, &examples);
    println!("FLORES-200 accuracy on {} examples: {:.3}", examples.len(), accuracy);
}
