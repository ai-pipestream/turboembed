//! Train the language identification model on Tatoeba samples and write a receipt.

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
    println!(
        "receipt: trained {} languages on {} samples, {} weights, {} biases",
        languages.len(),
        samples.len(),
        weights.len(),
        bias.len()
    );
}
