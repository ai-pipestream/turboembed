//! Train the language identification model on a small sample and write a receipt.

use turbo_train_recipes::langid::train_langid;

fn main() {
    let samples = vec![
        ("hello world".to_owned(), "en".to_owned()),
        ("hallo welt".to_owned(), "de".to_owned()),
        ("bonjour monde".to_owned(), "fr".to_owned()),
        ("hello there".to_owned(), "en".to_owned()),
        ("hallo dort".to_owned(), "de".to_owned()),
        ("bonjour la".to_owned(), "fr".to_owned()),
    ];
    let languages = vec!["en".to_owned(), "de".to_owned(), "fr".to_owned()];
    let (weights, bias) = train_langid(&samples, &languages).expect("training");
    println!("trained {} languages, {} weights, {} biases", languages.len(), weights.len(), bias.len());
}
