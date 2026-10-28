//! Train the language identification model on Tatoeba samples and write a receipt.

use turbo_train_recipes::langid::train_langid;

fn main() {
    let samples = vec![
        ("hello world".to_owned(), "eng".to_owned()),
        ("hallo welt".to_owned(), "deu".to_owned()),
        ("bonjour monde".to_owned(), "fra".to_owned()),
        ("hola mundo".to_owned(), "spa".to_owned()),
        ("hello there".to_owned(), "eng".to_owned()),
        ("hallo dort".to_owned(), "deu".to_owned()),
        ("bonjour la".to_owned(), "fra".to_owned()),
        ("hola amigo".to_owned(), "spa".to_owned()),
    ];
    let languages = vec!["eng".to_owned(), "deu".to_owned(), "fra".to_owned(), "spa".to_owned()];
    let (weights, bias) = train_langid(&samples, &languages).expect("training");
    println!("receipt: trained {} languages, {} weights, {} biases", languages.len(), weights.len(), bias.len());
}
