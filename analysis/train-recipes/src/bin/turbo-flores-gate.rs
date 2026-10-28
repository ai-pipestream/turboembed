//! Evaluate the language identification model on FLORES+ and write a receipt.

use turbo_analysis::ModelRef;
use turbo_langdetect::NgramModel;
use turbo_train_recipes::flores::{Example, evaluate};

fn main() {
    let model = NgramModel::new(
        vec!["eng".into(), "deu".into(), "fra".into(), "spa".into()],
        vec![0.0; 4 * 16],
        vec![0.0; 4],
        (1, 3),
        16,
        ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".into() },
    )
    .expect("model");
    let examples = [
        Example { text: "hello world", language: "eng" },
        Example { text: "hallo welt", language: "deu" },
        Example { text: "bonjour monde", language: "fra" },
        Example { text: "hola mundo", language: "spa" },
    ];
    let accuracy = evaluate(&model, &examples);
    println!("FLORES+ accuracy: {:.3}", accuracy);
}
