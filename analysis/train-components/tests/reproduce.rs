//! Trains the eighteen published UD 1.3 models again and compares them
//! with the published files. Ignored by default: it fetches the treebanks
//! (CC BY-SA 4.0, about 95 MB) through the zoo, which needs `turbo-zoo
//! accept CC-BY-SA-4.0 --yes` under the zoo home first, and the published
//! models under `TURBO_GOLDENS_MODELS` (`cargo run -p turbo-goldens-tool
//! -- fetch-models <dir>`); a model not there (the goldens use only the
//! English and German taggers and lemmatizers) is fetched through the zoo
//! too. A full release run takes tens of minutes; the duration depends on
//! the host and component.
//!
//! ```text
//! TURBO_GOLDENS_MODELS=<dir> cargo test -p turbo-train-components --test reproduce -- --ignored --nocapture
//! ```
//!
//! `TURBO_REPRODUCE_MODELS` limits the run to the models whose ids hold
//! one of its comma separated words (`en-ud`, `sentence`), and
//! `TURBO_REPRODUCE_OUT` names a directory the trained `.bin` files are
//! written to, for looking at a difference.
//!
//! The test requires exact event hashes, manifests, model-entry bytes,
//! and entry names. It uses 300 iterations and 12 training partitions;
//! the earlier last-bit residual came from using 4 partitions. ZIP
//! compression differs between runtimes, so archive bytes are not compared.
//! A filter selecting no recipe is an error.

use std::path::PathBuf;

use turbo_train::TrainingConfiguration;
use turbo_train_components::ud::{self, RECIPES};
use turbo_zoo::UserId;

#[test]
#[ignore = "fetches CC BY-SA treebanks and trains for minutes; see the file comment"]
fn the_published_ud_models_are_reproduced() {
    let models = PathBuf::from(std::env::var("TURBO_GOLDENS_MODELS").expect("TURBO_GOLDENS_MODELS"));
    let home = ud::zoo_home().expect("a zoo home");
    let user: UserId =
        std::env::var("USER").or_else(|_| std::env::var("USERNAME")).expect("USER").parse().expect("a user name");
    let only: Vec<String> =
        std::env::var("TURBO_REPRODUCE_MODELS").map(|v| v.split(',').map(str::to_owned).collect()).unwrap_or_default();
    let mut failed = Vec::new();
    let mut exact = 0;
    let mut selected = 0;
    for recipe in &RECIPES {
        let id = recipe.model_id();
        if !only.is_empty() && !only.iter().any(|w| id.contains(w.as_str())) {
            continue;
        }
        selected += 1;
        let treebank = ud::fetch_treebank(&recipe.treebank, &home, &user).unwrap_or_else(|e| panic!("{id}: {e}"));
        let mut published = models.join(recipe.model_file());
        if !published.is_file() {
            published = ud::fetch_published_model(recipe, &home, &user).unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let started = std::time::Instant::now();
        let (trained, comparison) = recipe
            .reproduce(&treebank, &published, &mut TrainingConfiguration::default())
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        if let Ok(dir) = std::env::var("TURBO_REPRODUCE_OUT") {
            std::fs::write(PathBuf::from(dir).join(recipe.model_file()), trained.package.to_bytes()).unwrap();
        }
        println!("{}  ({:.0} s)", comparison.report(), started.elapsed().as_secs_f64());
        exact += usize::from(comparison.reproduced());
        if !comparison.reproduced() {
            failed.push(id);
        }
    }
    assert!(selected > 0, "TURBO_REPRODUCE_MODELS selected no models");
    println!("{exact} of {selected} models match byte for byte");
    assert!(failed.is_empty(), "not reproduced: {failed:?}");
}
