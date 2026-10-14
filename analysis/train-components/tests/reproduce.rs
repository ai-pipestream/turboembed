//! Trains the eight published UD 1.3 sentence and token models again and
//! compares them with the published files. Ignored by default: it fetches
//! the treebanks (CC BY-SA 4.0, about 95 MB) through the zoo, which needs
//! `turbo-zoo accept CC-BY-SA-4.0 --yes` under the zoo home first, and
//! the published models under `TURBO_GOLDENS_MODELS` (`cargo run -p
//! turbo-goldens-tool -- fetch-models <dir>`). Training the five token
//! models takes some minutes each.
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
//! The test passes when every model agrees in its events, manifest, zip
//! entries, predicates and patterns, and in its parameters to within
//! [`TOLERANCE`]: the published files' parameters differ from this
//! port's in their last bits for a reason not yet known; the fork's
//! trainer in the goldens container writes this port's parameters bit for
//! bit (`turbo_train_components::ud` says what was checked).
//! The report printed with `--nocapture` says which models match byte for
//! byte and how far the others are.

use std::path::PathBuf;

use turbo_train::TrainingConfiguration;
use turbo_train_components::ud::{self, RECIPES};
use turbo_zoo::UserId;

/// The largest difference a parameter may show from the published value.
/// The eight models differ by 2.2e-15 to 9.8e-15 at most, and a wrong
/// iteration count or event stream moves parameters by far more.
const TOLERANCE: f64 = 1e-12;

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
    for recipe in &RECIPES {
        let id = recipe.model_id();
        if !only.is_empty() && !only.iter().any(|w| id.contains(w.as_str())) {
            continue;
        }
        let treebank = ud::fetch_treebank(&recipe.treebank, &home, &user).unwrap_or_else(|e| panic!("{id}: {e}"));
        let published = models.join(recipe.model_file());
        let started = std::time::Instant::now();
        let (trained, comparison) = recipe
            .reproduce(&treebank, &published, &mut TrainingConfiguration::default())
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        if let Ok(dir) = std::env::var("TURBO_REPRODUCE_OUT") {
            std::fs::write(PathBuf::from(dir).join(recipe.model_file()), trained.package.to_bytes()).unwrap();
        }
        println!("{}  ({:.0} s)", comparison.report(), started.elapsed().as_secs_f64());
        exact += usize::from(comparison.reproduced());
        if !comparison.agrees_within(TOLERANCE) {
            failed.push(id);
        }
    }
    println!("{exact} models match byte for byte");
    assert!(failed.is_empty(), "not reproduced: {failed:?}");
}
