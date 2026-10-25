//! Local case-preserving UD lemma recipes. These ids are not registered in
//! the published release index. Their four-partition protocol is fixed to
//! reproduce the native-v1 measurements independently of UD 1.3 reproduction.

use turbo_train::{Key, TrainingParameters};

use super::{UD_RECIPES, UdOutcome, UdRecipe};
use crate::card::ParametersCard;
use crate::error::{Error, Result};
use crate::treebank::TreebankText;
use crate::ud_card::UdModelCard;
use crate::{GateRun, Trained};

/// Iterations used by the recorded native-v1 comparison.
pub const ITERATIONS: i32 = 300;
/// Event cutoff used by the recorded native-v1 comparison.
pub const CUTOFF: i32 = 5;
/// Fixed GIS reduction partitions, independent of release reproduction settings.
pub const THREADS: i32 = 4;

pub(super) fn training_parameters() -> TrainingParameters {
    let mut p = TrainingParameters::new();
    for (key, value) in [
        (Key::ALGORITHM, "MAXENT".to_owned()),
        (Key::ITERATIONS, ITERATIONS.to_string()),
        (Key::CUTOFF, CUTOFF.to_string()),
        (Key::THREADS, THREADS.to_string()),
    ] {
        p.put_str(key, &value);
    }
    p
}

/// The distinct, unpublished output id for a canonical UD lemma recipe.
/// Refuses non-lemma or caller-modified source recipes.
pub fn model_id(source: &UdRecipe) -> Result<&'static str> {
    if !UD_RECIPES.contains(source) {
        return Err(Error::Model("native lemma training requires a canonical UD source recipe".into()));
    }
    match source.model_id {
        "turbo-es-ud-ancora-2.16-lemmas-1" => Ok("turbo-es-ud-ancora-2.16-lemmas-native-v1-local"),
        "turbo-ca-ud-ancora-2.16-lemmas-1" => Ok("turbo-ca-ud-ancora-2.16-lemmas-native-v1-local"),
        "turbo-pt-ud-porttinari-2.16-lemmas-1" => Ok("turbo-pt-ud-porttinari-2.16-lemmas-native-v1-local"),
        "turbo-fi-ud-ftb-2.16-lemmas-1" => Ok("turbo-fi-ud-ftb-2.16-lemmas-native-v1-local"),
        "turbo-it-ud-markit-2.16-lemmas-1" => Ok("turbo-it-ud-markit-2.16-lemmas-native-v1-local"),
        _ => Err(Error::Model("native lemma training requires a lemma source recipe".into())),
    }
}

/// Train native-v1 classes with the source recipe's sample reader and the
/// fixed 300-iteration, cutoff-5, four-partition protocol. No release is changed.
pub fn train(source: &UdRecipe, conllu: &str) -> Result<Trained> {
    model_id(source)?;
    source.train_with_lemma_format(conllu, true)
}

/// Train and run the existing held-out/reference gate with unchanged limits.
/// Reference bytes and the sample coverage checks run before training. The
/// caller must use `Treebank::read` to check corpus licence and file digests.
pub fn run(source: &UdRecipe, text: &TreebankText, reference: &[u8]) -> Result<UdOutcome> {
    let recipe = UdRecipe { model_id: model_id(source)?, ..*source };
    source.reference.check(reference)?;
    source.check_data(text, reference)?;
    let trained = train(source, &text.train)?;
    let GateRun { record: gate, held_out } = recipe.gate(&trained.bytes, reference, recipe.held_out(text)?)?;
    let mut card = UdModelCard::new(&recipe, &trained, held_out, &gate);
    card.training.parameters =
        ParametersCard { algorithm: "MAXENT".into(), iterations: ITERATIONS, cutoff: CUTOFF, threads: THREADS };
    Ok(UdOutcome { trained, gate, card })
}
