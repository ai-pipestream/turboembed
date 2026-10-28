use super::*;
use crate::card::MODEL_LICENCE;
use crate::fineweb::WebPlan;
use crate::langdetect::Plan;
use turbo_zoo::{Licence, LicenceClass, class_of};

#[test]
fn the_recipe_is_the_tatoeba_recipe_under_its_own_id() {
    let r = LANGDETECT_WEB;
    assert_eq!(r.tatoeba.model_id, MODEL_ID);
    assert_eq!(r.tatoeba.reference.id(), LANGDETECT.reference.id());
    assert_eq!(r.tatoeba.split, LANGDETECT.split);
    assert_eq!(r.tatoeba.parameters(), LANGDETECT.parameters());
    assert_eq!(r.tatoeba.languages, LANGDETECT.languages);
    assert_eq!(r.web, WEB);
    assert_eq!(r.compared.id(), COMPARED.id());
    let id: turbo_zoo::ArtifactId = MODEL_ID.parse().unwrap();
    assert_eq!(id.as_str(), MODEL_ID);
    assert_eq!(COMPARED.id(), langdetect::MODEL_ID);
    // The mappings over every language are the Tatoeba recipe's pairs
    // whose codes differ.
    let m: Vec<(String, String)> = r.web_mappings().into_iter().map(|m| (m.held_out, m.reference)).collect();
    assert_eq!(m.len(), 7);
    assert!(m.contains(&("est".to_owned(), "ekk".to_owned())));
    assert!(m.contains(&("zsm".to_owned(), "msa".to_owned())));
}

#[test]
fn the_training_data_is_permissive_with_attribution() {
    let data = training_data(&"0".repeat(64));
    assert_eq!(data.len(), 3);
    assert_eq!(data[1].license, Licence::OdcBy10);
    assert_eq!(data[1].name, "FineWeb-2 af9c13333eb9");
    assert_eq!(data[2].name, "FineWeb 9bb295ddab0e");
    let class = class_of(std::iter::once(MODEL_LICENCE).chain(data.iter().map(|t| t.license)));
    assert_eq!(class, LicenceClass::Permissive);
}

#[test]
fn train_and_score_reject_mismatched_web_and_tatoeba_languages() {
    let mut r = LANGDETECT_WEB;
    r.tatoeba.languages = &langdetect::LANGUAGES[..1];
    let mut plan = Plan {
        training: Vec::new(),
        held_out: Vec::new(),
        languages: Vec::new(),
        excluded: Vec::new(),
        records: 0,
        no_language: 0,
        other_languages: 0,
        cc0_records: 0,
        cc0_differing: 0,
        cc0_not_in_detailed: 0,
    };
    let web_plan = WebPlan { training: Vec::new(), held_out: Vec::new(), files: Vec::new() };
    assert!(matches!(r.train(&plan, &web_plan, &[]), Err(Error::Model(_))));
    assert!(matches!(r.score(&[], &[], &[], &mut plan, &web_plan, "0", "0"), Err(Error::Model(_))));
}
