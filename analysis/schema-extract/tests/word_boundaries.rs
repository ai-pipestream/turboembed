mod common;

use turbo_schema_extract::{Cardinality, EntityType, ExtractionSchema, Limits, window};

#[test]
fn cached_window_cost_matches_real_url_prompt_at_every_boundary() {
    let model = common::offline(Limits::default());
    let schema = ExtractionSchema {
        entities: vec![EntityType {
            name: "thing".into(),
            description: None,
            threshold: None,
            cardinality: Cardinality::Repeated,
            checks: Vec::new(),
        }],
        ..Default::default()
    };
    for text in
        ["https://a.b", "www.example.com", "ǅuro https://a.b", "https://a.b www.c.d", "https://a.b. x", "https://a.b "]
    {
        let full = model.prompt(text, &schema).unwrap();
        assert_eq!(full.text.as_ptr(), text.as_ptr(), "prompt text borrows the caller's bytes");
        assert_eq!(full.text.len(), text.len());
        let normalized = turbo_relation::gliner2::prompt::normalize(text);
        let reference = turbo_relation::gliner2::words::split(&normalized);
        assert_eq!(
            full.words.iter().map(|w| w.text.as_ref()).collect::<Vec<_>>(),
            reference.iter().map(|w| w.text.as_ref()).collect::<Vec<_>>()
        );
        let measure = full.measure(text.len());
        let actual = full.ids.len();
        let whole = 0..text.len();
        assert!(window::plan_measured(text, std::slice::from_ref(&whole), &measure, usize::MAX, actual).is_ok());
        assert!(window::plan_measured(text, std::slice::from_ref(&whole), &measure, usize::MAX, actual - 1).is_err());
        for a in 0..measure.words.len() {
            for b in a..measure.words.len() {
                let range = measure.words[a].0.start..measure.words[b].0.end;
                let actual = model.prompt(&text[range.clone()], &schema).unwrap().ids.len();
                let units = [range.clone()];
                assert_eq!(window::plan_measured(text, &units, &measure, usize::MAX, actual).unwrap(), vec![range]);
                assert!(
                    window::plan_measured(text, &units, &measure, usize::MAX, actual - 1).is_err(),
                    "{text} {a}..{b}"
                );
            }
        }
    }
}
