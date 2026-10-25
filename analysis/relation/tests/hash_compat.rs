use turbo_analysis::Annotator;
use turbo_relation::{ModalityAnnotator, ModalityConfig};

#[test]
fn merged_modality_hash_values_are_preserved() {
    let mut values = vec![ModalityAnnotator::default().config_hash()];
    for (max_window_tokens, subjects) in [
        (0, vec![]),
        (usize::MAX, vec!["é".into(), "you".into()]),
        (u32::MAX as usize, vec!["é".into(), "you".into()]),
        (8, vec!["ab".into(), "c".into()]),
        (8, vec!["a".into(), "bc".into()]),
    ] {
        values.push(
            ModalityAnnotator::with_config(ModalityConfig { max_window_tokens, deontic_subjects: subjects })
                .config_hash(),
        );
    }
    let widest = if usize::BITS == 64 { 11660522878699475096 } else { 5604952034185040972 };
    assert_eq!(
        values,
        [
            14509225540844368905,
            6688732525356290675,
            widest,
            5604952034185040972,
            12758862760275086247,
            17632532740573429875,
        ]
    );
}

#[test]
fn merged_event_hash_values_are_preserved() {
    use turbo_relation::{EventConfig, EventExtractor};
    let mut values = vec![EventExtractor::default().config_hash()];
    for anchor in [false, true] {
        for propagate_conj in [false, true] {
            values.push(
                EventExtractor::with_config(EventConfig {
                    nominal_triggers: vec!["ab".into(), "c".into()],
                    anchor,
                    propagate_conj,
                })
                .config_hash(),
            );
        }
    }
    for nominal_triggers in [vec!["a".into(), "bc".into()], vec!["é".into()], vec![]] {
        values.push(
            EventExtractor::with_config(EventConfig { nominal_triggers, anchor: true, propagate_conj: true })
                .config_hash(),
        );
    }
    assert_eq!(
        values,
        [
            13442099224136037103,
            15104243488633165598,
            17097080763888345955,
            6248665144614772709,
            4194607061912914272,
            18358868678729455144,
            1499377304164905690,
            5048337373682345938,
        ]
    );
}
