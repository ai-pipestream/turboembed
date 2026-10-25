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
