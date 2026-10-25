use turbo_analysis::Annotator;
use turbo_entity_linker::{EntitySalienceAnnotator, SalienceConfig};

#[test]
fn salience_identity_preserves_incoming_configuration_bits() {
    let configs = [
        SalienceConfig::default(),
        SalienceConfig { position_weight: 0.0, ..Default::default() },
        SalienceConfig { position_weight: -0.0, ..Default::default() },
        SalienceConfig { pagerank_iters: 7, ..Default::default() },
        SalienceConfig { heading_bonus: 0.75, ..Default::default() },
    ];
    let actual = configs.map(|c| EntitySalienceAnnotator::with_config(c).config_hash());
    // Captured through the incoming main implementation before migration.
    assert_eq!(
        actual,
        [16861992082950094533, 2896449896212479735, 8301550944341204599, 5346274319369297324, 2525375676948804869]
    );
}
