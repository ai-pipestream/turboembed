//! Actual typed-wrapper outputs captured on main2b89 before its framing migration.
use turbo_analysis::{Bound, ConditionalStep, Key, ViewId};
use turbo_analysis_layers::{QualityFlags, TextQuality, lexical_quality_gate_on, quality_gate_on};
use turbo_sentdetect::RuleSentenceDetector;

#[test]
fn incoming_typed_conditional_hashes_keep_main_framing_and_child_configuration() {
    let cases = [
        ("", 0x00000000, false, [1441871099724481884, 11131751841784537921, 9882539143749268343]),
        ("", 0x00000000, true, [6675568721773205062, 16768907480784507023, 17313157981643892077]),
        ("", 0x80000000, false, [8528938983591727674, 1412636249443905151, 11395687255528599144]),
        ("", 0x80000000, true, [2741135467072477708, 8328056926248822773, 17698497618143875466]),
        ("gold", 0x00000000, false, [7211152229605706228, 17009784371009589686, 3932581695335973623]),
        ("gold", 0x00000000, true, [13826135946996263142, 5912780041135964760, 10728310033059208381]),
        ("gold", 0x80000000, false, [12424924889804206595, 15735625034232460889, 17572392669807335685]),
        ("gold", 0x80000000, true, [12994352679228220721, 6733694057541533095, 14631350204148109371]),
    ];
    for (name, bits, custom, expected) in cases {
        let value = f32::from_bits(bits);
        let key = Key::<TextQuality>::named(name);
        let predicates = [
            quality_gate_on(key, ViewId::SOURCE, value, QualityFlags::CLEAN),
            lexical_quality_gate_on(key, ViewId::SOURCE, value, 72.0),
            lexical_quality_gate_on(key, ViewId::SOURCE, 0.5, value),
        ];
        let actual = predicates.map(|predicate| {
            let child = if custom {
                Bound::new(RuleSentenceDetector::new("de")).read(0, "custom-text").write(0, "custom-sentences")
            } else {
                Bound::new(RuleSentenceDetector::new("en"))
            };
            ConditionalStep::from_predicate("quality", predicate, child).config_hash()
        });
        assert_eq!(actual, expected, "name={name:?}, bits={bits:08x}, custom={custom}");
    }
}
