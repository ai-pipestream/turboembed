//! Actual typed-wrapper outputs captured on main2b89 before its framing migration,
//! re-captured when the shared layer ids moved to `turbo:` (the child step writes
//! `turbo:sentences`, and its configuration hash names what it writes).
use turbo_analysis::{Bound, ConditionalStep, Key, ViewId};
use turbo_analysis_layers::{QualityFlags, TextQuality, lexical_quality_gate_on, quality_gate_on};
use turbo_sentdetect::RuleSentenceDetector;

#[test]
fn incoming_typed_conditional_hashes_keep_main_framing_and_child_configuration() {
    let cases = [
        ("", 0x00000000, false, [5975575384875696200, 9450893687527538269, 10347297403397086635]),
        ("", 0x00000000, true, [16457913058011209726, 14771483159612901799, 11385399924628490181]),
        ("", 0x80000000, false, [11990372259166013694, 5140733398866726003, 213259536196001876]),
        ("", 0x80000000, true, [13959003527571638004, 9647514255026201165, 13193875931995130498]),
        ("gold", 0x00000000, false, [4280225159408264864, 11190671735203060810, 11767453591830570075]),
        ("gold", 0x00000000, true, [14971963725649055350, 7680559350845184824, 12911869275955675677]),
        ("gold", 0x80000000, false, [9767645273660559943, 10766850829224824213, 17550280085289327553]),
        ("gold", 0x80000000, true, [7155782263593664993, 4219588893628707719, 16239568063775790827]),
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
