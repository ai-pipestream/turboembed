//! Pins captured by executing the public API on main2b89 before migration.
//! This boundary retains raw float bits, including signed zeros and NaN payloads.
use turbo_analysis::{DocPredicate, Key, ViewId};
use turbo_analysis_layers::{
    QualityFlags, TextQuality, lexical_quality_gate, lexical_quality_gate_on, quality_gate, quality_gate_on,
};

#[test]
fn incoming_quality_predicate_hashes_retain_raw_float_bits_and_named_views() {
    let cases: &[(u32, [u64; 3])] = &[
        (0x00000000, [9365553255702249823, 5961817133653184874, 12252030975086979081]),
        (0x80000000, [3064571805700034527, 17862696527573808874, 12252171712575390089]),
        (0x3f000000, [10111077648691526526, 12144962731776241523, 12252058462877684356]),
        (0x3f800000, [15529498949144123390, 6971424661110356979, 12129616847984328196]),
        (0x7f800000, [12379008224143015742, 10121774648623053619, 12129546479240122692]),
        (0xff800000, [233245600435679422, 16177872868838556595, 12129405741751711684]),
        (0x7fc01234, [8876558556731288492, 9877170400676877869, 10490047721011038198]),
        (0x7fc05678, [10640781688288859708, 13269351326759037581, 6685448900168014998]),
    ];
    for &(bits, expected) in cases {
        let value = f32::from_bits(bits);
        let actual = [
            quality_gate(ViewId::SOURCE, value, QualityFlags::CLEAN).config_hash(),
            lexical_quality_gate(ViewId::SOURCE, value, 72.0).config_hash(),
            lexical_quality_gate(ViewId::SOURCE, 0.5, value).config_hash(),
        ];
        assert_eq!(actual, expected, "raw bits {bits:08x}");
    }
    for (name, view, flags, score, lexical) in [
        ("", 0, 127, 7704754869847381361, 12144962731776241523),
        ("gold", 1, 5, 5719124760864066792, 2880950894357356704),
        ("é😀", 65535, 65535, 17984819751977408923, 4656550772938158324),
    ] {
        let key = Key::<TextQuality>::named(name);
        let predicate = quality_gate_on(key, ViewId(view), 0.5, QualityFlags(flags));
        assert_eq!(predicate.config_hash(), score);
        assert_eq!(predicate.reads(), vec![key.id()]);
        assert_eq!(lexical_quality_gate_on(key, ViewId(view), 0.5, 72.0).config_hash(), lexical);
    }
}
