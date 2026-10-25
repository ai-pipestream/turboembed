//! Values captured from the production methods before shared-hash migration.
use turbo_analysis::Annotator;
use turbo_numeric::{RegionAwareMoneyAnnotator, RegionAwareNumberAnnotator, RegionAwareQuantityAnnotator};
#[test]
fn region_settings_keep_previous_hash_values() {
    let mut money = Vec::new();
    let mut numbers = Vec::new();
    let mut quantities = Vec::new();
    for margin in [0.0, -0.0, 0.1, 1.0] {
        money.push(RegionAwareMoneyAnnotator::with_minimum_margin(margin).unwrap().config_hash());
        for scale in [false, true] {
            numbers
                .push(RegionAwareNumberAnnotator::with_minimum_margin(margin).unwrap().with_scale(scale).config_hash());
        }
        quantities.push(RegionAwareQuantityAnnotator::with_minimum_margin(margin).unwrap().config_hash());
        quantities.push(
            RegionAwareQuantityAnnotator::with_minimum_margin(margin)
                .unwrap()
                .with_units(["km", "kg", "m", "kg"])
                .unwrap()
                .config_hash(),
        );
    }

    assert_eq!(money, [14695981039346656037, 5472609002491880229, 3029522752875930507, 8008135592701469477]);
    assert_eq!(
        numbers,
        [
            590684067820433389,
            590682968308805178,
            9814056104675209197,
            9814055005163580986,
            12502317877931951511,
            12502316778420323300,
            11728085946307669997,
            11728084846796041786
        ]
    );
    assert_eq!(
        quantities,
        [
            2288806449059916006,
            16095992209660327552,
            11512178485914691814,
            6872620172805551744,
            1184511484488686636,
            7966384851304436250,
            17065116826462513382,
            4075884804208473728
        ]
    );
}
