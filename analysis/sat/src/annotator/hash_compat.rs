use super::{SatOptions, config_hash};

#[test]
fn hash_compat_sat_raw_float_bits_and_window_options() {
    let mut hashes = Vec::new();
    for threshold in [0.0, -0.0, 0.25, 1.0] {
        for split in [false, true] {
            for label in [0, 1, u32::MAX] {
                let options =
                    SatOptions { threshold: Some(threshold), split_on_newlines: split, label, ..Default::default() };
                hashes.push(config_hash(&options, threshold, 512, 64));
            }
        }
    }
    // Fixed values from the previous implementation, before migration.
    assert_eq!(
        hashes,
        [
            6802270094317379597,
            13721298820012647292,
            10223833740331357545,
            5888352548284102058,
            17416067896298385979,
            8647411677811064070,
            6802270094317379597,
            13721298820012647292,
            10223833740331357545,
            5888352548284102058,
            17416067896298385979,
            8647411677811064070,
            15906774145559853386,
            8987745419864585691,
            15786473537544356006,
            16820691691593130925,
            5292976343578847004,
            17084816655894924553,
            18192612984699275306,
            11273584259004007611,
            8878075275716949894,
            659786457023001229,
            7578815182718268924,
            10176981344021162473
        ]
    );
}
