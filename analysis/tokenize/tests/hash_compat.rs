//! The comparison receipt's byte framing stays unchanged.
#[path = "../comparisons/output_hash.rs"]
mod output_hash;

#[test]
fn hash_compat_comparison_output() {
    let mut hashes = Vec::new();
    let mut hash = turbo_analysis::Fnv1a::new().finish();
    for tokens in [&["你好", "世界"][..], &["a", "bc"], &["ab", "c"], &["", "é\0"], &[]] {
        hash = output_hash::feed_line(hash, tokens);
        hashes.push(hash);
    }
    // Fixed values from the previous implementation, before migration.
    let expected = if usize::BITS == 64 {
        [2359106885265630378, 9996976480668105836, 8398308476612421478, 18002329342518815172, 17441645439690493761]
    } else {
        [17413943814891438986, 14501095583539585276, 15118451457296838070, 1836878892279729892, 1152549696194279393]
    };
    assert_eq!(hashes, expected);
}
