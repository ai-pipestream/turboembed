use super::*;

fn index_of(keys: &[&str]) -> FuzzyAliasIndex {
    let owned: Vec<String> = keys.iter().map(|s| (*s).to_owned()).collect();
    FuzzyAliasIndex::build(&owned)
}

#[test]
fn lookup_work_is_zero_for_empty_query_or_empty_index() {
    let empty = FuzzyAliasIndex::build(&[]);
    assert_eq!(empty.for_each_candidate("apple", 0.88, |_, _| {}), FuzzyLookupWork::default());
    let idx = index_of(&["apple inc", "barack obama"]);
    assert_eq!(idx.for_each_candidate("", 0.88, |_, _| {}), FuzzyLookupWork::default());
}

#[test]
fn lookup_work_counts_postings_and_visits_for_a_near_match() {
    let idx = index_of(&["apple incorporated", "apple computer", "barack obama", "synthetic entity 000001"]);
    let mut visited = 0usize;
    let work = idx.for_each_candidate("aple incorporated", 0.88, |_, _| {
        visited += 1;
    });
    assert_eq!(work.candidates_visited(), visited);
    assert!(work.postings_scanned > 0, "expected posting scans, got {work:?}");
    assert!(work.index_probe_units() >= work.posting_candidates_verified);
    // A second identical lookup must report the same deterministic counters.
    let again = idx.for_each_candidate("aple incorporated", 0.88, |_, _| {});
    assert_eq!(again, work);
}

#[test]
fn token_dice_saturation_does_not_starve_the_slice_verify_budget() {
    // A query whose token-dice path visits more than BASE_FUZZY_VERIFY
    // candidates before the posting-slice loop: the slice admission
    // budget counts only slice verifies, so slices still run (#1626).
    let toks = ["ana", "bob", "cyd", "dan", "eve", "fay", "gil", "hal", "ivy", "jay"];
    let mut keys: Vec<String> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for rot in 0..toks.len() {
        for a in 0..toks.len() {
            for b in 0..toks.len() {
                let mut p: Vec<&str> = toks.to_vec();
                p.rotate_left(rot);
                p.swap(a, b);
                let key = p.join(" ");
                if seen.insert(key.clone()) {
                    keys.push(key);
                }
            }
        }
    }
    assert!(keys.len() > BASE_FUZZY_VERIFY, "{} permutation keys", keys.len());
    // Shares nine of the ten query tokens: below the token-dice cutoff,
    // reachable from the posting slices.
    let target = "ana bob cyd dan eve fay gil hal ivy jax".to_owned();
    let target_idx = keys.len() as u32;
    keys.push(target);
    let owned: Vec<String> = keys;
    let idx = FuzzyAliasIndex::build(&owned);
    let query = toks.join(" ");
    let mut visited: Vec<u32> = Vec::new();
    let work = idx.for_each_candidate(&query, 0.88, |k_idx, _| visited.push(k_idx));
    assert!(
        work.posting_candidates_verified > BASE_FUZZY_VERIFY,
        "token-dice path must saturate the old shared counter, got {work:?}"
    );
    assert!(work.postings_scanned > 0, "posting slices must still be admitted after dice saturation, got {work:?}");
    assert!(visited.contains(&target_idx), "slice-reachable target {target_idx} not visited: {work:?}");
    assert_eq!(work.candidates_visited(), visited.len());
}

#[test]
fn lookup_work_index_probe_units_add_postings_and_posting_verifies() {
    let w = FuzzyLookupWork {
        postings_scanned: 10,
        posting_candidates_verified: 3,
        window_keys_considered: 100,
        window_candidates_visited: 7,
    };
    assert_eq!(w.index_probe_units(), 13);
    assert_eq!(w.candidates_visited(), 10);
}
