//! The catalog's fuzzy index: the work one lookup reports, and how that
//! work grows with the number of aliases.

use super::*;

#[test]
fn fuzzy_lookup_work_reports_postings_and_candidates_for_catalog_queries() {
    let kb = KnowledgeBase::from_entities(
        "work-kb",
        vec![
            KbEntity::new("Q312", "Apple Incorporated").with_alias("Apple Computer", 1.0),
            KbEntity::new("Q76", "Barack Obama").with_alias("President Barack Obama", 1.0),
            KbEntity::new("E0", "Synthetic Entity 000001").with_alias("alias_item_000001", 1.0),
        ],
    )
    .unwrap();
    let prov = CatalogProvider::new(kb);
    let work = prov.fuzzy_lookup_work("Aple Incorporated", 0.88);
    assert!(work.postings_scanned > 0, "{work:?}");
    assert!(work.candidates_visited() > 0, "{work:?}");
    assert_eq!(work.index_probe_units(), work.postings_scanned + work.posting_candidates_verified);
    assert_eq!(prov.fuzzy_lookup_work("Aple Incorporated", 0.88), work);
    assert_eq!(prov.fuzzy_lookup_work("", 0.88), FuzzyLookupWork::default());
}

#[test]
fn fuzzy_index_sublinear_lookup_scales_from_10k_to_160k_aliases() {
    let build_synthetic_provider = |n: usize| -> CatalogProvider {
        let mut entities = Vec::with_capacity(n + 2);
        for i in 0..n {
            entities.push(
                KbEntity::new(format!("E{i}"), format!("Synthetic Entity {i:06}"))
                    .with_alias(format!("alias_item_{i:06}"), 1.0)
                    .with_types(["ORG"]),
            );
        }
        entities.push(
            KbEntity::new("Q312", "Apple Incorporated")
                .with_alias("Apple Computer", 1.0)
                .with_types(["ORG"])
                .with_prior(0.95),
        );
        entities.push(
            KbEntity::new("Q76", "Barack Obama")
                .with_alias("President Barack Obama", 1.0)
                .with_types(["PERSON"])
                .with_prior(0.95),
        );
        CatalogProvider::new(KnowledgeBase::from_entities("scale-kb", entities).unwrap())
    };

    let prov_10k = build_synthetic_provider(5_000); // 10,000+ folded aliases
    let prov_160k = build_synthetic_provider(80_000); // 160,000+ folded aliases

    let queries = ["Aple Incorporated", "Barak Obama", "Synthetic Entty 004200", "alias_itm_004200"];
    for q in queries {
        let c10 = prov_10k.generate_candidates(q, None, 5, 0.88);
        let c160 = prov_160k.generate_candidates(q, None, 5, 0.88);
        assert!(!c10.is_empty(), "expected fuzzy candidate for {q} in 10k catalog");
        assert!(!c160.is_empty(), "expected fuzzy candidate for {q} in 160k catalog");
        assert_eq!(c10[0].kb_id, c160[0].kb_id);
    }

    // Deterministic inverted-index probe work (postings scanned + posting-path
    // verifies), not wall-clock: a loaded CI runner made the old <5ms/query gate
    // flake at ~6.4ms/query while the 10k->160k ratio was only ~2x. Measured
    // index_probe_units on this catalog stay near 1x (posting budgets cap the
    // walk), so 16x aliases must stay under 4x probe work.
    let mut work_10k = FuzzyLookupWork::default();
    let mut work_160k = FuzzyLookupWork::default();
    for q in queries {
        let w10 = prov_10k.fuzzy_lookup_work(q, 0.88);
        let w160 = prov_160k.fuzzy_lookup_work(q, 0.88);
        assert!(w10.index_probe_units() > 0, "expected fuzzy index probe work for {q} in 10k catalog, got {w10:?}");
        assert!(w160.index_probe_units() > 0, "expected fuzzy index probe work for {q} in 160k catalog, got {w160:?}");
        // Counters are deterministic across identical lookups.
        assert_eq!(prov_10k.fuzzy_lookup_work(q, 0.88), w10);
        assert_eq!(prov_160k.fuzzy_lookup_work(q, 0.88), w160);
        work_10k.postings_scanned += w10.postings_scanned;
        work_10k.posting_candidates_verified += w10.posting_candidates_verified;
        work_10k.window_keys_considered += w10.window_keys_considered;
        work_10k.window_candidates_visited += w10.window_candidates_visited;
        work_160k.postings_scanned += w160.postings_scanned;
        work_160k.posting_candidates_verified += w160.posting_candidates_verified;
        work_160k.window_keys_considered += w160.window_keys_considered;
        work_160k.window_candidates_visited += w160.window_candidates_visited;
    }
    let units_10k = work_10k.index_probe_units();
    let units_160k = work_160k.index_probe_units();
    assert!(
        units_160k <= units_10k.saturating_mul(4),
        "expected sublinear fuzzy index probe work: 10k={work_10k:?} ({units_10k} units), 160k={work_160k:?} ({units_160k} units); 16x aliases should stay under 4x probe work"
    );

    // Short and transposed aliases ("abcd" vs "acbd", "nyx" vs "nxy", "meta" vs "mtea")
    // that share no 2-char prefix or padded trigram must still be retrieved by FuzzyAliasIndex.
    let short_kb = KnowledgeBase::from_entities(
        "short-kb",
        vec![
            KbEntity::new("one", "abcd"),
            KbEntity::new("two", "nyx"),
            KbEntity::new("three", "meta"),
            KbEntity::new("four", "ibm"),
        ],
    )
    .unwrap();
    let short_prov = CatalogProvider::new(short_kb);
    let c_acbd = short_prov.generate_candidates("acbd", None, 32, 0.88);
    assert_eq!(c_acbd.len(), 1);
    assert_eq!(c_acbd[0].kb_id, "one");
    assert!((c_acbd[0].match_similarity - 0.925).abs() < 1e-6);

    let c_mtea = short_prov.generate_candidates("mtea", None, 32, 0.88);
    assert_eq!(c_mtea.len(), 1);
    assert_eq!(c_mtea[0].kb_id, "three");

    // Common-prefix catalog with > 2048 shared-prefix aliases must still surface a late
    // matching suffix alias ("Common Prefix zzyzzz" for query "Common Prefix yyyyyy" at 0.89).
    let mut prefix_entities: Vec<KbEntity> =
        (0..2048).map(|i| KbEntity::new(format!("P{i}"), format!("Common Prefix {i:06}"))).collect();
    prefix_entities.push(KbEntity::new("P_TARGET", "Common Prefix zzyzzz"));
    let prefix_prov = CatalogProvider::new(KnowledgeBase::from_entities("prefix-kb", prefix_entities).unwrap());
    let c_prefix = prefix_prov.generate_candidates("Common Prefix yyyyyy", None, 32, 0.89);
    assert!(
        c_prefix.iter().any(|c| c.kb_id == "P_TARGET"),
        "expected P_TARGET ('Common Prefix zzyzzz') in candidates for 'Common Prefix yyyyyy', got {c_prefix:?}"
    );

    // Crowded shared-feature catalog (4096 distractors sharing all query tokens/grams/characters)
    // must preserve recall for the sole qualifying target at threshold 0.95.
    let mut crowded: Vec<KbEntity> = Vec::with_capacity(4097);
    for i in 0..2048 {
        crowded.push(KbEntity::new(format!("D1_{i}"), format!("Common Prefix {i:04}abcacbzz")));
        crowded.push(KbEntity::new(format!("D2_{i}"), format!("Common Prefix {i:04}bcabaczz")));
    }
    crowded.push(KbEntity::new("CROWDED_TARGET", "Common Prefix zzzzabcabczz"));
    let crowded_prov = CatalogProvider::new(KnowledgeBase::from_entities("crowded-kb", crowded).unwrap());
    let c_crowded = crowded_prov.generate_candidates("Common Prefix abcabc", None, 32, 0.95);
    assert_eq!(c_crowded.len(), 1);
    assert_eq!(c_crowded[0].kb_id, "CROWDED_TARGET");
}
