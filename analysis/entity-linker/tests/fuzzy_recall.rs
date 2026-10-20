//! Randomized recall and top-1 optimality tests for `FuzzyAliasIndex` across
//! common-prefix and prefixless small-alphabet catalogs.

use std::collections::HashSet;
use turbo_entity_linker::{CatalogProvider, KbEntity, KbProvider, KnowledgeBase, alias_similarity, fold_alias};

fn lcg_u32(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

#[test]
fn codex_10k_common_prefix_7digit_and_fable_12char_recall_parity() {
    // 1. Codex's exact 10,000-alias "common prefix " + 7-digit reproduction (seed = 1u64, 600 trials, threshold = 0.90)
    let mut seed = 1u64;
    let mut entities = Vec::with_capacity(10_000);
    let mut aliases = Vec::with_capacity(10_000);
    let mut distinct = HashSet::with_capacity(10_000);
    for i in 0..10_000 {
        let mut s = String::from("common prefix ");
        for _ in 0..7 {
            let ch = (b'0' + (lcg_u32(&mut seed) % 10) as u8) as char;
            s.push(ch);
        }
        distinct.insert(s.clone());
        aliases.push(s.clone());
        entities.push(KbEntity::new(format!("id{i}"), s));
    }
    assert!(distinct.len() >= 9_900, "expected >= 9,900 distinct aliases, got {}", distinct.len());

    let kb = KnowledgeBase::from_entities("codex-10k", entities).unwrap();
    let provider = CatalogProvider::new(kb);
    let threshold = 0.90;

    // Explicit check on Codex's trial 451 query ("common prefix 1156390" -> "common prefix 6035191" @ 0.9714285714285714)
    let trial_451_got = provider.generate_candidates("common prefix 1156390", None, 32, threshold);
    assert!(!trial_451_got.is_empty());
    assert_eq!(trial_451_got[0].matched_alias, "common prefix 6035191");
    assert!((trial_451_got[0].match_similarity - 0.9714285714285714).abs() < 1e-12);

    for trial in 0..600 {
        let base_idx = (lcg_u32(&mut seed) as usize) % aliases.len();
        let mut chars: Vec<char> = aliases[base_idx].chars().collect();
        for _ in 0..2 {
            let pos = 14 + ((lcg_u32(&mut seed) as usize) % 7);
            chars[pos] = (b'0' + (lcg_u32(&mut seed) % 10) as u8) as char;
        }
        let query: String = chars.into_iter().collect();
        if distinct.contains(&query) {
            continue;
        }
        let qf = fold_alias(&query);
        let mut best_score = 0.0f64;
        for a in &aliases {
            let sim = alias_similarity(&qf, a);
            if sim >= threshold && sim > best_score {
                best_score = sim;
            }
        }
        if best_score >= threshold {
            let got = provider.generate_candidates(&query, None, 32, threshold);
            let actual = got.first().map(|c| c.match_similarity).unwrap_or(0.0);
            assert!(actual + 1e-12 >= best_score, "trial {trial}: query={query:?} best={best_score} actual={actual}");
        }
    }

    // 2. Fable's 40,000 x 12 of a..j catalog ("hbfjiefihhbe" -> "hfeijbbhihfe" @ 0.8875, threshold = 0.88)
    let mut fable_seed = 1u64;
    let mut fable_entities = Vec::with_capacity(40_000);
    let mut fable_aliases = Vec::with_capacity(40_000);
    let mut fable_distinct = HashSet::with_capacity(40_000);
    for i in 0..40_000 {
        let mut s = String::with_capacity(12);
        for _ in 0..12 {
            let ch = (b'a' + (lcg_u32(&mut fable_seed) % 10) as u8) as char;
            s.push(ch);
        }
        fable_distinct.insert(s.clone());
        fable_aliases.push(s.clone());
        fable_entities.push(KbEntity::new(format!("F{i}"), s));
    }
    if !fable_distinct.contains("hfeijbbhihfe") {
        fable_distinct.insert("hfeijbbhihfe".to_owned());
        fable_aliases.push("hfeijbbhihfe".to_owned());
        fable_entities.push(KbEntity::new("F_TARGET", "hfeijbbhihfe"));
    }
    assert!(fable_distinct.len() >= 39_500, "expected >= 39,500 distinct aliases, got {}", fable_distinct.len());

    let fable_prov = CatalogProvider::new(KnowledgeBase::from_entities("fable-40k", fable_entities).unwrap());
    let fable_got = fable_prov.generate_candidates("hbfjiefihhbe", None, 32, 0.88);
    let mut fable_best = 0.0f64;
    for a in &fable_aliases {
        if a != "hbfjiefihhbe" {
            let sim = alias_similarity("hbfjiefihhbe", a);
            if sim >= 0.88 && sim > fable_best {
                fable_best = sim;
            }
        }
    }
    assert!(fable_best >= 0.8875 - 1e-12);
    let fable_actual = fable_got.first().map(|c| c.match_similarity).unwrap_or(0.0);
    assert!(
        fable_actual + 1e-12 >= fable_best,
        "fable query 'hbfjiefihhbe': best={fable_best} actual={fable_actual} got={fable_got:?}"
    );

    for trial in 0..100 {
        let base_idx = (lcg_u32(&mut fable_seed) as usize) % fable_aliases.len();
        let mut chars: Vec<char> = fable_aliases[base_idx].chars().collect();
        for _ in 0..2 {
            let pos = (lcg_u32(&mut fable_seed) as usize) % 12;
            chars[pos] = (b'a' + (lcg_u32(&mut fable_seed) % 10) as u8) as char;
        }
        let query: String = chars.into_iter().collect();
        if fable_distinct.contains(&query) {
            continue;
        }
        let mut best = 0.0f64;
        for a in &fable_aliases {
            let sim = alias_similarity(&query, a);
            if sim >= 0.88 && sim > best {
                best = sim;
            }
        }
        if best >= 0.88 {
            let got = fable_prov.generate_candidates(&query, None, 32, 0.88);
            let actual = got.first().map(|c| c.match_similarity).unwrap_or(0.0);
            assert!(actual + 1e-12 >= best, "fable 40k trial {trial}: query={query:?} best={best} actual={actual}");
        }
    }

    // 3. 20,000 x 8 of a..p prefixless catalog (high entropy, >= 19,900 distinct aliases)
    let mut p_seed = 7u64;
    let mut p_entities = Vec::with_capacity(20_000);
    let mut p_aliases = Vec::with_capacity(20_000);
    let mut p_distinct = HashSet::with_capacity(20_000);
    for i in 0..20_000 {
        let mut s = String::with_capacity(8);
        for _ in 0..8 {
            let ch = (b'a' + (lcg_u32(&mut p_seed) % 16) as u8) as char;
            s.push(ch);
        }
        p_distinct.insert(s.clone());
        p_aliases.push(s.clone());
        p_entities.push(KbEntity::new(format!("P{i}"), s));
    }
    assert!(p_distinct.len() >= 19_900, "expected >= 19,900 distinct aliases, got {}", p_distinct.len());
    let p_prov = CatalogProvider::new(KnowledgeBase::from_entities("p-20k", p_entities).unwrap());
    for trial in 0..150 {
        let base_idx = (lcg_u32(&mut p_seed) as usize) % p_aliases.len();
        let mut chars: Vec<char> = p_aliases[base_idx].chars().collect();
        let pos = (lcg_u32(&mut p_seed) as usize) % 8;
        chars[pos] = (b'a' + (lcg_u32(&mut p_seed) % 16) as u8) as char;
        let query: String = chars.into_iter().collect();
        if p_distinct.contains(&query) {
            continue;
        }
        let mut best = 0.0f64;
        for a in &p_aliases {
            let sim = alias_similarity(&query, a);
            if sim >= 0.85 && sim > best {
                best = sim;
            }
        }
        if best >= 0.85 {
            let got = p_prov.generate_candidates(&query, None, 32, 0.85);
            let actual = got.first().map(|c| c.match_similarity).unwrap_or(0.0);
            assert!(
                actual + 1e-12 >= best,
                "prefixless 20k trial {trial}: query={query:?} best={best} actual={actual}"
            );
        }
    }
}
