//! Randomized recall and top-1 optimality tests for `FuzzyAliasIndex` across
//! common-prefix and prefixless catalogs.

use turbo_entity_linker::{CatalogProvider, KbEntity, KbProvider, KnowledgeBase, alias_similarity, fold_alias};

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

#[test]
fn common_prefix_and_prefixless_fuzzy_top1_parity_against_brute_force() {
    // 1. Codex's exact 5,000-alias common-prefix reproduction (seed = 1u64, 200 trials)
    let mut seed = 1u64;
    let mut entities = Vec::with_capacity(5_000);
    let mut aliases = Vec::with_capacity(5_000);
    for i in 0..5_000 {
        let mut s = String::from("common prefix ");
        for _ in 0..6 {
            let ch = (b'0' + (lcg_next(&mut seed) % 10) as u8) as char;
            s.push(ch);
        }
        aliases.push(s.clone());
        entities.push(KbEntity::new(format!("id{i}"), s));
    }
    let kb = KnowledgeBase::from_entities("test", entities).unwrap();
    let provider = CatalogProvider::new(kb);
    let threshold = 0.89;
    for trial in 0..200 {
        let base_idx = (lcg_next(&mut seed) as usize) % aliases.len();
        let mut chars: Vec<char> = aliases[base_idx].chars().collect();
        for _ in 0..2 {
            let pos = 14 + ((lcg_next(&mut seed) as usize) % 6);
            chars[pos] = (b'0' + (lcg_next(&mut seed) % 10) as u8) as char;
        }
        let query: String = chars.into_iter().collect();
        if aliases.iter().any(|a| a == &query) {
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
        let got = provider.generate_candidates(&query, None, 32, threshold);
        if best_score >= threshold {
            let actual = got.first().map(|c| c.match_similarity).unwrap_or(0.0);
            assert!(actual + 1e-12 >= best_score, "trial {trial}: query={query:?} best={best_score} actual={actual}");
        }
    }

    // 2. Fable's 3 catalog shapes:
    // - 5,000 x "Common Prefix " + 6 digits (threshold = 0.89)
    // - 5,000 x "Common Prefix " + 6 of a..j (threshold = 0.89)
    // - 20,000 x 8 of a..p, no prefix (threshold = 0.85)
    for (prefix, alphabet_len, base_char, suffix_len, count, thresh, trials) in [
        ("Common Prefix ", 10u64, b'0', 6usize, 5_000usize, 0.89, 300usize),
        ("Common Prefix ", 10u64, b'a', 6usize, 5_000usize, 0.89, 300usize),
        ("", 16u64, b'a', 8usize, 20_000usize, 0.85, 150usize),
    ] {
        let mut shape_seed = 1u64;
        let mut shape_entities = Vec::with_capacity(count);
        let mut folded_aliases = Vec::with_capacity(count);
        for i in 0..count {
            let mut s = String::from(prefix);
            for _ in 0..suffix_len {
                let ch = (base_char + (lcg_next(&mut shape_seed) % alphabet_len) as u8) as char;
                s.push(ch);
            }
            folded_aliases.push(fold_alias(&s));
            shape_entities.push(KbEntity::new(format!("E{i}"), s));
        }
        let shape_prov = CatalogProvider::new(KnowledgeBase::from_entities("shape", shape_entities).unwrap());
        let p_len = prefix.len();
        for trial in 0..trials {
            let base_idx = (lcg_next(&mut shape_seed) as usize) % folded_aliases.len();
            let mut chars: Vec<char> = folded_aliases[base_idx].chars().collect();
            for _ in 0..2 {
                let pos = p_len + ((lcg_next(&mut shape_seed) as usize) % suffix_len);
                chars[pos] = (base_char + (lcg_next(&mut shape_seed) % alphabet_len) as u8) as char;
            }
            let query: String = chars.into_iter().collect();
            if folded_aliases.iter().any(|a| a == &query) {
                continue;
            }
            let mut best = 0.0f64;
            for a in &folded_aliases {
                let sim = alias_similarity(&query, a);
                if sim >= thresh && sim > best {
                    best = sim;
                }
            }
            if best >= thresh {
                let got = shape_prov.generate_candidates(&query, None, 32, thresh);
                let actual = got.first().map(|c| c.match_similarity).unwrap_or(0.0);
                assert!(
                    actual + 1e-12 >= best,
                    "shape prefix={prefix:?} base={base_char} trial={trial}: query={query:?} best={best} actual={actual}"
                );
            }
        }
    }
}
