//! The near-duplicate test of YAKE's deduplication pass, an interface in
//! the design so a caller can plug a tier above token overlap: Jaccard
//! over the token multiset (the default, which catches the real failure
//! mode) or Jaccard over the character trigrams of the phrase (which also
//! catches "Goldbloom" against "Goldblooms", where token overlap sees
//! nothing).

/// A similarity on folded keyphrases: `[0, 1]`, 1.0 identical.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum PhraseSimilarity {
    /// Jaccard over the token multiset: the size of the intersection over
    /// the size of the union, counting a term as many times as the rarer
    /// side holds it.
    #[default]
    TokenOverlap,
    /// Jaccard over the sets of character 3-grams of the two phrases. A
    /// phrase shorter than three characters is its own single gram.
    TrigramJaccard,
}

impl PhraseSimilarity {
    /// How similar the two phrases are, in `[0, 1]`.
    pub fn similarity(&self, a: &[&str], b: &[&str]) -> f64 {
        match self {
            PhraseSimilarity::TokenOverlap => multiset_jaccard(a, b),
            PhraseSimilarity::TrigramJaccard => trigram_jaccard(&a.join(" "), &b.join(" ")),
        }
    }
}

/// The multiset Jaccard of two token lists: sum over distinct terms of
/// min(count) over sum of max(count).
fn multiset_jaccard(a: &[&str], b: &[&str]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    // Count the shorter side, then walk the longer one against it, so the
    // scratch is the smaller map and every term is visited once per side.
    let mut counts: std::collections::HashMap<&str, (u32, u32)> = std::collections::HashMap::new();
    for &t in small {
        counts.entry(t).or_default().0 += 1;
    }
    for &t in large {
        counts.entry(t).or_default().1 += 1;
    }
    let (mut inter, mut union) = (0u64, 0u64);
    for (x, y) in counts.values() {
        inter += (*x).min(*y) as u64;
        union += (*x).max(*y) as u64;
    }
    if union == 0 { 0.0 } else { inter as f64 / union as f64 }
}

/// The Jaccard of the two phrases' character 3-gram sets. A phrase with
/// fewer than three characters contributes one gram, itself.
fn trigram_jaccard(a: &str, b: &str) -> f64 {
    let grams = |s: &str| -> std::collections::HashSet<String> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() < 3 {
            let mut set = std::collections::HashSet::new();
            if !s.is_empty() {
                set.insert(s.to_owned());
            }
            return set;
        }
        chars.windows(3).map(|w| w.iter().collect()).collect()
    };
    let (ga, gb) = (grams(a), grams(b));
    if ga.is_empty() && gb.is_empty() {
        return 1.0;
    }
    let inter = ga.intersection(&gb).count();
    let union = ga.union(&gb).count();
    if union == 0 { 0.0 } else { inter as f64 / union as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v<'a>(items: &'a [&'a str]) -> Vec<&'a str> {
        items.to_vec()
    }

    #[test]
    fn token_overlap_is_the_multiset_jaccard() {
        let sim = PhraseSimilarity::TokenOverlap;
        assert_eq!(sim.similarity(&v(&["google", "cloud"]), &v(&["google", "cloud"])), 1.0);
        assert_eq!(sim.similarity(&v(&["google", "cloud"]), &v(&["google", "platform"])), 1.0 / 3.0);
        assert_eq!(sim.similarity(&v(&["a", "b", "b"]), &v(&["b", "b", "c"])), 2.0 / 4.0);
        assert_eq!(sim.similarity(&v(&["a"]), &v(&["b"])), 0.0);
        assert_eq!(sim.similarity(&[], &v(&["a"])), 0.0);
        assert_eq!(sim.similarity(&[], &[]), 0.0);
    }

    #[test]
    fn trigram_jaccard_catches_shared_stems() {
        let sim = PhraseSimilarity::TrigramJaccard;
        assert_eq!(sim.similarity(&v(&["goldbloom"]), &v(&["goldbloom"])), 1.0);
        let near = sim.similarity(&v(&["goldbloom"]), &v(&["goldblooms"]));
        assert!(near > 0.5 && near < 1.0, "{near}");
        assert_eq!(sim.similarity(&v(&["ab"]), &v(&["cd"])), 0.0);
        // Two empty phrases agree.
        assert_eq!(sim.similarity(&[], &[]), 1.0);
    }
}

/// The deduplication pass configuration: which similarity decides
/// near-duplicates and above which value a candidate is dropped. The
/// default is the design's token overlap at 0.8, the value the
/// calibration record pins until a marked-up fixture says otherwise.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dedup {
    pub similarity: PhraseSimilarity,
    pub threshold: f64,
}

impl Default for Dedup {
    fn default() -> Dedup {
        Dedup { similarity: PhraseSimilarity::TokenOverlap, threshold: 0.8 }
    }
}

impl Dedup {
    /// The threshold as a provenance config hash contribution.
    pub(crate) fn config_hash(&self) -> u64 {
        let disc = match self.similarity {
            PhraseSimilarity::TokenOverlap => 0xD1u64,
            PhraseSimilarity::TrigramJaccard => 0xD2u64,
        };
        (disc << 56) ^ self.threshold.to_bits()
    }
}
