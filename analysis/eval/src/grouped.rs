//! Paired cluster uncertainty conditional on the fixed evaluation corpus.
//! Queries from one source paper/book are resampled together. Contract
//! readers already collapse repeated questions to unique templates, which
//! are their resampling units. Cluster draws preserve the query-weighted
//! estimand by dividing sampled sums by sampled counts.
use crate::{
    Result,
    random::JavaRandom,
    set::LoadedSet,
    significance::{Bootstrap, percentile},
};
use std::collections::BTreeMap;

pub fn bootstrap(rows: &[(String, f64)], resamples: u32, seed: i64) -> Result<Bootstrap> {
    if rows.is_empty() || resamples == 0 {
        return Err("grouped bootstrap needs rows and resamples".into());
    }
    let mut groups: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    for (group, value) in rows {
        if group.is_empty() || !value.is_finite() {
            return Err("invalid bootstrap group or difference".into());
        }
        let g = groups.entry(group).or_default();
        g.0 += value;
        g.1 += 1;
    }
    let groups: Vec<_> = groups.into_values().collect();
    let n = i32::try_from(groups.len()).map_err(|_| "too many bootstrap groups")?;
    let mut random = JavaRandom::new(seed);
    let mut means = Vec::with_capacity(resamples as usize);
    for _ in 0..resamples {
        let (mut sum, mut count) = (0.0, 0);
        for _ in 0..n {
            let (s, c) = groups[random.next_int_bounded(n) as usize];
            sum += s;
            count += c;
        }
        means.push(sum / count as f64);
    }
    means.sort_by(f64::total_cmp);
    Ok(Bootstrap {
        mean: rows.iter().map(|r| r.1).sum::<f64>() / rows.len() as f64,
        low: percentile(&means, 0.025),
        high: percentile(&means, 0.975),
    })
}

/// Group membership is derived from pinned source judgements, never scores.
/// Unknown datasets are refused rather than silently assuming independence.
pub fn query_groups(set: &LoadedSet) -> Result<BTreeMap<String, String>> {
    let template = match set.name.as_str() {
        "cuad" | "contract-nli" => true,
        "qasper" | "gutenqa" => false,
        _ => return Err(format!("no grouping protocol for {}", set.name)),
    };
    let mut groups = BTreeMap::new();
    for q in &set.queries {
        let group = if template {
            q.id.clone()
        } else {
            let docs: Vec<_> = set.docs.iter().filter(|d| set.qrels.grade(&q.id, &d.id) > 0).collect();
            if docs.len() != 1 {
                return Err(format!("query {} needs exactly one source document", q.id));
            }
            docs[0].id.clone()
        };
        groups.insert(q.id.clone(), group);
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_queries_do_not_narrow_cluster_interval() {
        let rows = vec![("a".into(), -1.0), ("b".into(), 1.0)];
        let repeated = rows.iter().flat_map(|r| std::iter::repeat_n(r.clone(), 100)).collect::<Vec<_>>();
        assert_eq!(bootstrap(&rows, 10_000, 154).unwrap(), bootstrap(&repeated, 10_000, 154).unwrap());
    }
    #[test]
    fn unequal_clusters_preserve_query_weighted_mean() {
        let rows = vec![("a".into(), 0.0), ("a".into(), 0.0), ("b".into(), 1.0)];
        let b = bootstrap(&rows, 1000, 154).unwrap();
        assert_eq!(b.mean, 1.0 / 3.0);
        assert_eq!((b.low, b.high), (0.0, 1.0));
    }
    #[test]
    fn singleton_clusters_match_existing_bootstrap() {
        let diffs = [-0.5, 0.1, 0.4];
        let rows = diffs.iter().enumerate().map(|(i, &v)| (i.to_string(), v)).collect::<Vec<_>>();
        assert_eq!(bootstrap(&rows, 1000, 154).unwrap(), crate::significance::bootstrap(&diffs, 1000, 154).unwrap());
    }
    #[test]
    fn bad_samples_are_refused() {
        assert!(bootstrap(&[], 10, 154).is_err());
        assert!(bootstrap(&[("a".into(), f64::NAN)], 10, 154).is_err());
        assert!(bootstrap(&[("a".into(), 1.0)], 0, 154).is_err());
    }
}
