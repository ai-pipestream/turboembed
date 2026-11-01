//! Grouped uncertainty and per-query oracle diagnostics for #1486.
//! Usage: long_document_report <cache> <candidate> <windows> <baseline>...
//! Prints JSON. Both gate verdicts are retained; thresholds are unchanged.
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use turbo_eval::{
    Result,
    chunk_gate::{self, EVIDENCE_LOW, FLOOR_POINTS},
    grouped,
    record::EvalRecord,
    run::read_records,
    significance::{RESAMPLES, SEED},
};

fn comparison(
    a: &BTreeMap<String, f64>,
    b: &BTreeMap<String, f64>,
    groups: &BTreeMap<String, String>,
) -> Result<Value> {
    if a.keys().ne(b.keys()) {
        return Err("different scored query IDs".into());
    }
    let rows = a
        .iter()
        .map(|(q, &v)| {
            let group = groups.get(q).ok_or_else(|| format!("no source group for query {q}"))?;
            Ok((q, group, v, b[q]))
        })
        .collect::<Result<Vec<_>>>()?;
    let grouped = rows.iter().map(|(_, g, a, b)| ((*g).clone(), a - b)).collect::<Vec<_>>();
    let interval = grouped::bootstrap(&grouped, RESAMPLES, SEED)?;
    let queries: Vec<_> = rows
        .iter()
        .map(|(q, g, a, b)| {
            json!({"query": q, "group": g, "candidate": a,
        "reference": b, "delta": a - b, "oracle_gain": (a - b).max(0.0)})
        })
        .collect();
    Ok(json!({"interval": interval, "groups": rows.iter().map(|r| r.1).collect::<BTreeSet<_>>().len(),
        "changed_queries": rows.iter().filter(|r| r.2 != r.3).count(), "per_query": queries,
        "oracle_gain": rows.iter().map(|r| (r.2 - r.3).max(0.0)).sum::<f64>() / rows.len() as f64}))
}
fn scores(r: &EvalRecord, metric: &str) -> Result<BTreeMap<String, f64>> {
    if metric == "ndcg" {
        return Ok(r.metrics.per_query.iter().map(|q| (q.query.clone(), q.ndcg_at_10)).collect());
    }
    let spans = r.metrics.span.as_ref().ok_or("missing span metrics")?;
    Ok(spans
        .per_query
        .iter()
        .map(|q| (q.query.clone(), if metric == "recall" { q.recall } else { q.contained }))
        .collect())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        return Err("long_document_report <cache> <candidate> <windows> <baseline>...".into());
    }
    let candidate = read_records(Path::new(&args[1]))?;
    let baselines = args[2..].iter().map(|p| read_records(Path::new(p))).collect::<Result<Vec<_>>>()?;
    if candidate.iter().any(|r| r.provenance.commit != candidate[0].provenance.commit) {
        return Err("candidate sets were not measured at one frozen commit".into());
    }
    let reference = &baselines[0];
    let others = baselines[1..].iter().map(Vec::as_slice).collect::<Vec<_>>();
    let raw = chunk_gate::chunk_gate(&candidate, reference, &others)?;
    let mut sets = Vec::new();
    let mut pass = true;
    for c in &candidate {
        let set = turbo_eval::sets::find(&c.set.name)?.load(Path::new(&args[0]))?;
        if set.sha256 != c.set.sha256 {
            return Err(format!("{}: source digest differs", set.name));
        }
        let groups = grouped::query_groups(&set)?;
        let mut comparisons = Vec::new();
        for records in &baselines {
            let b = records.iter().find(|r| r.set.name == set.name).ok_or("missing baseline set")?;
            if c.provenance.commit != b.provenance.commit
                || c.provenance.bundle != b.provenance.bundle
                || c.provenance.device != b.provenance.device
                || c.pipeline.analyzer != b.pipeline.analyzer
                || c.pipeline.retrieval != b.pipeline.retrieval
            {
                return Err("comparison provenance/settings differ".into());
            }
            comparisons.push(json!({"baseline": b.pipeline.chunker.name,
                "ndcg": comparison(&scores(c,"ndcg")?, &scores(b,"ndcg")?, &groups)?,
                "recall": comparison(&scores(c,"recall")?, &scores(b,"recall")?, &groups)?,
                "contained": comparison(&scores(c,"contained")?, &scores(b,"contained")?, &groups)?}));
        }
        let evidence = raw.evidence.iter().find(|e| e.set == set.name).ok_or("missing evidence gate")?;
        let floor = raw.floors.iter().find(|e| e.set == set.name).ok_or("missing floor gate")?;
        let best = comparisons.iter().find(|x| x["baseline"] == floor.baseline).ok_or("missing best baseline")?;
        let evidence_pass = comparisons[0]["recall"]["interval"]["low"].as_f64().unwrap() > EVIDENCE_LOW
            && comparisons[0]["contained"]["interval"]["low"].as_f64().unwrap() > EVIDENCE_LOW
            && evidence.split.0 < evidence.split.1;
        let floor_pass = 100.0 * best["ndcg"]["interval"]["high"].as_f64().unwrap() >= FLOOR_POINTS;
        pass &= (!evidence.gated || evidence_pass) && floor_pass;
        sets.push(json!({"set": c.set, "source_commit": c.provenance.commit, "bundle": c.provenance.bundle,
            "documents": set.docs.len(), "templates_or_questions": set.queries.len(),
            "gated": evidence.gated, "split": [evidence.split.0, evidence.split.1],
            "grouped_evidence_pass": evidence_pass, "grouped_floor_pass": floor_pass,
            "floor_baseline": floor.baseline, "comparisons": comparisons}));
    }
    pass &= raw.evidence.iter().any(|e| e.gated);
    println!("{}", serde_json::to_string_pretty(&json!({"protocol": "1486-v1",
        "uncertainty": "paired source-document clusters for qasper/gutenqa; unique templates for contracts; conditional on fixed corpus",
        "resamples": RESAMPLES, "seed": SEED, "raw_gate": raw.to_string(), "grouped_gate_pass": pass, "sets": sets})).map_err(|e| e.to_string())?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn report_retains_losses_and_oracle_gains_with_source_groups() {
        let a = BTreeMap::from([("q1".into(), 0.2), ("q2".into(), 0.8)]);
        let b = BTreeMap::from([("q1".into(), 0.4), ("q2".into(), 0.6)]);
        let g = BTreeMap::from([("q1".into(), "book".into()), ("q2".into(), "book".into())]);
        let r = comparison(&a, &b, &g).unwrap();
        assert_eq!(r["groups"], 1);
        assert_eq!(r["changed_queries"], 2);
        assert!(r["per_query"][0]["delta"].as_f64().unwrap() < 0.0);
        assert!((r["oracle_gain"].as_f64().unwrap() - 0.1).abs() < 1e-12);
        assert_eq!(r["interval"]["low"], r["interval"]["high"]);
    }
    #[test]
    fn mismatched_queries_and_missing_groups_are_refused() {
        let a = BTreeMap::from([("q1".into(), 0.2)]);
        let b = BTreeMap::from([("q2".into(), 0.4)]);
        assert!(comparison(&a, &b, &BTreeMap::new()).is_err());
        assert!(comparison(&a, &a, &BTreeMap::new()).is_err());
    }
}
