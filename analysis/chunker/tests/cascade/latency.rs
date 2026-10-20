use super::*;

/// The direct contextual-only baseline: cosine of every
/// unit row against the contextual query, no stage 1 at
/// all. Returns how many units it scored.
fn direct_contextual_scan(rows: &Vectors<Levels>, query: &[f32]) -> usize {
    let qnorm = query.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
    let mut scored = 0;
    let mut best = f64::MIN;
    for i in 0..rows.len() as u32 {
        let unit = Ix::new(i);
        if rows.is_empty(unit) {
            continue;
        }
        let row = rows.row(unit);
        let norm = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        if norm <= 0.0 {
            continue;
        }
        scored += 1;
        let dot: f64 = row.iter().zip(query).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        best = best.max(dot / (norm * qnorm));
    }
    assert!(best > 0.0);
    scored
}

#[test]
fn cascade_at_64_candidates_beats_direct_contextual_scan() {
    // The fixture document scaled up until stage-2 cost is
    // measurable: the baseline scores every unit in the
    // contextual space, the cascade scores its 64-wide pool.
    let mut text = String::new();
    let mut sentences: Vec<String> = Vec::new();
    for i in 0..120 {
        text.push_str(&format!("# Topic{i}\n\nCats purr{i}. Cats nap. Cats hunt mice. Dogs bark{i}. Dogs run.\n\n"));
        for s in ["# Topic", "Cats purr", "Cats nap.", "Cats hunt mice.", "Dogs bark", "Dogs run."] {
            sentences.push(match s {
                "# Topic" => format!("# Topic{i}"),
                "Cats purr" => format!("Cats purr{i}."),
                "Dogs bark" => format!("Dogs bark{i}."),
                other => other.to_string(),
            });
        }
    }
    let sentence_refs: Vec<&str> = sentences.iter().map(String::as_str).collect();
    let doc = document_with(&text, &sentence_refs);
    let levels = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let vocab = doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let static_rows = rows(&doc, &["Cats", "Dogs"], "static");
    // Contextual rows at a realistic encoder width: the
    // 8-probe pattern tiled to 512 dimensions, so scoring a
    // unit contextually costs what a real encoder row costs
    // while the static stage stays 2-dimensional and cheap.
    let narrow = rows(&doc, &["Cats", "Dogs", "mice", "purr", "bark", "nap", "run", "hunt"], "contextual");
    let contextual_rows = Vectors {
        dim: 512,
        data: narrow.data.iter().flat_map(|v| std::iter::repeat_n(*v, 64)).collect(),
        empty: Vec::new(),
        model: model("contextual"),
        _of: PhantomData,
    };
    let (sm, cm) = (model("static"), model("contextual"));
    let contextual_query: Vec<f32> =
        [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0].iter().flat_map(|v| std::iter::repeat_n(*v, 64)).collect();
    let cascade = Cascade::new(
        levels,
        CascadeConfig { candidates: 64, top_k: 10, ..CascadeConfig::default() },
        Some(VectorSpace { rows: &static_rows, query: QueryVector { model: &sm, vector: &[1.0, 0.0] } }),
        Some(VectorSpace { rows: &contextual_rows, query: QueryVector { model: &cm, vector: &contextual_query } }),
        Some(lexical(&doc, &collection)),
    )
    .unwrap();
    let out = cascade.search(&["cats"]);
    let baseline_scored = direct_contextual_scan(&contextual_rows, &contextual_query);
    assert_eq!(baseline_scored, levels.len());
    assert!(out.contextual_scored <= 128, "pool-bounded stage 2: {} of {}", out.contextual_scored, levels.len());
    assert!(levels.len() > 500, "scaled fixture has {} units", levels.len());

    // Alternating rounds, best of each: the work ratio is
    // over 5x, far above timer noise.
    let (mut cascade_best, mut baseline_best) = (std::time::Duration::MAX, std::time::Duration::MAX);
    for _ in 0..15 {
        let t = std::time::Instant::now();
        std::hint::black_box(cascade.search(&["cats"]));
        cascade_best = cascade_best.min(t.elapsed());
        let t = std::time::Instant::now();
        std::hint::black_box(direct_contextual_scan(&contextual_rows, &contextual_query));
        baseline_best = baseline_best.min(t.elapsed());
    }
    println!(
        "units={} cascade_scored={} cascade_best={cascade_best:?} direct_best={baseline_best:?}",
        levels.len(),
        out.contextual_scored
    );
    assert!(cascade_best < baseline_best, "cascade {cascade_best:?} vs direct scan {baseline_best:?}");
}
