//! Numeric CRF decoding against an exhaustive probability calculation.
use turbo_analysis::{IxRange, Tag, TagSet};
use turbo_sequence::crf::{CrfScores, EntityDecoder, EntityEncoding, EntityLabel};

#[test]
fn native_scores_decode_without_a_model_archive() {
    let decoder = EntityDecoder::new(
        EntityEncoding::Bio,
        TagSet::new("entities", &["person"]),
        vec![EntityLabel::Outside, EntityLabel::Begin(Tag(0)), EntityLabel::Inside(Tag(0))],
    )
    .unwrap();
    let emissions = vec![vec![0.0; 3]];
    let lattice = decoder
        .lattice(CrfScores { emissions: &emissions, transitions: &[0.0; 9], start: &[0.0; 3], end: &[0.0; 3] })
        .unwrap();
    let paths = lattice.best_paths(4).unwrap();
    assert_eq!(paths.len(), 2);
    assert_eq!(lattice.span_distribution(IxRange::new(0, 1)).unwrap(), vec![0.5, 0.5]);
    let entity = &paths.iter().find(|p| !p.entities.is_empty()).unwrap().entities[0];
    assert_eq!(entity.tokens, IxRange::new(0, 1));
    assert_eq!(entity.kind, Tag(0));
    assert_eq!(entity.probability, 0.5);
}

fn schema(encoding: EntityEncoding) -> Vec<EntityLabel> {
    let mut labels = vec![EntityLabel::Outside];
    for t in [Tag(0), Tag(1)] {
        labels.extend([EntityLabel::Begin(t), EntityLabel::Inside(t)]);
        if encoding == EntityEncoding::Bilou {
            labels.extend([EntityLabel::Last(t), EntityLabel::Unit(t)]);
        }
    }
    labels.reverse(); // State columns need not be grouped or put Outside first.
    labels
}

// Exhaustive oracle: parse complete entities directly, without library masks,
// codec classes, feature strings, a decoder or an external model.
fn entities(path: &[EntityLabel], encoding: EntityEncoding) -> Option<Vec<(usize, usize, Tag)>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < path.len() {
        let start = at;
        match path[at] {
            EntityLabel::Outside => at += 1,
            EntityLabel::Unit(kind) if encoding == EntityEncoding::Bilou => {
                at += 1;
                out.push((start, at, kind));
            }
            EntityLabel::Begin(kind) => {
                at += 1;
                while path.get(at) == Some(&EntityLabel::Inside(kind)) {
                    at += 1;
                }
                if encoding == EntityEncoding::Bilou {
                    if path.get(at) != Some(&EntityLabel::Last(kind)) {
                        return None;
                    }
                    at += 1;
                }
                out.push((start, at, kind));
            }
            _ => return None,
        }
    }
    Some(out)
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}");
}

#[test]
fn both_schemas_match_all_paths_and_all_span_marginals() {
    for encoding in [EntityEncoding::Bio, EntityEncoding::Bilou] {
        let labels = schema(encoding);
        let n = labels.len();
        let decoder =
            EntityDecoder::new(encoding, TagSet::new("entities", &["person", "place"]), labels.clone()).unwrap();
        let transitions: Vec<_> = (0..n * n).map(|i| ((i * 13 + 1) % 19) as f64 / 20.0 - 0.45).collect();
        let start: Vec<_> = (0..n).map(|i| i as f64 / 31.0).collect();
        let end: Vec<_> = (0..n).map(|i| i as f64 / -17.0).collect();
        for length in 0..=3 {
            let emissions: Vec<Vec<_>> =
                (0..length).map(|t| (0..n).map(|i| ((i * 7 + t * 11) % 23) as f64 / 13.0).collect()).collect();
            let mut oracle = Vec::new();
            for mut code in 0..n.pow(length as u32) {
                let path: Vec<_> = (0..length)
                    .map(|_| {
                        let y = code % n;
                        code /= n;
                        y
                    })
                    .collect();
                let states: Vec<_> = path.iter().map(|&y| labels[y]).collect();
                let Some(spans) = entities(&states, encoding) else {
                    continue;
                };
                let mut score = 0.0;
                for (t, &y) in path.iter().enumerate() {
                    score += emissions[t][y];
                    score += if t == 0 { start[y] } else { transitions[path[t - 1] * n + y] };
                    if t + 1 == length {
                        score += end[y];
                    }
                }
                oracle.push((path, spans, score.exp()));
            }
            let total: f64 = oracle.iter().map(|p| p.2).sum();
            let lattice = decoder
                .lattice(CrfScores { emissions: &emissions, transitions: &transitions, start: &start, end: &end })
                .unwrap();
            close(lattice.numeric().log_partition().unwrap(), total.ln());
            let paths = lattice.best_paths(4096).unwrap();
            assert_eq!(paths.len(), oracle.len());
            let mut unique = std::collections::BTreeSet::new();
            for path in &paths {
                assert!(unique.insert(path.path.labels.clone()));
                let expected = oracle.iter().find(|p| p.0 == path.path.labels).unwrap();
                close(path.path.log_probability.exp(), expected.2 / total);
                let spans: Vec<_> =
                    path.entities.iter().map(|e| (e.tokens.start.get(), e.tokens.end.get(), e.kind)).collect();
                assert_eq!(spans, expected.1);
                for e in &path.entities {
                    let mass: f64 = oracle
                        .iter()
                        .filter(|p| p.1.contains(&(e.tokens.start.get(), e.tokens.end.get(), e.kind)))
                        .map(|p| p.2)
                        .sum();
                    close(e.probability, mass / total);
                }
            }
            assert!(paths.windows(2).all(|p| p[0].path.log_probability >= p[1].path.log_probability));
            for (t, row) in lattice.numeric().marginals().iter().enumerate() {
                for (y, &probability) in row.iter().enumerate() {
                    let mass: f64 = oracle.iter().filter(|p| p.0[t] == y).map(|p| p.2).sum();
                    close(probability, mass / total);
                }
            }
            for a in 0..length {
                for b in a + 1..=length {
                    let probabilities = lattice.span_distribution(IxRange::new(a as u32, b as u32)).unwrap();
                    let mut exact = 0.0;
                    for kind in 0..2 {
                        let mass: f64 = oracle.iter().filter(|p| p.1.contains(&(a, b, Tag(kind)))).map(|p| p.2).sum();
                        close(probabilities[1 + kind as usize], mass / total);
                        exact += mass / total;
                    }
                    close(probabilities[0], 1.0 - exact);
                    close(probabilities.iter().sum(), 1.0);
                }
            }
        }
    }
}

#[test]
fn invalid_schemas_and_scores_are_errors() {
    let valid = schema(EntityEncoding::Bio);
    let kinds = TagSet::new("entities", &["person", "place"]);
    for labels in [vec![], valid[..4].to_vec(), vec![EntityLabel::Outside; 5], vec![EntityLabel::Outside; 4097]] {
        assert!(EntityDecoder::new(EntityEncoding::Bio, kinds.clone(), labels).is_err());
    }
    for bad in [EntityLabel::Begin(Tag::NONE), EntityLabel::Last(Tag(0))] {
        let mut labels = valid.clone();
        labels[0] = bad;
        assert!(EntityDecoder::new(EntityEncoding::Bio, kinds.clone(), labels).is_err());
    }
    for names in [["person", "person"], ["", "place"]] {
        assert!(EntityDecoder::new(EntityEncoding::Bio, TagSet::new("entities", &names), valid.clone()).is_err());
    }
    let decoder = EntityDecoder::new(EntityEncoding::Bio, kinds, valid).unwrap();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            decoder
                .lattice(CrfScores {
                    emissions: &[vec![value; 5]],
                    transitions: &[0.0; 25],
                    start: &[0.0; 5],
                    end: &[0.0; 5]
                })
                .is_err()
        );
    }
    assert!(
        decoder
            .lattice(CrfScores {
                emissions: &[vec![0.0; 4]],
                transitions: &[0.0; 25],
                start: &[0.0; 5],
                end: &[0.0; 5]
            })
            .is_err()
    );
    let lattice = decoder
        .lattice(CrfScores { emissions: &[vec![0.0; 5]], transitions: &[0.0; 25], start: &[0.0; 5], end: &[0.0; 5] })
        .unwrap();
    for range in [IxRange::new(0, 0), IxRange::new(1, 0), IxRange::new(0, 2), IxRange::new(u32::MAX - 1, u32::MAX - 1)]
    {
        assert!(lattice.span_distribution(range).is_err());
    }
    assert!(lattice.best_paths(0).is_err());
    assert!(lattice.best_paths(4097).is_err());
}

#[test]
fn schema_identity_includes_kind_names_state_order_and_encoding() {
    let a = EntityDecoder::new(
        EntityEncoding::Bio,
        TagSet::new("entities", &["person", "place"]),
        schema(EntityEncoding::Bio),
    )
    .unwrap();
    let b = EntityDecoder::new(
        EntityEncoding::Bio,
        TagSet::new("entities", &["animal", "place"]),
        schema(EntityEncoding::Bio),
    )
    .unwrap();
    let mut order = schema(EntityEncoding::Bio);
    order.swap(0, 1);
    let c = EntityDecoder::new(EntityEncoding::Bio, a.kinds().clone(), order).unwrap();
    let d = EntityDecoder::new(EntityEncoding::Bilou, a.kinds().clone(), schema(EntityEncoding::Bilou)).unwrap();
    for other in [&b, &c, &d] {
        assert_ne!(a.schema_hash(), other.schema_hash());
    }
    let repeated = EntityDecoder::new(EntityEncoding::Bio, a.kinds().clone(), a.labels().to_vec()).unwrap();
    assert_eq!(a.schema_hash(), repeated.schema_hash());
}

#[test]
fn empty_kind_set_and_tied_paths_are_deterministic() {
    for encoding in [EntityEncoding::Bio, EntityEncoding::Bilou] {
        let decoder = EntityDecoder::new(encoding, TagSet::new("empty", &[]), vec![EntityLabel::Outside]).unwrap();
        let lattice = decoder
            .lattice(CrfScores { emissions: &vec![vec![0.0]; 3], transitions: &[0.0], start: &[0.0], end: &[0.0] })
            .unwrap();
        let paths = lattice.best_paths(8).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].path.labels, vec![0, 0, 0]);
        assert!(paths[0].entities.is_empty());
        assert_eq!(lattice.span_distribution(IxRange::new(0, 3)).unwrap(), vec![1.0]);
        let labels = schema(encoding);
        let n = labels.len();
        let decoder = EntityDecoder::new(encoding, TagSet::new("kinds", &["a", "b"]), labels).unwrap();
        let lattice = decoder
            .lattice(CrfScores {
                emissions: &vec![vec![0.0; n]; 3],
                transitions: &vec![0.0; n * n],
                start: &vec![0.0; n],
                end: &vec![0.0; n],
            })
            .unwrap();
        let all = lattice.best_paths(4096).unwrap();
        assert_eq!(lattice.best_paths(8).unwrap(), all[..8]);
        assert_eq!(lattice.best_paths(8).unwrap(), lattice.best_paths(8).unwrap());
    }
}

#[test]
fn common_offsets_preserve_probabilities_and_bad_parameters_are_rejected() {
    for encoding in [EntityEncoding::Bio, EntityEncoding::Bilou] {
        let labels = schema(encoding);
        let n = labels.len();
        let decoder = EntityDecoder::new(encoding, TagSet::new("kinds", &["a", "b"]), labels).unwrap();
        let emissions: Vec<Vec<_>> = (0..3).map(|t| (0..n).map(|i| ((t + i) % 7) as f64 * 0.25).collect()).collect();
        let transition: Vec<_> = (0..n * n).map(|i| (i % 5) as f64 * 0.25).collect();
        let start: Vec<_> = (0..n).map(|i| (i % 3) as f64 * 0.25).collect();
        let end: Vec<_> = (0..n).map(|i| (i % 2) as f64 * 0.25).collect();
        let base = decoder
            .lattice(CrfScores { emissions: &emissions, transitions: &transition, start: &start, end: &end })
            .unwrap();
        let shifted = |values: &[f64]| values.iter().map(|v| v + 1e12).collect::<Vec<_>>();
        let shifted_emissions: Vec<_> = emissions.iter().map(|row| shifted(row)).collect();
        let offset = decoder
            .lattice(CrfScores {
                emissions: &shifted_emissions,
                transitions: &shifted(&transition),
                start: &shifted(&start),
                end: &shifted(&end),
            })
            .unwrap();
        assert_eq!(base.best_paths(8).unwrap(), offset.best_paths(8).unwrap());
        assert_eq!(
            base.span_distribution(IxRange::new(0, 2)).unwrap(),
            offset.span_distribution(IxRange::new(0, 2)).unwrap()
        );
        for parameter in 0..3 {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut arrays = [transition.clone(), start.clone(), end.clone()];
                arrays[parameter][0] = value;
                assert!(
                    decoder
                        .lattice(CrfScores {
                            emissions: &emissions,
                            transitions: &arrays[0],
                            start: &arrays[1],
                            end: &arrays[2]
                        })
                        .is_err()
                );
            }
            let mut arrays = [transition.clone(), start.clone(), end.clone()];
            arrays[parameter].pop();
            assert!(
                decoder
                    .lattice(CrfScores {
                        emissions: &emissions,
                        transitions: &arrays[0],
                        start: &arrays[1],
                        end: &arrays[2]
                    })
                    .is_err()
            );
        }
    }
}
