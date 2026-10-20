//! Extraction of coreference clusters and cross-sentence continuity links from
//! `Mentions` and `Entities` layers.

use turbo_analysis::{Edges, Groups, Ix, Span, View};
use turbo_analysis_layers::{Entities, Sentences};

use super::{CorefChains, CorefLink, MILLI, MentionKind, Mentions, ResolvedMention, is_pronoun_text};

fn locate_sentence(span: Span, sents: &Sentences) -> Option<usize> {
    let n = sents.spans.len();
    if n == 0 {
        return None;
    }
    let (_, ends) = sents.spans.columns();
    let idx = ends.partition_point(|&e| e <= span.start);
    (idx < n).then_some(idx)
}

fn push_edge_link(links: &mut Vec<CorefLink>, candidate: CorefLink) {
    if let Some(existing) = links.iter_mut().find(|l| {
        l.s_lo == candidate.s_lo
            && l.s_hi == candidate.s_hi
            && l.span_lo == candidate.span_lo
            && l.span_hi == candidate.span_hi
    }) {
        existing.weight_milli = existing.weight_milli.max(candidate.weight_milli);
        existing.target_is_pronoun |= candidate.target_is_pronoun;
    } else {
        links.push(candidate);
    }
}

fn push_cluster_link(links: &mut Vec<CorefLink>, candidate: CorefLink) {
    if let Some(existing) = links.iter_mut().find(|l| {
        l.s_lo == candidate.s_lo
            && l.s_hi == candidate.s_hi
            && l.span_lo == candidate.span_lo
            && l.span_hi == candidate.span_hi
    }) {
        existing.target_is_pronoun |= candidate.target_is_pronoun;
    } else {
        links.push(candidate);
    }
}

fn push_cluster_only(mut cluster: Vec<ResolvedMention>, clusters: &mut Vec<Vec<ResolvedMention>>) {
    cluster.sort_unstable_by_key(|m| (m.sentence, m.span.start, m.span.end));
    cluster.dedup_by(|a, b| a.sentence == b.sentence && a.span == b.span);
    if cluster.len() >= 2 {
        clusters.push(cluster);
    }
}

fn add_cluster_links(
    mut cluster: Vec<ResolvedMention>,
    clusters: &mut Vec<Vec<ResolvedMention>>,
    links: &mut Vec<CorefLink>,
) {
    cluster.sort_unstable_by_key(|m| (m.sentence, m.span.start, m.span.end));
    cluster.dedup_by(|a, b| a.sentence == b.sentence && a.span == b.span);
    if cluster.len() < 2 {
        return;
    }
    for w in cluster.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a.sentence < b.sentence {
            push_cluster_link(
                links,
                CorefLink {
                    s_lo: a.sentence,
                    s_hi: b.sentence,
                    span_lo: a.span,
                    span_hi: b.span,
                    target_is_pronoun: b.is_pronoun,
                    weight_milli: MILLI,
                },
            );
        }
    }
    for idx in 1..cluster.len() {
        let b = cluster[idx];
        if b.is_pronoun
            && let Some(ant) = cluster[..idx].iter().rev().find(|a| a.sentence < b.sentence)
        {
            push_cluster_link(
                links,
                CorefLink {
                    s_lo: ant.sentence,
                    s_hi: b.sentence,
                    span_lo: ant.span,
                    span_hi: b.span,
                    target_is_pronoun: true,
                    weight_milli: MILLI,
                },
            );
        }
    }
    clusters.push(cluster);
}

/// Extracts [`CorefChains`] from optional `Mentions` / `Entities` layers and
/// their `Groups` and `Edges` coreference annotations.
#[allow(clippy::too_many_arguments)]
pub fn extract_chains(
    sentences: &Sentences,
    view: Option<&View>,
    mentions: Option<&Mentions>,
    mention_groups: Option<&Groups<Mentions>>,
    mention_edges: Option<&Edges<Mentions, Mentions>>,
    entities: Option<&Entities>,
    entity_groups: Option<&Groups<Entities>>,
    entity_edges: Option<&Edges<Entities, Entities>>,
) -> CorefChains {
    let n = sentences.spans.len();
    let mut sentence_has_mention = vec![false; n];
    let mut clusters = Vec::new();
    let mut links = Vec::new();
    if n == 0 {
        return CorefChains { sentence_count: 0, sentence_has_mention, clusters, links };
    }

    if let Some(m_layer) = mentions {
        let resolve = |ix: Ix<Mentions>| -> Option<ResolvedMention> {
            let i = ix.get();
            if i >= m_layer.spans.len() {
                return None;
            }
            let span = m_layer.spans.get(ix);
            let s_idx = m_layer
                .sentence
                .get(i)
                .map(|s| s.get())
                .filter(|&s| s < n)
                .or_else(|| locate_sentence(span, sentences))?;
            let text_pron = view.and_then(|v| v.text().get(span.range())).is_some_and(is_pronoun_text);
            let is_pronoun =
                matches!(m_layer.kind.get(i), Some(MentionKind::Pronominal | MentionKind::Demonstrative)) || text_pron;
            Some(ResolvedMention { sentence: s_idx, span, is_pronoun })
        };
        for i in 0..m_layer.spans.len() {
            if let Some(rm) = resolve(Ix::new(i as u32)) {
                sentence_has_mention[rm.sentence] = true;
            }
        }
        if let Some(edges) = mention_edges {
            for (_, edge) in edges.iter() {
                let (Some(a), Some(b)) = (resolve(edge.from), resolve(edge.to)) else {
                    continue;
                };
                if mention_groups.is_none() {
                    push_cluster_only(vec![a, b], &mut clusters);
                }
                if a.sentence != b.sentence {
                    let (lo, hi) = if a.sentence < b.sentence { (a, b) } else { (b, a) };
                    let w = if edge.score.is_finite() && edge.score > 0.0 {
                        (edge.score.clamp(0.0, 1.0) * (MILLI as f64)).round() as i64
                    } else {
                        MILLI
                    };
                    push_edge_link(
                        &mut links,
                        CorefLink {
                            s_lo: lo.sentence,
                            s_hi: hi.sentence,
                            span_lo: lo.span,
                            span_hi: hi.span,
                            target_is_pronoun: hi.is_pronoun,
                            weight_milli: w.clamp(1, MILLI),
                        },
                    );
                }
            }
        }
        if let Some(groups) = mention_groups {
            for (_, members) in groups.iter() {
                let cluster: Vec<ResolvedMention> = members.iter().copied().filter_map(resolve).collect();
                add_cluster_links(cluster, &mut clusters, &mut links);
            }
        }
    }

    if let Some(e_layer) = entities {
        let resolve = |ix: Ix<Entities>| -> Option<ResolvedMention> {
            let i = ix.get();
            if i >= e_layer.spans.len() {
                return None;
            }
            let span = e_layer.spans.get(ix);
            let s_idx = locate_sentence(span, sentences)?;
            let kind_pron = e_layer
                .kind
                .get(i)
                .map(|&t| e_layer.kinds.label(t))
                .is_some_and(|l| l.eq_ignore_ascii_case("pron") || l.eq_ignore_ascii_case("pronoun"));
            let text_pron = view.and_then(|v| v.text().get(span.range())).is_some_and(is_pronoun_text);
            Some(ResolvedMention { sentence: s_idx, span, is_pronoun: kind_pron || text_pron })
        };
        for i in 0..e_layer.spans.len() {
            if let Some(rm) = resolve(Ix::new(i as u32)) {
                sentence_has_mention[rm.sentence] = true;
            }
        }
        if let Some(edges) = entity_edges {
            for (_, edge) in edges.iter() {
                let (Some(a), Some(b)) = (resolve(edge.from), resolve(edge.to)) else {
                    continue;
                };
                if entity_groups.is_none() {
                    push_cluster_only(vec![a, b], &mut clusters);
                }
                if a.sentence != b.sentence {
                    let (lo, hi) = if a.sentence < b.sentence { (a, b) } else { (b, a) };
                    let w = if edge.score.is_finite() && edge.score > 0.0 {
                        (edge.score.clamp(0.0, 1.0) * (MILLI as f64)).round() as i64
                    } else {
                        MILLI
                    };
                    push_edge_link(
                        &mut links,
                        CorefLink {
                            s_lo: lo.sentence,
                            s_hi: hi.sentence,
                            span_lo: lo.span,
                            span_hi: hi.span,
                            target_is_pronoun: hi.is_pronoun,
                            weight_milli: w.clamp(1, MILLI),
                        },
                    );
                }
            }
        }
        if let Some(groups) = entity_groups {
            for (_, members) in groups.iter() {
                let cluster: Vec<ResolvedMention> = members.iter().copied().filter_map(resolve).collect();
                add_cluster_links(cluster, &mut clusters, &mut links);
            }
        }
    }

    CorefChains { sentence_count: n, sentence_has_mention, clusters, links }
}
