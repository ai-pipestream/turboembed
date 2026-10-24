//! The entities and relations of the groups `groups.rs` finds.

use std::collections::BTreeMap;

use turbo_analysis::{Ix, IxRange, Layer, Span};
use turbo_analysis_layers::{
    ConsolidatedMention, DocumentEntities, DocumentEntity, EntityRelation, SyntacticRole, SyntacticRoleCounts, Tokens,
};

use super::groups;
use super::{ConsolidatorConfig, Inputs};

/// One mention while an entity is built: its bytes, its name row, its
/// tokens and head token when known.
struct Draft {
    span: Span,
    entity: Option<usize>,
    tokens: Option<IxRange<Tokens>>,
    head: Option<Ix<Tokens>>,
}

pub(super) fn build(inp: &Inputs<'_>, config: ConsolidatorConfig) -> Result<DocumentEntities, String> {
    let e = inp.entities;
    let n = e.spans.len();
    let m = inp.mentions.map_or(0, Layer::len);
    let mut sets = groups::group(inp, config);
    let mut by_root: BTreeMap<usize, Vec<Draft>> = BTreeMap::new();
    for row in 0..n {
        let ix = Ix::new(row as u32);
        let tokens = e.tokens[row];
        let draft =
            Draft { span: e.spans.get(ix), entity: Some(row), tokens: Some(tokens), head: head_of(inp, tokens) };
        by_root.entry(sets.find(row)).or_default().push(draft);
    }
    if let Some(mentions) = inp.mentions {
        for i in 0..m {
            let node = groups::mention_node(inp, i);
            if node < n {
                continue;
            }
            let ix = Ix::new(i as u32);
            let draft = Draft {
                span: mentions.spans.get(ix),
                entity: None,
                tokens: Some(mentions.tokens[i]),
                head: Some(mentions.head[i]),
            };
            by_root.entry(sets.find(node)).or_default().push(draft);
        }
    }
    let mut out = DocumentEntities::new(inp.view.id());
    for (_, mut drafts) in by_root {
        if drafts.iter().all(|d| d.entity.is_none()) {
            continue;
        }
        drafts.sort_by_key(|d| (d.span.start, d.span.end, d.entity.is_none()));
        drafts.dedup_by(|b, a| a.span == b.span);
        out.entities.push(entity(inp, &drafts));
    }
    out.entities.sort_by_key(|x| (x.mentions[0].span.start, x.mentions[0].span.end));
    out.relations = relations(inp, &out);
    Ok(out)
}

/// The head token of a name: the first of its tokens whose head lies
/// outside it.
fn head_of(inp: &Inputs<'_>, tokens: IxRange<Tokens>) -> Option<Ix<Tokens>> {
    let deps = inp.dependencies?;
    let (lo, hi) = (tokens.start.get(), tokens.end.get());
    (lo..hi.min(deps.head.len()))
        .find(|&t| deps.head[t].is_none_or(|h| h.get() < lo || h.get() >= hi))
        .map(|t| Ix::new(t as u32))
}

fn entity(inp: &Inputs<'_>, drafts: &[Draft]) -> DocumentEntity {
    let e = inp.entities;
    let rows: Vec<usize> = drafts.iter().filter_map(|d| d.entity).collect();
    // The kind with the most score, the lowest tag on a tie.
    let mut weight: BTreeMap<u16, f64> = BTreeMap::new();
    for &r in &rows {
        *weight.entry(e.kind[r].0).or_default() += e.score[r].max(0.0) + 1e-9;
    }
    let kind = weight.iter().fold((0u16, f64::MIN), |best, (&k, &w)| if w > best.1 { (k, w) } else { best }).0;
    let link = inp.linked.and_then(|l| {
        rows.iter()
            .filter(|&&r| l.kb_id[r].is_some())
            .max_by(|&&a, &&b| l.score[a].total_cmp(&l.score[b]).then(b.cmp(&a)))
            .map(|&r| (r, l))
    });
    let longest =
        *rows.iter().max_by_key(|&&r| (groups::surface(inp, r).chars().count(), usize::MAX - r)).unwrap_or(&rows[0]);
    let (canonical_name, representative, kb_id, link_score) = match link {
        Some((r, l)) => (l.canonical_name[r].as_str().into(), r, l.kb_id[r].as_deref().map(Into::into), l.score[r]),
        None => (groups::surface(inp, longest).into(), longest, None, 0.0),
    };
    let mut role_counts = SyntacticRoleCounts::default();
    let mentions: Vec<ConsolidatedMention> = drafts
        .iter()
        .map(|d| {
            let role = role_of(inp, d.head);
            if let Some(role) = role {
                role_counts.add(role);
            }
            ConsolidatedMention {
                span: d.span,
                entity: d.entity.map(|r| Ix::new(r as u32)),
                is_representative: d.entity == Some(representative),
                role,
            }
        })
        .collect();
    let (geo_coord, country_code) = place(inp, drafts);
    let (mean_sentiment, sentiment_mentions) = sentiment(inp, drafts);
    let best = rows.iter().map(|&r| e.score[r]).fold(0.0f64, f64::max).max(link_score);
    DocumentEntity {
        canonical_name,
        kind: e.kinds.label(turbo_analysis::Tag(kind)).into(),
        kb_id,
        geo_coord,
        country_code,
        mentions: mentions.into(),
        role_counts,
        mean_sentiment,
        sentiment_mentions,
        confidence: best.clamp(0.0, 1.0) as f32,
    }
}

fn role_of(inp: &Inputs<'_>, head: Option<Ix<Tokens>>) -> Option<SyntacticRole> {
    let deps = inp.dependencies?;
    let t = head?.get();
    let tag = *deps.relation.get(t)?;
    Some(SyntacticRole::of_relation(deps.relations.label(tag)))
}

/// The most confident resolved location over one of the entity's names.
fn place(inp: &Inputs<'_>, drafts: &[Draft]) -> (Option<(f32, f32)>, Option<[u8; 2]>) {
    let Some(locs) = inp.locations else { return (None, None) };
    let mut best: Option<(f64, usize)> = None;
    for i in 0..locs.spans.len() {
        let s = locs.spans.get(Ix::new(i as u32));
        let hit = drafts.iter().any(|d| d.entity.is_some() && d.span.start < s.end && s.start < d.span.end);
        if hit && best.is_none_or(|(c, _)| locs.confidence[i] > c) {
            best = Some((locs.confidence[i], i));
        }
    }
    let Some((_, i)) = best else { return (None, None) };
    let entry = &locs.entry[i];
    let at = entry.location();
    let code = entry.country_code().and_then(|c| match c.as_bytes() {
        &[a, b] if a.is_ascii_alphabetic() && b.is_ascii_alphabetic() => {
            Some([a.to_ascii_uppercase(), b.to_ascii_uppercase()])
        }
        _ => None,
    });
    (Some((at.latitude() as f32, at.longitude() as f32)), code)
}

/// The mean `positive - negative` of the explicit aspects whose tokens
/// meet a mention's, each aspect counted once.
fn sentiment(inp: &Inputs<'_>, drafts: &[Draft]) -> (f32, u16) {
    let Some(aspects) = inp.aspects else { return (0.0, 0) };
    let (mut sum, mut count) = (0.0f64, 0u16);
    for (i, range) in aspects.tokens.iter().enumerate() {
        let Some(a) = range else { continue };
        let meets = drafts.iter().filter_map(|d| d.tokens).any(|t| t.start < a.end && a.start < t.end);
        let polarity = aspects.positive[i] - aspects.negative[i];
        if meets && polarity.is_finite() {
            sum += polarity.clamp(-1.0, 1.0);
            count = count.saturating_add(1);
        }
    }
    if count == 0 { (0.0, 0) } else { ((sum / f64::from(count)) as f32, count) }
}

/// Each relation edge lifted to its arguments' entities, an edge within
/// one entity left out.
fn relations(inp: &Inputs<'_>, out: &DocumentEntities) -> Vec<EntityRelation> {
    let Some(edges) = inp.relations else { return Vec::new() };
    let mut owner = vec![u32::MAX; inp.entities.spans.len()];
    for (i, x) in out.entities.iter().enumerate() {
        for row in x.mentions.iter().filter_map(|m| m.entity) {
            owner[row.get()] = i as u32;
        }
    }
    let mut merged: BTreeMap<(u32, u32, &str), (f64, u16)> = BTreeMap::new();
    for r in 0..edges.from.len() {
        let (s, o) = (edges.from[r].get(), edges.to[r].get());
        let (Some(&s), Some(&o)) = (owner.get(s), owner.get(o)) else { continue };
        if s == u32::MAX || o == u32::MAX || s == o || !edges.score[r].is_finite() {
            continue;
        }
        let at = merged.entry((s, o, edges.kinds.label(edges.kind[r]))).or_insert((f64::MIN, 0));
        at.0 = at.0.max(edges.score[r]);
        at.1 = at.1.saturating_add(1);
    }
    merged
        .into_iter()
        .map(|((subject, object, kind), (score, support))| EntityRelation {
            subject,
            object,
            kind: kind.into(),
            score: score as f32,
            support,
        })
        .collect()
}
