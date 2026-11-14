//! Entity linking, syntactic negation, and epistemic modality partitioning
//! for `SearchDocumentProjection` (`#2345`).

use std::any::TypeId;

use turbo_analysis::{Annotator, Document, Ix, IxRange, Key, Layer, Span, View, ViewId};
use turbo_analysis_layers::{
    Block, BoilerplateKind, Cut, Dependencies, Entities, EpistemicStatus, Modality, ModalityPolarity, ModalityTag,
    TokenArcKind, Tokens,
};
use turbo_entity_linker::LinkedEntities;
use turbo_relation::ModalityAnnotator;

use super::{Failure, SearchProjCx};
use crate::proto::result::{SearchEntityPolarities, SearchLinkedEntity};
use crate::results::typed::find_aligned_to;

pub(crate) fn boilerplate_kind_str(k: BoilerplateKind) -> &'static str {
    match k {
        BoilerplateKind::Header => "header",
        BoilerplateKind::Footer => "footer",
        BoilerplateKind::Navigation => "navigation",
        BoilerplateKind::Disclaimer => "disclaimer",
        BoilerplateKind::Template => "template",
    }
}

fn block_str(b: Block) -> &'static str {
    match b {
        Block::Section => "section",
        Block::Heading => "heading",
        Block::Paragraph => "paragraph",
        Block::List => "list",
        Block::ListItem => "list_item",
        Block::Table => "table",
        Block::TableRow => "table_row",
        Block::TableCell => "table_cell",
        Block::Code => "code",
        Block::Quote => "quote",
        Block::Caption => "caption",
        Block::Footnote => "footnote",
        Block::PageHeader => "page_header",
        Block::PageFooter => "page_footer",
        Block::LooseList => "loose_list",
        Block::Html => "html",
    }
}

pub(crate) fn cut_reason_str(c: Cut) -> String {
    match c {
        Cut::DocumentStart => "document_start".to_owned(),
        Cut::Sentence => "sentence".to_owned(),
        Cut::Section => "section".to_owned(),
        Cut::Heading => "heading".to_owned(),
        Cut::Paragraph => "paragraph".to_owned(),
        Cut::TopicShift => "topic_shift".to_owned(),
        Cut::Clause => "clause".to_owned(),
        Cut::Item => "item".to_owned(),
        Cut::Inside(b) => format!("inside:{}", block_str(b)),
        Cut::Overlap(b) => format!("overlap:{}", block_str(b)),
    }
}

pub(crate) fn token_arc_kind_str(k: TokenArcKind) -> &'static str {
    match k {
        TokenArcKind::Primary => "primary",
        TokenArcKind::DehyphenatedPart => "dehyphenated_part",
        TokenArcKind::CompoundPart => "compound_part",
        TokenArcKind::CompoundWhole => "compound_whole",
        TokenArcKind::LatticeAlternative => "lattice_alternative",
        TokenArcKind::AbbreviationShort => "abbreviation_short",
        TokenArcKind::AbbreviationLong => "abbreviation_long",
        TokenArcKind::Synonym => "synonym",
    }
}

fn epistemic_str(e: EpistemicStatus) -> &'static str {
    match e {
        EpistemicStatus::Asserted => "asserted",
        EpistemicStatus::Hedged => "hedged",
        EpistemicStatus::Reported => "reported",
        EpistemicStatus::Conditional => "conditional",
        EpistemicStatus::Presupposed => "presupposed",
    }
}

fn find_view(doc: &Document, view_id: ViewId) -> Option<&View> {
    doc.layers()
        .filter(|k| k.ty == TypeId::of::<View>())
        .find_map(|k| doc.get_key(&Key::<View>::named(k.name)).filter(|v| v.id() == view_id))
}

fn resolve_modality_layer(doc: &Document) -> Option<Modality> {
    if let Some(existing) = doc.get::<Modality>() {
        return Some(existing.clone());
    }
    let tokens = doc.get::<Tokens>()?;
    let view = find_view(doc, tokens.spans.view())?;
    let deps = doc.get::<Dependencies>();
    ModalityAnnotator::new().annotate((view, tokens, deps)).ok().map(|(m,)| m)
}

fn base_rel(deps: &Dependencies, tok: usize) -> &str {
    let raw = deps.relations.label(deps.relation[tok]);
    raw.split(':').next().unwrap_or(raw)
}

fn token_under(deps: &Dependencies, mut tok: usize, ancestor: usize) -> bool {
    let limit = deps.head.len();
    for _ in 0..=limit {
        if tok == ancestor {
            return true;
        }
        match deps.head.get(tok).copied().flatten() {
            Some(h) => tok = h.get(),
            None => return false,
        }
    }
    false
}

/// Refines clausal determiner negation (`"Acme Corp found no evidence of fraud"`):
/// when a `det` cue (`"no"`, `"neither"`) modifies a non-subject dependent
/// nominal `H`, only entities governed inside `H`'s dependency subtree inherit
/// `Negated`, leaving sibling arguments (such as `nsubj` `"Acme Corp"`) affirmed.
fn nominal_det_excludes_entity(
    doc: &Document,
    modality_view: ViewId,
    cue_span: Span,
    ent_tokens: IxRange<Tokens>,
) -> bool {
    let (Some(tokens), Some(deps)) = (doc.get::<Tokens>(), doc.get::<Dependencies>()) else {
        return false;
    };
    if tokens.spans.view() != modality_view || deps.head.len() != tokens.len() {
        return false;
    }
    let Some(cue_tok) = (0..tokens.len()).find(|&t| tokens.spans.get(Ix::new(t as u32)).start == cue_span.start) else {
        return false;
    };
    if base_rel(deps, cue_tok) != "det" {
        return false;
    }
    let Some(head_ix) = deps.head[cue_tok] else {
        return false;
    };
    let head_tok = head_ix.get();
    // Root nominals ("No evidence of fraud.") and negated subjects ("No auditor
    // found fraud.") scope over the entire clause.
    if deps.head[head_tok].is_none() || matches!(base_rel(deps, head_tok), "nsubj" | "csubj") {
        return false;
    }
    let start = ent_tokens.start.get();
    let end = ent_tokens.end.get().min(tokens.len());
    if start >= end {
        return false;
    }
    !(start..end).any(|t| token_under(deps, t, head_tok))
}

fn classify_entity_modality(
    doc: &Document,
    modality: Option<&Modality>,
    ent_view: ViewId,
    ent_span: Span,
    ent_tokens: IxRange<Tokens>,
) -> (&'static str, &'static str, Option<String>) {
    let Some(m) = modality else {
        return ("affirmed", "asserted", None);
    };
    let m_view = m.spans.view();
    let mapped = if ent_view == m_view {
        ent_span
    } else {
        match doc.map_span(ent_view, m_view, ent_span) {
            Ok(sp) if !sp.is_empty() => sp,
            _ => return ("affirmed", "asserted", None),
        }
    };
    let ModalityTag { mut polarity, epistemic, cue, .. } = m.tag_for_span(mapped).unwrap_or_default();
    let mut resolved_cue = cue;
    if polarity == ModalityPolarity::Negated
        && let Some(c_span) = cue
        && nominal_det_excludes_entity(doc, m_view, c_span, ent_tokens)
    {
        polarity = ModalityPolarity::Affirmed;
        resolved_cue = None;
    }
    let cue_str = resolved_cue.and_then(|sp| doc.text(m_view).get(sp.range()).map(ToOwned::to_owned));
    let bucket = if polarity == ModalityPolarity::Negated {
        "negated"
    } else if matches!(epistemic, EpistemicStatus::Hedged | EpistemicStatus::Reported | EpistemicStatus::Conditional) {
        "uncertain"
    } else {
        "affirmed"
    };
    (bucket, epistemic_str(epistemic), cue_str)
}

pub(crate) fn project_entities(cx: &SearchProjCx<'_>) -> Result<SearchEntityPolarities, Failure> {
    let modality = resolve_modality_layer(cx.doc);
    let mut affirmed = Vec::new();
    let mut negated = Vec::new();
    let mut uncertain = Vec::new();

    for key in cx.doc.layers().filter(|k| k.ty == TypeId::of::<Entities>()) {
        let Some(ents) = cx.doc.get_key(&Key::<Entities>::named(key.name)) else { continue };
        let linked = find_aligned_to::<LinkedEntities>(cx.doc, key);
        let view = ents.spans.view();
        let view_text = cx.doc.text(view);
        for i in 0..ents.len() {
            let ix = Ix::new(i as u32);
            let span = ents.spans.get(ix);
            let (span_start, span_end) = cx.map_to_wire(view, span)?;
            let surface = view_text.get(span.range()).unwrap_or_default();
            let (kb_id, provider_id, canonical_name, matched_alias, score) = match linked {
                Some(le) if i < le.len() => {
                    let row = le.get(ix);
                    let canon = if row.canonical_name.is_empty() { surface } else { row.canonical_name };
                    let alias = if row.matched_alias.is_empty() { surface } else { row.matched_alias };
                    let sc = if row.score > 0.0 { row.score } else { ents.score[i] };
                    (
                        row.kb_id.map(ToOwned::to_owned),
                        row.provider_id.to_owned(),
                        canon.to_owned(),
                        alias.to_owned(),
                        sc,
                    )
                }
                _ => (None, String::new(), surface.to_owned(), surface.to_owned(), ents.score[i]),
            };
            let (polarity, epistemic, cue) =
                classify_entity_modality(cx.doc, modality.as_ref(), view, span, ents.tokens[i]);
            let item = SearchLinkedEntity {
                entity_index: i as u32,
                span_start,
                span_end,
                entity_kind: ents.kinds.label(ents.kind[i]).to_owned(),
                kb_id,
                provider_id,
                canonical_name,
                matched_alias,
                score,
                polarity: polarity.to_owned(),
                epistemic: epistemic.to_owned(),
                cue,
            };
            match polarity {
                "negated" => negated.push(item),
                "uncertain" => uncertain.push(item),
                _ => affirmed.push(item),
            }
        }
    }

    Ok(SearchEntityPolarities { affirmed, negated, uncertain })
}
