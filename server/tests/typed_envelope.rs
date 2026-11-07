//! The result envelope's view edit chain (issue #1907): a consumer that
//! rebuilds the views from the wire chain alone maps every span between
//! every pair of views exactly as `Document::map_span` does, on chained
//! views with combining marks, emoji ZWJ sequences, a ligature expanded
//! across a mention boundary, insertion and deletion anchors, unequal
//! replacements and sibling views; and the run encoding decodes to the
//! native edit list byte for byte.

#[path = "typed_envelope/fixtures.rs"]
mod fixtures;

use turbo_analysis::{Alignment, Document, Edit, Span, ViewBuilder, ViewId};
use turbo_kserve::api::TURBO_E_CAPACITY;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{EditRun, InlineMapping, mapping_payload};
use turbo_kserve::results::MAX_MAPPING_EDITS;
use turbo_kserve::results::typed::chain::{MAPPING_SCHEMA_VERSION, WireChain, decode_runs, encode_runs};
use turbo_kserve::results::typed::document_to_typed;

use fixtures::*;

/// Every span on character boundaries of `text`, the empty ones included.
fn all_spans(text: &str) -> Vec<Span> {
    let bounds: Vec<u32> =
        text.char_indices().map(|(i, _)| i as u32).chain(std::iter::once(text.len() as u32)).collect();
    let mut out = Vec::new();
    for (i, &s) in bounds.iter().enumerate() {
        for &e in &bounds[i..] {
            out.push(Span::new(s, e));
        }
    }
    out
}

/// The chain a consumer rebuilds from `doc`'s typed result, checked
/// against the document on every view pair and every span.
fn check_chain(name: &str, doc: &Document) -> WireChain {
    let typed = document_to_typed(name, doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let chain = WireChain::from_payload(typed.mapping.as_ref()).unwrap();
    assert_eq!(chain.views().len(), doc.view_count(), "{name}: every view is in the chain");
    for v in chain.views() {
        assert_eq!(v.text_len as usize, doc.text(v.id).len(), "{name}: view {} length", v.id.0);
        assert_eq!(&v.text_digest, doc.text_digest(v.id).as_bytes(), "{name}: view {} digest", v.id.0);
    }
    let n = doc.view_count() as u16;
    for from in 0..n {
        let spans = all_spans(doc.text(ViewId(from)));
        for to in 0..n {
            for &s in &spans {
                let native = doc.map_span(ViewId(from), ViewId(to), s).unwrap();
                let wire = chain.map_span(ViewId(from), ViewId(to), s).unwrap();
                assert_eq!(wire, native, "{name}: span {s:?} from view {from} to view {to}");
            }
        }
    }
    chain
}

#[test]
fn chained_views_map_from_the_wire_exactly_as_the_document_does() {
    for (name, doc) in fixture_documents() {
        check_chain(name, &doc);
    }
}

#[test]
fn the_run_encoding_decodes_to_the_native_edit_list_byte_for_byte() {
    for (name, doc) in fixture_documents() {
        let typed = document_to_typed(name, &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
        let Some(mapping_payload::Payload::Inline(inline)) = typed.mapping.as_ref().and_then(|m| m.payload.as_ref())
        else {
            panic!("{name}: no inline mapping");
        };
        for (id, wire) in inline.views.iter().enumerate() {
            let native = view_edits(&doc, ViewId(id as u16));
            let decoded = decode_runs("view", &wire.edits).unwrap();
            assert_eq!(decoded, native, "{name}: view {id} decodes to the native edits");
            assert_eq!(wire.edit_count as usize, native.len());
            assert_eq!(wire.edits, encode_runs(&native));
        }
    }
}

#[test]
fn equal_width_edits_become_one_run_and_mixed_widths_do_not_merge() {
    // A masked view's one edit per character: five ASCII letters masked
    // one byte each is one run of five.
    let per_char: Vec<Edit> =
        (0..5).map(|i| Edit { parent: Span::new(i, i + 1), child: Span::new(i, i + 1) }).collect();
    let runs = encode_runs(&per_char);
    assert_eq!(runs, vec![EditRun { parent_start: 0, parent_width: 1, child_start: 0, child_width: 1, count: 5 }]);
    assert_eq!(decode_runs("view", &runs).unwrap(), per_char);

    // "é" (2 bytes) then "a" masked to one byte each: widths differ, two runs.
    let mixed = vec![
        Edit { parent: Span::new(0, 2), child: Span::new(0, 1) },
        Edit { parent: Span::new(2, 3), child: Span::new(1, 2) },
    ];
    let runs = encode_runs(&mixed);
    assert_eq!(runs.len(), 2);
    assert_eq!(decode_runs("view", &runs).unwrap(), mixed);

    // Equal widths with an unchanged byte between them do not merge.
    let gapped = vec![
        Edit { parent: Span::new(0, 1), child: Span::new(0, 1) },
        Edit { parent: Span::new(2, 3), child: Span::new(2, 3) },
    ];
    assert_eq!(encode_runs(&gapped).len(), 2);
    assert_eq!(decode_runs("view", &encode_runs(&gapped)).unwrap(), gapped);

    // Deletions and insertions keep their empty side.
    let del_ins = vec![
        Edit { parent: Span::new(1, 2), child: Span::new(1, 1) },
        Edit { parent: Span::new(2, 3), child: Span::new(1, 1) },
        Edit { parent: Span::new(5, 5), child: Span::new(3, 4) },
    ];
    let runs = encode_runs(&del_ins);
    assert_eq!(runs[0].count, 2, "two adjacent one-byte deletions are one run");
    assert_eq!(decode_runs("view", &runs).unwrap(), del_ins);
}

#[test]
fn a_run_that_changes_nothing_or_leaves_u32_is_refused() {
    let nothing = [EditRun { parent_start: 0, parent_width: 0, child_start: 0, child_width: 0, count: 1 }];
    assert!(decode_runs("view", &nothing).unwrap_err().message.contains("changes nothing"));
    let empty = [EditRun { parent_start: 0, parent_width: 1, child_start: 0, child_width: 1, count: 0 }];
    assert!(decode_runs("view", &empty).unwrap_err().message.contains("no edits"));
    let overflow = [EditRun { parent_start: u32::MAX - 1, parent_width: 1, child_start: 0, child_width: 1, count: 3 }];
    assert!(decode_runs("view", &overflow).unwrap_err().message.contains("leaves u32"));
}

#[test]
fn a_mapping_over_max_mapping_edits_is_refused_not_cut() {
    let over = (MAX_MAPPING_EDITS + 1) as u32;
    let runs = vec![EditRun { parent_start: 0, parent_width: 1, child_start: 0, child_width: 1, count: over }];
    let err = decode_runs("view", &runs).unwrap_err();
    assert_eq!(err.code, TURBO_E_CAPACITY);
    assert!(err.message.contains("max_mapping_edits"), "{}", err.message);

    // Spread over two views the total is still held to the limit.
    let half = (MAX_MAPPING_EDITS / 2 + 1) as u32;
    let views = (0..3)
        .map(|i| turbo_kserve::proto::result::ViewChain {
            view: i,
            instance: if i == 0 { String::new() } else { format!("v{i}") },
            parent: if i == 0 { None } else { Some(0) },
            text_digest: vec![0; 32],
            text_len: half,
            edit_count: if i == 0 { 0 } else { half },
            edits: if i == 0 {
                Vec::new()
            } else {
                vec![EditRun { parent_start: 0, parent_width: 1, child_start: 0, child_width: 1, count: half }]
            },
        })
        .collect();
    let err =
        WireChain::from_inline(&InlineMapping { mapping_schema_version: MAPPING_SCHEMA_VERSION, views }).unwrap_err();
    assert_eq!(err.code, TURBO_E_CAPACITY);
}

#[test]
fn a_document_over_max_mapping_edits_is_refused_on_the_producing_path() {
    let n = (MAX_MAPPING_EDITS + 1) as usize;
    let text = "a".repeat(n);
    let mut doc = Document::new(text.clone());
    let source = doc.get::<turbo_analysis::View>().unwrap().clone();
    let mut b = ViewBuilder::new(&text);
    for i in 0..n {
        b.replace(i + 1, "b");
    }
    let (child, alignment) = b.finish();
    assert_eq!(alignment.edits().len(), n);
    doc.add(
        &turbo_analysis::Key::<turbo_analysis::View>::named("masked"),
        turbo_analysis::View::derived(&source, child, alignment),
        turbo_analysis::Provenance::by("test:mask"),
    )
    .unwrap();
    let err = document_to_typed("over", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap_err();
    assert_eq!(err.code, TURBO_E_CAPACITY);
    assert!(err.message.contains("max_mapping_edits"), "{}", err.message);
}

#[test]
fn a_chain_that_is_not_a_view_tree_is_refused() {
    let doc = sibling_views();
    let typed = document_to_typed("tree", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let Some(mapping_payload::Payload::Inline(inline)) = typed.mapping.as_ref().and_then(|m| m.payload.as_ref()) else {
        panic!("no inline mapping");
    };
    let refused = |edit: &dyn Fn(&mut InlineMapping)| {
        let mut m = inline.clone();
        edit(&mut m);
        WireChain::from_inline(&m).unwrap_err().message
    };
    assert!(refused(&|m| m.mapping_schema_version = 2).contains("schema version"));
    assert!(refused(&|m| m.views[1].parent = None).contains("no parent"));
    assert!(refused(&|m| m.views[1].parent = Some(2)).contains("not before"));
    assert!(refused(&|m| m.views[0].parent = Some(0)).contains("source view has a parent"));
    assert!(refused(&|m| m.views.swap(1, 2)).contains("id order"));
    assert!(refused(&|m| m.views[1].text_digest.pop().map(|_| ()).unwrap()).contains("not 32"));
    assert!(refused(&|m| m.views[1].edit_count += 1).contains("runs hold"));
    assert!(refused(&|m| m.views[1].text_len += 1).contains("do not turn"));
    assert!(refused(&|m| m.views[1].edits[0].child_start = 1000).contains("edit 0"));
    assert!(refused(&|m| m.views[1].edits.last_mut().unwrap().child_width = 1000).contains("past"));
    assert!(refused(&|m| m.views.clear()).contains("no views"));
    // Instance names: the source is the unnamed instance, derived views
    // are named, and no two views share a name.
    assert!(refused(&|m| m.views[0].instance = "ascii".into()).contains("source view is instance"));
    assert!(refused(&|m| m.views[1].instance = String::new()).contains("no instance name"));
    assert!(refused(&|m| m.views[2].instance = "ascii".into()).contains("duplicate mapping view instance"));
    let mut dup = inline.clone();
    dup.views[2].instance = "ascii".into();
    assert!(WireChain::from_inline(&dup).is_err(), "by_instance cannot be left to pick between two views");
    // The same tampering refuses the whole typed document.
    let mut bad = typed.clone();
    if let Some(mapping_payload::Payload::Inline(i)) = bad.mapping.as_mut().and_then(|m| m.payload.as_mut()) {
        i.views[1].edit_count += 1;
    }
    assert!(turbo_kserve::results::typed::round_trip_typed_document(&bad).is_err());
    let mut none = typed;
    none.mapping = None;
    assert!(turbo_kserve::results::typed::round_trip_typed_document(&none).unwrap_err().message.contains("inline"));
}

#[test]
fn edit_coordinates_stay_utf8_whatever_the_display_offsets_are() {
    let doc = emoji_zwj_masked();
    let utf8 = document_to_typed("d", &doc, ViewId(1), OffsetMode::Utf8).unwrap();
    let utf16 = document_to_typed("d", &doc, ViewId(1), OffsetMode::Utf16).unwrap();
    let cp = document_to_typed("d", &doc, ViewId(1), OffsetMode::CodePoint).unwrap();
    assert_eq!(utf8.mapping, utf16.mapping);
    assert_eq!(utf8.mapping, cp.mapping);
    // The masked token is ASCII in every unit; the emoji token is not.
    assert_eq!(utf8.layers[0].annotations[0].spans, utf16.layers[0].annotations[0].spans);
    assert_ne!(utf8.layers[0].annotations[1].spans, utf16.layers[0].annotations[1].spans);
    assert_ne!(utf8.layers[0].annotations[1].spans, cp.layers[0].annotations[1].spans);
    let chain = WireChain::from_payload(utf8.mapping.as_ref()).unwrap();
    let edits = view_edits(&doc, ViewId(1));
    assert_eq!(chain.view(ViewId(1)).unwrap().alignment, Alignment::from_edits(edits).unwrap());
}
