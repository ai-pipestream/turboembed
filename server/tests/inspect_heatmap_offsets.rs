//! Raw shared coverage and the RPC's projected display coordinates.
#[path = "inspect_safety/common.rs"]
#[allow(dead_code)]
mod common;
use common::*;
use std::num::NonZeroU32;
use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{DocumentHeatmap, Tokens};
use turbo_kserve::{api::TURBO_E_CAPACITY, inspect::inspect_document_window, proto::result::OffsetMode};
fn snap(text: &str, at: u32) -> usize {
    (0..=at as usize).rev().find(|&i| text.is_char_boundary(i)).unwrap()
}
fn units(text: &str, at: usize, mode: OffsetMode) -> u32 {
    match mode {
        OffsetMode::Utf8 => at as u32,
        OffsetMode::Utf16 => text[..at].encode_utf16().count() as u32,
        OffsetMode::CodePoint => text[..at].chars().count() as u32,
    }
}
#[test]
fn all_offsets_preserve_raw_bins_scores_counts_and_points() {
    let mut inside_scalar_empty = false;
    let mut collapsed_covered = false;
    for text in ["", "é", "😀", "é😀a"] {
        let mut d = Document::new(text);
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, text.len() as u32));
        d.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
        let bytes = graph(&d);
        for n in [0, 1, 2, 4, 9] {
            let native = DocumentHeatmap::build_instances(&d, ViewId::SOURCE, NonZeroU32::new(n)).unwrap();
            let expected = &native.lanes[0];
            for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
                let mut req = window(bytes.clone(), mode);
                req.heatmap_buckets = n;
                let out = inspect_document_window(&req).unwrap();
                let actual = &out.heatmap_lanes[0];
                assert_eq!(actual.buckets.len(), expected.segments.len());
                if n > 0 {
                    assert_eq!(actual.buckets.len(), n as usize);
                }
                assert_eq!((actual.min_score, actual.max_score), (expected.min_score, expected.max_score));
                for (a, b) in actual.buckets.iter().zip(&expected.segments) {
                    assert_eq!((a.byte_start, a.byte_end), (b.span.start, b.span.end));
                    assert_eq!((a.score, a.count, a.primary_item), (b.score, b.count, b.primary_item));
                    assert_eq!(
                        (a.start, a.end),
                        (units(text, snap(text, b.span.start), mode), units(text, snap(text, b.span.end), mode))
                    );
                    inside_scalar_empty |= a.byte_start == a.byte_end && !text.is_char_boundary(a.byte_start as usize);
                    collapsed_covered |= a.start == a.end && a.count > 0 && a.byte_start < a.byte_end;
                }
            }
        }
    }
    assert!(inside_scalar_empty, "exercise a raw empty bin inside a BMP/astral scalar");
    assert!(collapsed_covered, "snapping display coordinates must not discard raw-bin evidence");
}
#[test]
fn bucket_cap_is_checked_before_graph_import() {
    let mut req = window(b"invalid graph".to_vec(), OffsetMode::Utf16);
    req.heatmap_buckets = 4097;
    assert_eq!(inspect_document_window(&req).unwrap_err().code, TURBO_E_CAPACITY);
}

#[test]
fn failed_and_skipped_sources_remain_metadata_without_heatmap_lanes() {
    let mut d = Document::new("é");
    d.skip(Key::<Tokens>::named("skipped").id(), "no tokens".into(), Provenance::by("test")).unwrap();
    let key = Key::<turbo_analysis_layers::Incivility>::named("failed").id();
    d.fail(
        key,
        turbo_analysis::Error::Invalid { layer: key, detail: "producer failed".into() },
        Provenance::by("test"),
    )
    .unwrap();
    assert!(DocumentHeatmap::build_instances(&d, ViewId::SOURCE, None).unwrap().lanes.is_empty());
    let bytes = graph(&d);
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = inspect_document_window(&window(bytes.clone(), mode)).unwrap();
        assert!(out.heatmap_lanes.is_empty());
        assert_eq!(out.summary.unwrap().layers.len(), 2);
    }
}
