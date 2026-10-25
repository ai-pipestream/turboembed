//! Checked bucket storage with fallible count-controlled reservations.
use std::alloc::Layout;

use super::{
    Document, Error, HeatmapLaneKind, HeatmapSegment, Item, KeyId, Span, ViewId, aggregate, check_items, invalid,
};

// Check array sizes before reservation. On 32-bit targets, u32::MAX
// buckets would need an unrepresentable N + 1 edges.
fn checked_sizes(key: KeyId, n: usize) -> Result<usize, Error> {
    let edge_count = n.checked_add(1).ok_or_else(|| invalid(key, "heatmap bucket edge count exceeds usize".into()))?;
    for layout in [
        Layout::array::<HeatmapSegment>(n),
        Layout::array::<u32>(edge_count),
        Layout::array::<Vec<usize>>(n),
        Layout::array::<u32>(n),
    ] {
        layout.map_err(|_| invalid(key, "heatmap bucket storage exceeds the target allocation layout".into()))?;
    }
    Ok(edge_count)
}

fn reserved<T>(key: KeyId, count: usize, what: &str) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|e| invalid(key, format!("cannot reserve {count} heatmap {what}: {e}")))?;
    Ok(values)
}

pub(super) fn bucket_segments(
    doc: &Document,
    view: ViewId,
    key: KeyId,
    text_len: u32,
    buckets: u32,
    kind: HeatmapLaneKind,
    items: &[Item],
) -> Result<Vec<HeatmapSegment>, Error> {
    check_items(doc, view, key, text_len, items)?;
    let n = usize::try_from(buckets).map_err(|_| invalid(key, "heatmap bucket count exceeds usize".into()))?;
    let edge_count = checked_sizes(key, n)?;
    let edge = |i: usize| -> u32 { (i as u64 * text_len as u64 / buckets as u64) as u32 };
    let mut segments = reserved(key, n, "bucket segments")?;
    if items.is_empty() {
        segments.extend((0..n).map(|i| HeatmapSegment {
            span: Span { start: edge(i), end: edge(i + 1) },
            score: 0.0,
            count: 0,
            primary_item: None,
        }));
        return Ok(segments);
    }
    let mut edges = reserved(key, edge_count, "bucket edges")?;
    let mut covering: Vec<Vec<usize>> = reserved(key, n, "bucket membership lists")?;
    let mut seen = reserved(key, n, "bucket item markers")?;
    edges.extend((0..edge_count).map(edge));
    covering.resize_with(n, Vec::new);
    seen.resize(n, 0u32);
    for (ix, it) in items.iter().enumerate() {
        let mark = ix as u32 + 1;
        for &(s, e) in &it.frags {
            // Use the emitted rounded edges for positive overlap and points;
            // positive-length fragments must not cover empty bins.
            let (first, end) = if s == e {
                let at = if text_len == 0 {
                    0
                } else if s == text_len {
                    n - 1
                } else {
                    edges[1..].partition_point(|&end| end <= s)
                };
                (at, at + 1)
            } else {
                (edges[1..].partition_point(|&end| end <= s), edges[..n].partition_point(|&start| start < e))
            };
            for (b, cov) in covering.iter_mut().enumerate().take(end).skip(first) {
                if (s == e || edges[b] < edges[b + 1]) && seen[b] != mark {
                    seen[b] = mark;
                    cov.try_reserve(1)
                        .map_err(|e| invalid(key, format!("cannot reserve heatmap bucket membership: {e}")))?;
                    cov.push(ix);
                }
            }
        }
    }
    segments.extend((0..n).map(|i| {
        let (score, count, primary_item) = aggregate(kind, items, &covering[i]);
        HeatmapSegment { span: Span { start: edges[i], end: edges[i + 1] }, score, count, primary_item }
    }));
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tokens;
    use turbo_analysis::Key;

    fn key() -> KeyId {
        Key::<Tokens>::DEFAULT.id()
    }

    #[test]
    fn bucket_sizes_check_edge_overflow_and_native_byte_capacity() {
        assert!(checked_sizes(key(), usize::MAX).is_err());
        let element_bytes =
            [size_of::<HeatmapSegment>(), size_of::<Vec<usize>>(), size_of::<u32>()].into_iter().max().unwrap();
        let max = isize::MAX as usize / element_bytes;
        assert_eq!(checked_sizes(key(), max).unwrap(), max + 1);
        assert!(matches!(checked_sizes(key(), max + 1), Err(Error::Invalid { layer, .. }) if layer == key()));
    }

    #[test]
    fn u32_max_layout_is_checked_without_allocating_bucket_arrays() {
        let n = usize::try_from(u32::MAX).unwrap();
        let result = checked_sizes(key(), n);
        if usize::BITS == 32 {
            assert!(matches!(result, Err(Error::Invalid { layer, .. }) if layer == key()));
        } else {
            assert_eq!(result.unwrap(), n + 1);
        }
    }

    #[test]
    fn impossible_reservation_returns_a_layer_named_error_without_oom() {
        // This byte layout cannot reach the allocator. Test the reservation
        // error mapping without requesting a large representable allocation.
        assert!(Layout::array::<HeatmapSegment>(usize::MAX).is_err());
        let error = reserved::<HeatmapSegment>(key(), usize::MAX, "bucket segments").unwrap_err();
        match error {
            Error::Invalid { layer, detail } => {
                assert_eq!(layer, key());
                assert!(detail.contains("cannot reserve") && detail.contains("bucket segments"), "{detail}");
            }
            other => panic!("{other:?}"),
        }
    }
}
