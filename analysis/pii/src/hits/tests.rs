use super::*;
use turbo_analysis::{Layer, ViewId};
use turbo_analysis_layers::PiiEvidence;

#[test]
fn order_is_start_then_longer_then_priority_and_repeats_go() {
    let mut h = Hits::default();
    let f = PiiEvidence::Shape;
    h.begin_detector("test", "1");
    h.add(4, 6, PiiKind::Phone, f, |s| s.push('a'));
    h.add(0, 3, PiiKind::Phone, f, |s| s.push('b'));
    h.add(0, 5, PiiKind::Phone, f, |s| s.push('c'));
    h.add(0, 5, PiiKind::Email, f, |s| s.push('d'));
    h.add(0, 5, PiiKind::Email, f, |s| s.push('d'));
    h.add(0, 5, PiiKind::Email, f, |s| s.push('e'));
    let p = h.into_layer(true, ViewId::SOURCE);
    assert_eq!(p.check(), Ok(()));
    let rows: Vec<_> = p.mentions().map(|m| (m.span.start, m.span.end, m.kind, m.normalized.to_owned())).collect();
    assert_eq!(
        rows,
        [
            (0, 5, PiiKind::Email, "d".to_owned()),
            (0, 5, PiiKind::Email, "e".to_owned()),
            (0, 5, PiiKind::Phone, "c".to_owned()),
            (0, 3, PiiKind::Phone, "b".to_owned()),
            (4, 6, PiiKind::Phone, "a".to_owned()),
        ]
    );
}

fn resolved_with(keep_nested: bool, adds: &[(u32, u32, PiiKind, &str)]) -> Vec<(u32, u32, PiiKind, String)> {
    let mut h = Hits::default();
    h.begin_detector("test", "1");
    for &(start, end, kind, value) in adds {
        h.add(start as usize, end as usize, kind, PiiEvidence::Shape, |s| s.push_str(value));
    }
    h.into_layer(keep_nested, ViewId::SOURCE)
        .mentions()
        .map(|m| (m.span.start, m.span.end, m.kind, m.normalized.to_owned()))
        .collect()
}

fn resolved(adds: &[(u32, u32, PiiKind, &str)]) -> Vec<(u32, u32, PiiKind, String)> {
    resolved_with(true, adds)
}

fn row(start: u32, end: u32, kind: PiiKind, value: &str) -> (u32, u32, PiiKind, String) {
    (start, end, kind, value.to_owned())
}

#[test]
fn nested_candidates_drop_by_default_and_same_or_crossing_spans_stay() {
    use PiiKind::*;
    // Prefix, interior and suffix spans inside 2..10 drop; a second
    // kind on 2..10 and a crossing span 8..14 stay.
    let adds = [
        (2, 10, Iban, "iban"),
        (2, 10, Card, "same-span"),
        (2, 6, Phone, "prefix"),
        (4, 8, Phone, "interior"),
        (4, 10, UkNhs, "suffix"),
        (8, 14, Email, "crossing"),
        (9, 12, Phone, "inside-crossing"),
    ];
    assert_eq!(
        resolved_with(false, &adds),
        [row(2, 10, Iban, "iban"), row(2, 10, Card, "same-span"), row(8, 14, Email, "crossing")]
    );
}

#[test]
fn composite_orders() {
    use PiiKind::*;
    // Crossing candidates, in either order found.
    for adds in [[(4, 12, Email, "e"), (0, 8, Card, "c")], [(0, 8, Card, "c"), (4, 12, Email, "e")]] {
        assert_eq!(resolved(&adds), [row(0, 8, Card, "c"), row(4, 12, Email, "e")]);
    }
    assert_eq!(resolved(&[(2, 6, Phone, "p"), (2, 10, Card, "c")]), [row(2, 10, Card, "c"), row(2, 6, Phone, "p")]);
    // The priority decides an exact span tie, in either order found.
    for adds in [[(2, 10, Phone, "p"), (2, 10, Card, "c")], [(2, 10, Card, "c"), (2, 10, Phone, "p")]] {
        assert_eq!(resolved(&adds), [row(2, 10, Card, "c"), row(2, 10, Phone, "p")]);
    }
    // Equal kinds keep the order found.
    assert_eq!(resolved(&[(0, 5, Email, "b"), (0, 5, Email, "a")]), [row(0, 5, Email, "b"), row(0, 5, Email, "a")]);
    assert_eq!(resolved(&[(0, 4, Card, "c"), (4, 8, Phone, "p")]), [row(0, 4, Card, "c"), row(4, 8, Phone, "p")]);
    assert_eq!(
        resolved(&[(0, 6, Card, "a"), (10, 16, Card, "b"), (4, 12, Phone, "p"), (5, 7, Ipv4, "i")]),
        [row(0, 6, Card, "a"), row(4, 12, Phone, "p"), row(5, 7, Ipv4, "i"), row(10, 16, Card, "b")]
    );
    assert_eq!(resolved(&[(9, 25, Email, "x"), (9, 25, Email, "x")]), [row(9, 25, Email, "x")]);
    // Every pair of kinds on one span: the smaller priority first.
    let mut by_priority: Vec<PiiKind> = PiiKind::builtins().collect();
    by_priority.sort_by_key(|k| k.priority());
    for (i, &a) in by_priority.iter().enumerate() {
        for &b in &by_priority[i + 1..] {
            let want = [row(0, 5, a, "higher"), row(0, 5, b, "lower")];
            assert_eq!(resolved(&[(0, 5, a, "higher"), (0, 5, b, "lower")]), want);
            assert_eq!(resolved(&[(0, 5, b, "lower"), (0, 5, a, "higher")]), want);
        }
    }
}
