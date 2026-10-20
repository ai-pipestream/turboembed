//! The `Categories` layer on its own invariants: the document item,
//! per-unit items, the abstention, and the selected label sets.

use turbo_analysis::{Anchor, Layer, Tag, TagSet};

use crate::categories::Categories;
use crate::{Chunks, Sentences, Structure};

/// The first highest score, as OpenNLP's `getBestCategory`.
fn first_best(scores: &[f64]) -> Tag {
    let mut best = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    Tag(best as u16)
}

fn document(names: &[&str], scores: &[f64]) -> Categories {
    Categories::document(TagSet::new("test", names), first_best(scores), scores)
}

#[test]
fn the_document_has_one_item_and_ranks_its_categories() {
    let c = document(&["weather", "sports", "cooking"], &[0.25, 0.5, 0.25]);
    assert_eq!(c.len(), 1);
    assert_eq!(c.best(), Some(("sports", 0.5)));
    assert_eq!(c.ranked(), [("sports", 0.5), ("weather", 0.25), ("cooking", 0.25)]);
    assert_eq!(c.get(0).unwrap().score_of("cooking"), Some(0.25));
    assert_eq!(c.get(0).unwrap().score_of("finance"), None);
    assert_eq!(c.check(), Ok(()));

    let empty: Categories = Categories::new(TagSet::new("test", &["a"]));
    assert_eq!(empty.best(), None);
    assert!(empty.ranked().is_empty());
    assert_eq!(empty.check(), Ok(()));
}

#[test]
fn names_and_anchors() {
    assert_eq!(<Categories>::NAME.to_string(), "turbo:categories");
    assert_eq!(<Categories>::anchor(), Anchor::Document);
    assert_eq!(Categories::<Sentences>::NAME.to_string(), "turbo:categories<opennlp:sentences>");
    assert_eq!(Categories::<Sentences>::anchor(), Anchor::aligned::<Sentences>());
    assert_eq!(Categories::<Structure>::NAME.to_string(), "turbo:categories<turbo:structure>");
    assert_eq!(Categories::<Chunks>::NAME.to_string(), "turbo:categories<turbo:chunks>");
    assert_eq!(Categories::<Chunks>::anchor(), Anchor::aligned::<Chunks>());
}

#[test]
fn an_item_per_unit() {
    let mut c = Categories::<Sentences>::new(TagSet::new("test", &["a", "b"]));
    c.push(Tag(1), &[0.25, 0.75]);
    c.push(Tag(0), &[0.5, 0.5]);
    assert_eq!(c.len(), 2);
    assert_eq!(c.best_category(0), "b");
    assert_eq!(c.row_scores(1), [0.5, 0.5]);
    assert_eq!(c.score_of(0, "a"), Some(0.25));
    let rows: Vec<_> = c.iter().map(|r| (r.best, r.ranked())).collect();
    assert_eq!(rows, [("b", vec![("b", 0.75), ("a", 0.25)]), ("a", vec![("a", 0.5), ("b", 0.5)])]);
    assert!(c.get(2).is_none());
    assert_eq!(c.check(), Ok(()));
}

#[test]
fn check_refuses_inconsistent_layers() {
    let mut c = document(&["a", "b"], &[0.5, 0.5]);
    c.push(Tag(0), &[0.5, 0.5]);
    assert_eq!(c.check(), Err("2 items in the document's categories, at most 1".into()));

    let mut c = Categories::<Sentences>::new(TagSet::new("test", &["a", "b"]));
    c.category.push(Tag(0));
    c.selected.push(Tag(0));
    c.selected_end.push(1);
    c.scores.push(1.0);
    assert_eq!(c.check(), Err("1 items of 2 categories and 1 scores".into()));
    c.scores.push(0.0);
    c.category[0] = Tag(2);
    assert_eq!(c.check(), Err("a best category outside test".into()));
    c.category[0] = Tag(0);
    c.scores[1] = f64::NAN;
    assert_eq!(c.check(), Err("a score that is negative or NaN".into()));

    let mut c = Categories::<Sentences>::new(TagSet::new("test", &[]));
    c.category.push(Tag(0));
    assert_eq!(c.check(), Err("no categories to score".into()));
}

#[test]
fn an_item_can_abstain_with_no_best_category() {
    // A multi-label categorizer whose no label cleared its threshold
    // pushes Tag::NONE: the item is scored, nothing is predicted.
    let mut c = document(&["a", "b"], &[0.1, 0.2]);
    c.category[0] = Tag::NONE;
    c.selected.clear();
    c.selected_end[0] = 0;
    assert_eq!(c.check(), Ok(()));
    assert_eq!(c.best(), None);
    assert_eq!(c.best_category(0), "");
    let row = c.get(0).unwrap();
    assert_eq!(row.best_tag, Tag::NONE);
    assert_eq!(row.best, "");
    assert_eq!(row.best_score(), 0.0);
    // The scores are still the item's: ranked lists every category.
    assert_eq!(row.ranked(), [("b", 0.2), ("a", 0.1)]);

    let mut spans = Categories::<Sentences>::new(TagSet::new("test", &["a", "b"]));
    spans.push(Tag(1), &[0.1, 0.9]);
    spans.push(Tag::NONE, &[0.3, 0.4]);
    assert_eq!(spans.check(), Ok(()));
    assert_eq!(spans.best_category(1), "");
    assert_eq!(spans.get(1).unwrap().best_score(), 0.0);
    assert_eq!(spans.best_category(0), "b");

    // Tag::NONE still names no category: a real tag must be in the set.
    let mut c = document(&["a", "b"], &[0.5, 0.5]);
    c.category[0] = Tag(2);
    assert_eq!(c.check(), Err("a best category outside test".into()));
}

#[test]
fn push_selects_the_best_and_abstention_selects_nothing() {
    let mut c = Categories::<Sentences>::new(TagSet::new("test", &["a", "b"]));
    c.push(Tag(1), &[0.4, 0.9]);
    c.push(Tag::NONE, &[0.2, 0.3]);
    assert_eq!(c.selected(0), [Tag(1)]);
    assert!(c.selected(1).is_empty());
    assert_eq!(c.get(0).unwrap().selected, [Tag(1)]);
    assert!(c.get(1).unwrap().selected.is_empty());
    assert_eq!(c.check(), Ok(()));
}

#[test]
fn push_selected_records_the_full_label_set() {
    // The multi-label regression: same best, same scores, different
    // selected sets must stay distinguishable on the layer.
    let cats = || TagSet::new("test", &["a", "b"]);
    let both = Categories::document_selected(cats(), Tag(0), &[Tag(0), Tag(1)], &[0.8, 0.7]);
    let one = Categories::document_selected(cats(), Tag(0), &[Tag(0)], &[0.8, 0.7]);
    assert_eq!(both.selected(0), [Tag(0), Tag(1)]);
    assert_eq!(one.selected(0), [Tag(0)]);
    assert_ne!(both, one, "the selected set is part of the layer's value");
    assert_ne!(both.get(0).unwrap(), one.get(0).unwrap());
    assert_eq!(both.check(), Ok(()));
    assert_eq!(one.check(), Ok(()));

    // Abstention is the empty selected set.
    let none = Categories::document_selected(cats(), Tag::NONE, &[], &[0.8, 0.7]);
    assert!(none.selected(0).is_empty());
    assert_eq!(none.check(), Ok(()));
}

#[test]
fn check_refuses_inconsistent_selected_sets() {
    let cats = || TagSet::new("test", &["a", "b"]);
    // A selected label outside the tag set.
    let c = Categories::document_selected(cats(), Tag(0), &[Tag(0), Tag(2)], &[0.8, 0.7]);
    assert_eq!(c.check(), Err("a selected label of item 0 outside test".into()));

    // Not in category order.
    let c = Categories::document_selected(cats(), Tag(0), &[Tag(1), Tag(0)], &[0.8, 0.7]);
    assert_eq!(c.check(), Err("the selected labels of item 0 not in category order".into()));

    // Abstained but holding labels.
    let c = Categories::document_selected(cats(), Tag::NONE, &[Tag(0)], &[0.8, 0.7]);
    assert_eq!(c.check(), Err("item 0 abstained but selected labels".into()));

    // The best category not among the selected labels.
    let c = Categories::document_selected(cats(), Tag(0), &[Tag(1)], &[0.8, 0.7]);
    assert_eq!(c.check(), Err("item 0's best category not among its selected labels".into()));

    // A column shorter than the items.
    let mut c = Categories::document(cats(), Tag(0), &[0.8, 0.7]);
    c.selected_end.clear();
    assert_eq!(c.check(), Err("1 items and 0 selected row ends".into()));

    // A row end past the selected labels.
    let mut c = Categories::document(cats(), Tag(0), &[0.8, 0.7]);
    c.selected_end[0] = 2;
    assert_eq!(c.check(), Err("selected row 0 ends past the selected labels".into()));

    // Selected labels past the last row end.
    let mut c = Categories::document(cats(), Tag(0), &[0.8, 0.7]);
    c.selected.push(Tag(1));
    assert_eq!(c.check(), Err("1 selected labels past the last row end".into()));
}

#[test]
fn ten_thousand_pushes_keep_the_reserved_capacity() {
    // The allocation regression: `selected` used to be one Vec per item,
    // so pushing 10,000 items made 10,000 heap allocations. The CSR pair
    // is reserved with the items; pushing appends, never reallocating.
    let mut c = Categories::<Sentences>::with_capacity(TagSet::new("test", &["a", "b"]), 10_000);
    let selected_capacity = c.selected.capacity();
    let ends_capacity = c.selected_end.capacity();
    let selected_ptr = c.selected.as_ptr();
    let ends_ptr = c.selected_end.as_ptr();
    for i in 0..10_000 {
        let best = if i % 2 == 0 { Tag(0) } else { Tag(1) };
        c.push(best, &[0.4, 0.6]);
    }
    assert_eq!(c.selected.capacity(), selected_capacity, "selected reallocated");
    assert_eq!(c.selected_end.capacity(), ends_capacity, "selected_end reallocated");
    assert_eq!(c.selected.as_ptr(), selected_ptr, "selected moved");
    assert_eq!(c.selected_end.as_ptr(), ends_ptr, "selected_end moved");
    assert_eq!(c.len(), 10_000);
    assert_eq!(c.selected.len(), 10_000);
    assert_eq!(c.selected_end.len(), 10_000);
    assert_eq!(c.selected(0), [Tag(0)]);
    assert_eq!(c.selected(9_999), [Tag(1)]);
    assert_eq!(c.get(5_432).unwrap().selected, [Tag(0)]);
    assert!(c.check().is_ok());

    // Multi-label pushes grow the CSR columns correctly too.
    let mut c = Categories::<Sentences>::with_capacity(TagSet::new("test", &["a", "b"]), 2);
    c.push_selected(Tag(0), &[Tag(0), Tag(1)], &[0.7, 0.6]);
    c.push(Tag::NONE, &[0.1, 0.2]);
    assert_eq!(c.selected(0), [Tag(0), Tag(1)]);
    assert!(c.selected(1).is_empty());
    assert_eq!(c.selected_end, [2, 2]);
    assert!(c.check().is_ok());
}
