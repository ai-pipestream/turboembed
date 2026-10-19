//! Regressions from review: projected kinds and scores where flagged
//! blocks touch or alternate, and malformed inputs handed to the steps
//! directly, which are refused instead of panicking.

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Boilerplate, BoilerplateKind, Sentences, Structure};
use turbo_group::{BoilerplateDetector, BoilerplateProjector, NearDuplicates, Sketcher, Sketches, Thresholds};

const TEXT: &str = "abcdefghijklmnopqrstuvwxyz";

/// Sibling paragraphs at `spans`, each with one fragment set.
fn blocks(spans: &[&[Span]]) -> Structure {
    let mut st =
        Structure { spans: Spans::new(ViewId::SOURCE), kind: Vec::new(), parent: Vec::new(), depth: Vec::new() };
    for fragments in spans {
        st.spans.push_fragments(fragments);
        st.kind.push(Block::Paragraph);
        st.parent.push(None);
        st.depth.push(0);
    }
    st
}

fn flags(rows: &[(BoilerplateKind, f32)]) -> Boilerplate<Structure> {
    Boilerplate::new(rows.iter().map(|r| Some(r.0)).collect(), rows.iter().map(|r| r.1).collect())
}

fn sentences(spans: &[(u32, u32)]) -> Sentences {
    let mut s = Spans::new(ViewId::SOURCE);
    spans.iter().for_each(|&(a, b)| _ = s.push(Span::new(a, b)));
    Sentences { spans: s }
}

fn project(st: &Structure, f: &Boilerplate<Structure>, to: &Sentences) -> Boilerplate<Sentences> {
    BoilerplateProjector::<Sentences>::default().annotate((st, f, to)).unwrap().0
}

#[test]
fn a_sentence_after_a_touching_block_takes_its_own_blocks_kind_and_score() {
    use BoilerplateKind::*;
    let st = blocks(&[&[Span::new(0, 1)], &[Span::new(1, 11)]]);
    let b = project(&st, &flags(&[(Header, 0.2), (Footer, 0.9)]), &sentences(&[(1, 11), (0, 1)]));
    assert_eq!(b.kind, [Some(Footer), Some(Header)]);
    assert_eq!(b.score, [0.9, 0.2]);
}

#[test]
fn the_projected_kind_is_the_one_covering_the_most_bytes_in_all() {
    use BoilerplateKind::*;
    let st = blocks(&[&[Span::new(0, 4)], &[Span::new(5, 10)], &[Span::new(11, 15)]]);
    let f = flags(&[(Footer, 1.0), (Header, 1.0), (Footer, 1.0)]);
    let b = project(&st, &f, &sentences(&[(0, 15)]));
    assert_eq!(b.kind, [Some(Footer)], "Footer covers 8 bytes, Header 5");

    // The same across the fragments of one target: the second fragment
    // holds 5 Header bytes and 4 Footer bytes, the first 4 Footer bytes.
    let target = blocks(&[&[Span::new(0, 4), Span::new(5, 15)]]);
    let b = BoilerplateProjector::<Structure>::default().annotate((&st, &f, &target)).unwrap().0;
    assert_eq!(b.kind, [Some(Footer)]);
}

#[test]
fn where_flagged_blocks_overlap_the_innermost_decides() {
    use BoilerplateKind::*;
    // A navigation list holding a footer block.
    let mut st = blocks(&[&[Span::new(0, 20)], &[Span::new(5, 10)]]);
    st.parent[1] = Some(Ix::new(0));
    let b = project(&st, &flags(&[(Navigation, 0.5), (Footer, 1.0)]), &sentences(&[(0, 5), (5, 10), (10, 20)]));
    assert_eq!(b.kind, [Some(Navigation), Some(Footer), Some(Navigation)]);
    assert_eq!(b.score, [0.5, 1.0, 0.5]);
}

#[test]
fn misaligned_flags_are_refused_by_the_projector() {
    let st = blocks(&[&[Span::new(0, 4)]]);
    let f = flags(&[(BoilerplateKind::Footer, 1.0), (BoilerplateKind::Footer, 1.0)]);
    let r = BoilerplateProjector::<Sentences>::default().annotate((&st, &f, &sentences(&[(0, 4)])));
    assert!(r.unwrap_err().to_string().contains("2 boilerplate flags for 1 items"));
    let mut short = flags(&[(BoilerplateKind::Footer, 1.0)]);
    short.score.clear();
    let r = BoilerplateProjector::<Sentences>::default().annotate((&st, &short, &sentences(&[(0, 4)])));
    assert!(r.unwrap_err().to_string().contains("0 scores for 1 items"));
}

/// Sketches of the blocks of `TEXT`, then cut short in one column.
fn broken_sketches(st: &Structure) -> Vec<Sketches<Structure>> {
    let doc = Document::new(TEXT);
    let good = Sketcher::<Structure>::default().annotate((doc.get::<View>().unwrap(), st)).unwrap().0;
    let mut out = Vec::new();
    for cut in 0..4 {
        let mut s = good.clone();
        match cut {
            0 => _ = s.exact.pop(),
            1 => _ = s.shape.pop(),
            2 => _ = s.simhash.pop(),
            _ => s.minhash.truncate(s.minhash.len() - 1),
        }
        out.push(s);
    }
    out
}

#[test]
fn sketches_with_short_columns_are_refused_not_indexed() {
    let st = blocks(&[&[Span::new(0, 4)], &[Span::new(5, 10)]]);
    for s in broken_sketches(&st) {
        assert!(NearDuplicates::<Structure>::default().annotate((&st, &s)).is_err());
        assert!(BoilerplateDetector::<Structure>::default().annotate((&st, &s)).is_err());
        assert!(turbo_group::cross_duplicates(&[(&st, &s)], &Thresholds::default()).is_err());
    }
}
