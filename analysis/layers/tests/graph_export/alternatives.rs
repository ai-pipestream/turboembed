use super::*;
use turbo_analysis::{Alternatives, Layer, ModeSet, NBest, SplitMode};

fn lattice() -> Alternatives<Tokens> {
    let mut alt = Alternatives::new(ViewId::SOURCE);
    alt.push_full(Span::new(0, 5), 0, 1, ModeSet::BEST | SplitMode::A, Some(-0.5), None);
    alt.push_full(Span::new(0, 11), 0, 2, ModeSet::C, Some(-1.5), Some("hello world"));
    alt.push(Span::new(6, 11), 1, 1, ModeSet::ALL);
    alt.push(Span::new(11, 12), 2, 1, ModeSet::ALL);
    alt.push(Span::new(13, 16), 3, 1, ModeSet::ALL);
    alt.push(Span::new(17, 20), 4, 1, ModeSet::ALL);
    alt.push(Span::new(20, 21), 5, 1, ModeSet::ALL);
    alt
}

fn nbest() -> NBest<PosTags> {
    let mut nb = NBest::new(tagset("pos", &["NN", "VB", "."]), 6);
    nb.push(-1.0, &[Tag(0), Tag(0), Tag(2), Tag(1), Tag(0), Tag(2)], Some(&[0.9, 0.8, 1.0, 0.7, 0.9, 1.0])).unwrap();
    nb.push(-2.5, &[Tag(0), Tag(1), Tag(2), Tag(1), Tag(0), Tag(2)], Some(&[0.9, 0.2, 1.0, 0.7, 0.9, 1.0])).unwrap();
    nb
}

#[test]
fn alternatives_and_nbest_round_trip() {
    let mut doc = document();
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, lattice(), by.clone()).unwrap();
    doc.add(&Key::DEFAULT, nbest(), by).unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();

    let alt = back.get::<Alternatives<Tokens>>().unwrap();
    assert_eq!(alt.num_positions, 6);
    assert_eq!(alt.len(), 7);
    let compound = alt.get(Ix::new(1));
    assert_eq!((compound.position, compound.position_length), (0, 2));
    assert_eq!(compound.score, Some(-1.5));
    assert_eq!(compound.form, Some("hello world"));
    assert!(compound.mode.contains(SplitMode::C));
    assert_eq!(alt.best_path().len(), 6);

    let nb = back.get::<NBest<PosTags>>().unwrap();
    assert_eq!(nb.candidates(), 2);
    assert_eq!(nb.score_of(0), -1.0);
    assert_eq!(nb.tags_of(1)[1], Tag(1));
    assert_eq!(nb.at(0, Ix::new(3)), (Tag(1), Some(0.7)));

    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

fn write_spans(w: &mut Writer, spans: &[Span]) {
    w.str("").unwrap();
    w.usize(spans.len()).unwrap();
    for s in spans {
        w.usize(1).unwrap();
        w.span(*s);
    }
}

fn u32_col(w: &mut Writer, values: &[u32]) {
    w.usize(values.len()).unwrap();
    for &v in values {
        w.u32(v);
    }
}

fn lattice_body(lengths: &[u32], num_positions: u32, scores: &[f64]) -> Vec<u8> {
    let mut w = Writer::new();
    write_spans(&mut w, &[Span::new(0, 5), Span::new(6, 11)]);
    u32_col(&mut w, &[0, 1]);
    u32_col(&mut w, lengths);
    w.u32(num_positions);
    w.usize(2).unwrap();
    w.u8(1);
    w.u8(1);
    w.u8(1);
    w.usize(scores.len()).unwrap();
    for &s in scores {
        w.f64(s);
    }
    w.u8(0);
    w.finish()
}

fn splice(graph: &mut DocumentGraph, layer_prefix: &str, body: Vec<u8>, len: u32) {
    let at = graph
        .layers
        .iter()
        .position(|l| l.layer.starts_with(layer_prefix))
        .unwrap_or_else(|| panic!("layers: {:?}", graph.layers.iter().map(|l| &l.layer).collect::<Vec<_>>()));
    graph.layers[at].body = Some(body);
    graph.layers[at].len = len;
}

#[test]
fn corrupted_alternatives_are_refused() {
    let mut doc = document();
    doc.add(&Key::DEFAULT, lattice(), Provenance::by("test")).unwrap();
    let base = export(&doc, &codecs()).unwrap();

    for (what, payload) in [
        ("a zero position length", lattice_body(&[0, 1], 2, &[0.0, 0.0])),
        ("an end past the positions", lattice_body(&[1, 5], 2, &[0.0, 0.0])),
        ("a NaN score", lattice_body(&[1, 1], 2, &[f64::NAN, 0.0])),
        ("an infinite score", lattice_body(&[1, 1], 2, &[f64::INFINITY, 0.0])),
        ("no complete path", lattice_body(&[1, 1], 3, &[0.0, 0.0])),
        ("truncated", vec![0]),
    ] {
        let mut graph = base.clone();
        splice(&mut graph, "turbo:alternatives", payload, 2);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }

    // A billion spans claimed by four bytes.
    let mut w = Writer::new();
    w.str("").unwrap();
    w.usize(u32::MAX as usize).unwrap();
    let mut graph = base.clone();
    splice(&mut graph, "turbo:alternatives", w.finish(), 0);
    assert!(import(&graph, &codecs()).is_err(), "a billion alternatives from eight bytes imported");

    // The valid control behind the cases imports.
    let mut graph = base;
    splice(&mut graph, "turbo:alternatives", lattice_body(&[1, 1], 2, &[0.0, 0.0]), 2);
    assert!(import(&graph, &codecs()).is_ok(), "the hand-built lattice control was refused");
}

fn nbest_body(items: usize, scores: &[f64], tags: &[u16], prob: Option<&[f64]>) -> Vec<u8> {
    let mut w = Writer::new();
    put_tagset(&mut w, &tagset("pos", &["NN", "VB", "."])).unwrap();
    w.usize(items).unwrap();
    w.usize(scores.len()).unwrap();
    for &s in scores {
        w.f64(s);
    }
    w.usize(tags.len()).unwrap();
    for &t in tags {
        w.u16(t);
    }
    match prob {
        None => w.u8(0),
        Some(p) => {
            w.u8(1);
            w.usize(p.len()).unwrap();
            for &x in p {
                w.f64(x);
            }
        }
    }
    w.finish()
}

#[test]
fn corrupted_nbest_is_refused() {
    let mut doc = document();
    doc.add(&Key::DEFAULT, nbest(), Provenance::by("test")).unwrap();
    let base = export(&doc, &codecs()).unwrap();
    let six = [0u16, 0, 2, 1, 0, 2];

    for (what, payload) in [
        ("items short of the aligned layer", nbest_body(5, &[-1.0], &six[..5], None)),
        ("scores in increasing order", nbest_body(6, &[-2.5, -1.0], &[0, 0, 2, 1, 0, 2, 0, 0, 2, 1, 0, 2], None)),
        ("a NaN score", nbest_body(6, &[f64::NAN], &six, None)),
        ("a tag outside the tag set", nbest_body(6, &[-1.0], &[9, 0, 2, 1, 0, 2], None)),
        ("tags short of candidates times items", nbest_body(6, &[-1.0, -2.0], &six, None)),
        ("a probability past one", nbest_body(6, &[-1.0], &six, Some(&[1.5, 0.0, 0.0, 0.0, 0.0, 0.0]))),
        ("truncated", vec![0]),
    ] {
        let mut graph = base.clone();
        splice(&mut graph, "turbo:nbest", payload, 6);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }

    let mut graph = base;
    splice(&mut graph, "turbo:nbest", nbest_body(6, &[-1.0], &six, None), 6);
    assert!(import(&graph, &codecs()).is_ok(), "the hand-built nbest control was refused");
}
