use turbo::turbo_token_range;
use turbo_analysis::*;
use turbo_analysis_embed::window::{Which, pieces_of};
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Chunks, Levels, Sentences, Subwords, Unit, Vectors};

use crate::common::*;

/// Twelve sentences in rows of 24 tokens: several windows, run two rows
/// and three ranges at a time, so a run ends inside a window's ranges and
/// the window's row goes again in the next run.
#[test]
fn a_long_document_spans_several_windows() {
    let a = open_with(24, 2, 3);
    let (text, s) = text_of(12);
    let chunks: Vec<_> = (0..6).map(|i| (2 * i, 2 * i + 2, 0, None)).collect();
    let mut docs = [document(&text, &s, &chunks)];
    run(SentenceChunkVectors(a.clone()), &mut docs);
    let (ws, wc, plan) = reference(&a, &docs[0], true, true);
    assert!(plan.windows.len() >= 6, "{} windows", plan.windows.len());
    assert!(plan.split.is_empty(), "two sentences fit a window of {}", a.width());
    close("sentences", &sentence_vectors(&docs[0]).data, &ws);
    close("chunks", &chunk_vectors(&docs[0]).data, &wc);

    // Runs of the model's size give the same vectors.
    let big = open(24);
    let mut again = [document(&text, &s, &chunks)];
    run(SentenceChunkVectors(big), &mut again);
    close("sentences in bigger runs", &sentence_vectors(&again[0]).data, &sentence_vectors(&docs[0]).data);
    close("chunks in bigger runs", &chunk_vectors(&again[0]).data, &chunk_vectors(&docs[0]).data);
}

/// A sentence that would cross the end of the window before it starts a
/// window of its own, and is pooled whole there.
#[test]
fn a_sentence_across_a_window_end_is_pooled_from_a_window_that_holds_it() {
    let a = open(24);
    let (text, s) = text_of(6);
    let mut docs = [document(&text, &s, &[(0, 6, 0, None)])];
    run(SentenceVectors(a.clone()), &mut docs);
    let (want, _, plan) = reference(&a, &docs[0], true, false);
    let sub = docs[0].get::<Subwords>().unwrap();
    let (starts, ends) = sub.spans.columns();
    let width = a.width();
    // Some sentence starts inside the window before it and ends past it.
    let crossing = s.iter().enumerate().skip(1).find_map(|(i, &(x, y))| {
        let r = pieces_of(starts, ends, x, y);
        let before = plan.windows.iter().find(|w| w.start < r.start && r.start < w.end)?;
        (r.end > before.end).then_some((i, r))
    });
    let (i, r) = crossing.expect("a sentence crosses a window's end");
    let p = plan.placed.iter().find(|x| x.target.item == i as u32).unwrap();
    let w = plan.windows[p.window as usize];
    assert_eq!((w.start, p.start, p.end), (r.start, 0, r.end - r.start), "sentence {i}");
    assert!(r.end - r.start <= width);
    close("sentences", &sentence_vectors(&docs[0]).data, &want);
}

/// A chunk over the whole text in rows of 16 tokens is pooled in parts,
/// one window each, and its vector is their mean weighted by pieces.
#[test]
fn an_item_longer_than_a_window_is_the_weighted_mean_of_its_parts() {
    let a = open(16);
    let (text, s) = text_of(8);
    let chunks = [(0, 8, 1, None), (0, 4, 0, Some(0)), (4, 8, 0, Some(0))];
    let mut docs = [document(&text, &s, &chunks)];
    run(SentenceChunkVectors(a.clone()), &mut docs);
    let (ws, wc, plan) = reference(&a, &docs[0], true, true);
    // Every chunk is longer than a window of 14 pieces. Its parts follow
    // the sentences' windows, each piece in one part, and add no window.
    assert_eq!(plan.split.len(), 3);
    for sp in &plan.split {
        let parts: Vec<_> = plan.placed.iter().filter(|x| x.target == sp.target).collect();
        assert!(parts.iter().all(|x| !x.whole));
        assert!(parts.len() as u32 >= sp.pieces.div_ceil(a.width()));
        assert_eq!(parts.iter().map(|x| x.end - x.start).sum::<u32>(), sp.pieces);
    }
    let (_, _, alone) = reference(&a, &docs[0], true, false);
    assert_eq!(plan.windows, alone.windows);
    let vc = chunk_vectors(&docs[0]);
    close("chunks", &vc.data, &wc);
    close("sentences", &sentence_vectors(&docs[0]).data, &ws);
    assert!(vc.data.as_chunks::<32>().0.iter().all(|r| is_unit(r)));
    // Not just the first part's vector.
    let first = plan.placed.iter().find(|x| x.target == plan.split[0].target).unwrap();
    let w = plan.windows[first.window as usize];
    let sub = docs[0].get::<Subwords>().unwrap();
    let mut enc = Encoder::open(&bert(), &EncoderOptions { max_seq: 16, ..EncoderOptions::default() }).unwrap();
    let mut rows = Rows::new(16, 0);
    let (head, tail) = a.row_ends();
    rows.push(&[head, &sub.ids[w.start as usize..w.end as usize], tail]);
    let mut one = Vec::new();
    let h = head.len() as u32;
    enc.run_ranges(&rows, &[turbo_token_range { row: 0, start: h + first.start, end: h + first.end }], &mut one)
        .unwrap();
    normalize(&mut one);
    assert!(one.iter().zip(&vc.data[..32]).any(|(x, y)| (x - y).abs() > 1e-3));
}

/// An empty sentence and one of whitespace alone have no pieces: the zero
/// vector. The others are untouched by them.
#[test]
fn items_with_no_pieces_get_the_zero_vector() {
    let a = open(0);
    let text = "First sentence here.   Second one.";
    let s = [(0, 20), (20, 20), (20, 23), (23, 34)];
    let mut docs = [document(text, &s, &[(0, 4, 1, None), (1, 3, 0, Some(0))])];
    run(SentenceChunkVectors(a.clone()), &mut docs);
    let (vs, vc) = (sentence_vectors(&docs[0]), chunk_vectors(&docs[0]));
    assert_eq!(Layer::len(vs), 4);
    for i in [1, 2] {
        assert!(vs.row(Ix::new(i)).iter().all(|&x| x == 0.0), "sentence {i}");
    }
    assert!(vc.row(Ix::new(1)).iter().all(|&x| x == 0.0), "a chunk of the two");
    // The layers name them, and pass their own check.
    assert_eq!(vs.empty, [Ix::new(1), Ix::new(2)]);
    assert_eq!(vc.empty, [Ix::new(1)]);
    assert!(vs.is_empty(Ix::new(2)) && !vs.is_empty(Ix::new(3)) && !vc.is_empty(Ix::new(0)));
    assert_eq!((vs.check(), vc.check()), (Ok(()), Ok(())));
    assert!(is_unit(vs.row(Ix::new(0))) && is_unit(vs.row(Ix::new(3))) && is_unit(vc.row(Ix::new(0))));
    let (ws, wc, _) = reference(&a, &docs[0], true, true);
    close("sentences", &vs.data, &ws);
    close("chunks", &vc.data, &wc);

    // The chunks alone lay out the windows: the same chunk has no vector.
    let mut alone = [document(text, &s, &[(0, 4, 1, None), (1, 3, 0, Some(0))])];
    run(ChunkVectors(a.clone()), &mut alone);
    assert_eq!(chunk_vectors(&alone[0]).empty, [Ix::new(1)]);

    // A document with no sentences and no chunks: empty layers.
    let mut empty = [document("", &[], &[])];
    run(SentenceChunkVectors(a), &mut empty);
    assert_eq!(Layer::len(sentence_vectors(&empty[0])), 0);
    assert_eq!(Layer::len(chunk_vectors(&empty[0])), 0);
}

/// Three documents in one batch share runs, and get the vectors each gets
/// alone.
#[test]
fn documents_of_a_batch_get_what_each_gets_alone() {
    let a = open_with(24, 3, 4);
    let texts: Vec<_> = [3, 7, 5].iter().map(|&n| text_of(n)).collect();
    let make = |(t, s): &(String, Vec<(u32, u32)>)| document(t, s, &[(0, s.len() as u32, 0, None)]);
    let mut together: Vec<_> = texts.iter().map(make).collect();
    run(SentenceChunkVectors(a.clone()), &mut together);
    for (t, d) in texts.iter().zip(&together) {
        let mut alone = [make(t)];
        run(SentenceChunkVectors(a.clone()), &mut alone);
        close("sentences", &sentence_vectors(d).data, &sentence_vectors(&alone[0]).data);
        close("chunks", &chunk_vectors(d).data, &chunk_vectors(&alone[0]).data);
        let (ws, wc, _) = reference(&a, d, true, true);
        close("sentences against the reference", &sentence_vectors(d).data, &ws);
        close("chunks against the reference", &chunk_vectors(d).data, &wc);
    }
}

/// The chunk hierarchy of `text_of(12)` in rows of 24 tokens: the whole
/// text, halves and quarters longer than a window, and pairs of sentences
/// that fit one.
fn hierarchy() -> Vec<(u32, u32, u8, Option<u32>)> {
    let mut c = vec![(0, 12, 3, None), (0, 6, 2, Some(0)), (6, 12, 2, Some(0))];
    for q in 0..4 {
        c.push((3 * q, 3 * q + 3, 1, Some(1 + q / 2)));
    }
    for i in 0..6 {
        c.push((2 * i, 2 * i + 2, 0, Some(1 + i / 3)));
    }
    c
}

/// The sentences lay out the windows whether or not the chunks come with
/// them, so a sentence's vector is the same bits from either step, and the
/// chunks, long and short, add no window where the sentences hold every
/// piece.
#[test]
fn sentence_vectors_are_the_same_with_or_without_the_chunks() {
    let a = open(24);
    let (text, s) = text_of(12);
    let mut alone = [document(&text, &s, &hierarchy())];
    let mut both = [document(&text, &s, &hierarchy())];
    run(SentenceVectors(a.clone()), &mut alone);
    run(SentenceChunkVectors(a.clone()), &mut both);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&sentence_vectors(&alone[0]).data), bits(&sentence_vectors(&both[0]).data));

    let (_, _, by_sentences) = reference(&a, &alone[0], true, false);
    let (ws, wc, with_chunks) = reference(&a, &both[0], true, true);
    assert!(by_sentences.windows.len() > 1);
    assert_eq!(with_chunks.windows, by_sentences.windows);
    // Some chunks are longer than a window, and one is the whole text.
    assert!(with_chunks.split.iter().any(|x| x.target.item == 0));
    assert!(with_chunks.split.iter().any(|x| x.pieces > a.width()));
    close("sentences", &sentence_vectors(&both[0]).data, &ws);
    close("chunks", &chunk_vectors(&both[0]).data, &wc);

    // Without the pairs, the chunks alone lay out other windows than the
    // sentences do, so their vectors can differ; provenance says which
    // layer laid out the windows.
    let mut chunks = [document(&text, &s, &hierarchy()[..7])];
    run(ChunkVectors(a.clone()), &mut chunks);
    let (_, wc, by_chunks) = reference(&a, &chunks[0], false, true);
    assert_ne!(by_chunks.windows, by_sentences.windows);
    close("chunks laid out by the chunks", &chunk_vectors(&chunks[0]).data, &wc);
    let key = Key::<Vectors<Chunks>>::DEFAULT.id();
    assert_ne!(chunks[0].provenance(key).unwrap().config, both[0].provenance(key).unwrap().config);
    let key = Key::<Vectors<Sentences>>::DEFAULT.id();
    assert_eq!(alone[0].provenance(key).unwrap().config, both[0].provenance(key).unwrap().config);
}

/// Text between two sentences that no sentence covers: the sentences'
/// windows do not hold it, so a chunk over both adds windows there.
#[test]
fn a_chunk_over_text_no_sentence_covers_adds_a_window_there() {
    let a = open(16);
    let (text, all) = text_of(5);
    // Sentences 0 and 4 alone; 1 to 3 lie between them in no sentence.
    let s = [all[0], all[4]];
    let mut docs = [document(&text, &s, &[(0, 2, 0, None)])];
    run(SentenceChunkVectors(a.clone()), &mut docs);
    let (ws, wc, plan) = reference(&a, &docs[0], true, true);
    let (_, _, by_sentences) = reference(&a, &docs[0], true, false);
    assert_eq!(by_sentences.windows.len(), 2);
    assert!(plan.windows.len() > 2, "{:?}", plan.windows);
    assert_eq!(plan.windows[..2], by_sentences.windows[..]);
    // The chunk's parts cover each of its pieces once.
    let n: u32 = plan.placed.iter().filter(|x| x.target.which == Which::Chunks).map(|x| x.end - x.start).sum();
    assert_eq!(n, plan.split[0].pieces);
    close("sentences", &sentence_vectors(&docs[0]).data, &ws);
    close("chunks", &chunk_vectors(&docs[0]).data, &wc);
}

/// `embed` takes its input from any caller, so it checks what an analyzer
/// would: an id per piece, pieces in text order, spans that do not end
/// before they start. Each such document fails alone.
#[test]
fn malformed_input_fails_the_document() {
    let a = open(0);
    let (text, s) = text_of(2);
    let mut docs = [document(&text, &s, &[(0, 2, 0, None)])];
    run(SentenceVectors(a.clone()), &mut docs);
    let sub = docs[0].get::<Subwords>().unwrap();
    let sentences = &docs[0].get::<Sentences>().unwrap().spans;
    let (starts, ends) = sub.spans.columns();
    let with = |spans: &[(u32, u32)], ids: Vec<i32>| {
        let mut columns = Spans::new(sub.spans.view());
        for &(x, y) in spans {
            columns.push(Span { start: x, end: y });
        }
        Subwords { spans: columns, ids, model: sub.model.clone(), token: None }
    };
    let pairs: Vec<(u32, u32)> = starts.iter().copied().zip(ends.iter().copied()).collect();
    let short = with(&pairs, sub.ids[1..].to_vec());
    let mut swapped = pairs.clone();
    swapped.swap(0, 1);
    let unordered = with(&swapped, sub.ids.clone());
    let mut inverted = pairs.clone();
    inverted[0] = (inverted[0].1, inverted[0].0);
    let inverted = with(&inverted, sub.ids.clone());
    let mut backwards = Spans::<Sentences>::new(sentences.view());
    backwards.push(Span { start: 9, end: 3 });
    let mut backwards_chunk = Spans::<Chunks>::new(sentences.view());
    backwards_chunk.push(Span { start: 9, end: 3 });
    // Levels from (start, end, unit, parent), and a parent column of its own.
    let levels = |rows: &[(u32, u32, Unit, Option<u32>)], parent: Option<Vec<Option<Ix<Levels>>>>| {
        let mut spans = Spans::new(sentences.view());
        for &(x, y, ..) in rows {
            spans.push(Span::new(x, y));
        }
        let unit = rows.iter().map(|r| r.2).collect();
        let parent = parent.unwrap_or_else(|| rows.iter().map(|r| r.3.map(Ix::new)).collect());
        Levels { spans, unit, parent }
    };
    let chunk = Unit::Chunk(Ix::new(0));
    let parent_after = levels(&[(0, 40, Unit::Document, None), (0, 5, chunk, Some(7))], None);
    let short_parent = levels(&[(0, 40, Unit::Document, None), (0, 5, chunk, Some(0))], Some(vec![None]));
    let outside = levels(&[(0, 10, Unit::Document, None), (0, 40, chunk, Some(0))], None);
    let inputs = [
        Input { subwords: &short, sentences: Some(sentences), chunks: None, levels: None },
        Input { subwords: &unordered, sentences: Some(sentences), chunks: None, levels: None },
        Input { subwords: &inverted, sentences: Some(sentences), chunks: None, levels: None },
        Input { subwords: sub, sentences: Some(&backwards), chunks: None, levels: None },
        Input { subwords: sub, sentences: None, chunks: Some(&backwards_chunk), levels: None },
        Input { subwords: sub, sentences: None, chunks: None, levels: Some(&parent_after) },
        Input { subwords: sub, sentences: None, chunks: None, levels: Some(&short_parent) },
        Input { subwords: sub, sentences: None, chunks: None, levels: Some(&outside) },
        Input { subwords: sub, sentences: Some(sentences), chunks: None, levels: None },
    ];
    let mut out = a.embed("test", &inputs);
    for want in [
        "an id for every subword",
        "subwords in text order",
        "subword 0 ends before it starts",
        "sentence 0 ends before it starts: 9..3",
        "chunk 0 ends before it starts: 9..3",
        "the levels: unit 1 has a parent at or after it",
        "the levels: a unit and a parent for every span",
        "the levels: unit 1 is outside its parent",
    ] {
        let e = out.remove(0).err().unwrap_or_else(|| panic!("{want}: refused")).to_string();
        assert!(e.contains(want), "{want}: {e}");
    }
    let ok = out.remove(0).unwrap();
    close("the good document", &ok.sentences.unwrap().data, &sentence_vectors(&docs[0]).data);
}
