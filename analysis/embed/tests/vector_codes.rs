//! Narrower vectors from the vector steps (#1462), on the small BERT of
//! testdata/tiny-bert-bundle (dim 32) on the CPU, its manifest edited to
//! declare `embed.output_dims`: a cut vector is the full-width pooled
//! vector's first values, normalized after the cut, for sentences, chunks
//! and items split over several windows; a width the bundle does not
//! declare is refused. Then `VectorCodesStep`: its int8 and ubinary codes
//! are `turbo::quantize`'s codes of the vectors the step read, the bits
//! the vectors' signs top bit first, int8 against ranges calibrated on
//! other text; and the provider options for both.

#[path = "vectors/common.rs"]
mod common;

use common::*;
use turbo::manifest::{Manifest, Normalize};
use turbo::quantize::{self, Quantization};
use turbo_analysis::registry::{Registry, Spec};
use turbo_analysis::*;
use turbo_analysis_embed::window::Which;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Chunks, CodeKind, Sentences, VectorCodes, Vectors};

fn declare(m: &mut Manifest) {
    m.embed.as_mut().unwrap().output_dims = vec![16, 8];
}

/// The annotator on `bundle` with rows of at most 24 tokens, so twelve
/// sentences take several windows and a chunk over them is split.
fn open_at(bundle: &std::path::Path, output_dim: u32) -> Result<VectorAnnotator, turbo::status::Error> {
    let o = VectorOptions {
        encoder: EncoderOptions { max_seq: 24, ..EncoderOptions::default() },
        output_dim,
        ..VectorOptions::default()
    };
    VectorAnnotator::open(bundle, &o)
}

/// Twelve sentences, one chunk over all of them and one over the first two.
fn docs() -> [Document; 1] {
    let (text, s) = text_of(12);
    [document(&text, &s, &[(0, 12, 1, None), (0, 2, 0, Some(0))])]
}

#[test]
fn cut_vectors_are_the_pooled_vectors_first_values_normalized_after_the_cut() {
    let cut = Edited::new(declare);
    let raw = Edited::new(|m| {
        declare(m);
        m.embed.as_mut().unwrap().normalize = Normalize::None;
    });
    let full = std::sync::Arc::new(open_at(&raw.0, 0).unwrap());
    let mut want = docs();
    run_bundle(&raw.0, SentenceChunkVectors(full.clone()), &mut want);
    for w in [16u32, 8] {
        let a = std::sync::Arc::new(open_at(&cut.0, w).unwrap());
        assert_eq!((a.dim(), a.hidden_dim(), a.token_dim()), (w, 32, 32));
        let mut got = docs();
        run_bundle(&cut.0, SentenceChunkVectors(a.clone()), &mut got);
        let (vs, vc) = (sentence_vectors(&got[0]), chunk_vectors(&got[0]));
        assert_eq!((vs.dim, Layer::len(vs), vc.dim, Layer::len(vc)), (w, 12, w, 2));
        let mut windows = 0;
        for (g, f) in [(&vs.data, &sentence_vectors(&want[0]).data), (&vc.data, &chunk_vectors(&want[0]).data)] {
            for (g, f) in g.chunks_exact(w as usize).zip(f.as_chunks::<32>().0) {
                let mut f = f[..w as usize].to_vec();
                normalize(&mut f);
                assert_eq!(g, &f[..], "width {w}: the pooled vector cut, then normalized");
                windows += 1;
            }
        }
        assert_eq!(windows, 14);
        // A cut vector is not the full one's hash; full width keeps it.
        assert_ne!(a.config(Which::Chunks), full.config(Which::Chunks));
    }
    let mut v = open_at(&cut.0, 32).unwrap().config(Which::Sentences);
    assert_eq!(v, open_at(&cut.0, 0).unwrap().config(Which::Sentences));
    v = open_at(&bert(), 0).unwrap().config(Which::Sentences);
    assert_eq!(v, open_at(&cut.0, 0).unwrap().config(Which::Sentences), "the declaration alone changes nothing");
}

#[test]
fn a_width_the_bundle_does_not_declare_is_refused() {
    let cut = Edited::new(declare);
    let e = open_at(&cut.0, 12).err().expect("undeclared");
    assert_eq!(e.code, turbo::status::UNSUPPORTED_OPTION, "{e}");
    assert!(e.to_string().contains("output_dim 12: the model was trained to be cut to [16, 8]"), "{e}");
    let e = open_at(&cut.0, 33).err().expect("over dim");
    assert_eq!(e.code, turbo::status::INVALID_ARGUMENT, "{e}");
    assert!(e.to_string().contains("output_dim 33 is over the model's dim 32"), "{e}");
    assert!(open_at(&bert(), 16).is_err(), "the bundle as shipped declares no widths");
}

/// Per-dimension minimums then maximums of rows of `dim` values.
fn ranges_of(data: &[f32], dim: usize) -> Vec<f32> {
    let rows: Vec<&[f32]> = data.chunks_exact(dim).collect();
    let min = (0..dim).map(|d| rows.iter().map(|r| r[d]).fold(f32::INFINITY, f32::min));
    let max = (0..dim).map(|d| rows.iter().map(|r| r[d]).fold(f32::NEG_INFINITY, f32::max));
    min.chain(max).collect()
}

#[test]
fn codes_are_the_codes_of_the_vectors_the_step_read() {
    let cut = Edited::new(declare);
    for w in [32u32, 8] {
        let a = std::sync::Arc::new(open_at(&cut.0, w).unwrap());
        // int8 ranges calibrated on other text: the last six sentences.
        let (text, s) = text_of(12);
        let mut calib = [document(&text[s[6].0 as usize..], &rebase(&s[6..]), &[(0, 6, 0, None)])];
        run_bundle(&cut.0, SentenceVectors(a.clone()), &mut calib);
        let ranges = ranges_of(&sentence_vectors(&calib[0]).data, w as usize);
        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Chunks>::DEFAULT)
            .add(SubwordAnnotator::open(&cut.0).unwrap())
            .add(SentenceChunkVectors(a.clone()))
            .add(VectorCodesStep::<Sentences>::ubinary())
            .add(VectorCodesStep::<Chunks>::int8(ranges.clone()).unwrap())
            .build()
            .unwrap();
        let mut d = docs();
        analyzer.run(&mut d).remove(0).unwrap();
        let (vs, vc) = (sentence_vectors(&d[0]), chunk_vectors(&d[0]));
        let bits = d[0].get::<VectorCodes<Sentences>>().unwrap();
        let mut want = vec![0u8; 12 * w.div_ceil(8) as usize];
        quantize::rows(Quantization::UBinary, &vs.data, w as usize, &[], &mut want);
        assert_eq!((bits.kind, bits.dim, &bits.data), (CodeKind::UBinary, w, &want), "width {w}");
        for (r, row) in vs.data.chunks_exact(w as usize).enumerate() {
            for (i, &x) in row.iter().enumerate() {
                let bit = bits.row(Ix::new(r as u32))[i / 8] >> (7 - i % 8) & 1;
                assert_eq!(bit == 1, x > 0.0, "sentence {r} value {i}: {x}");
            }
        }
        let codes = d[0].get::<VectorCodes<Chunks>>().unwrap();
        let mut want = vec![0u8; 2 * w as usize];
        quantize::rows(Quantization::Int8, &vc.data, w as usize, &ranges, &mut want);
        assert_eq!((codes.kind, codes.dim, &codes.data), (CodeKind::Int8, w, &want), "width {w}");
        assert_eq!(codes.model, vc.model);
    }
}

/// Sentence spans moved to start at the first one.
fn rebase(s: &[(u32, u32)]) -> Vec<(u32, u32)> {
    s.iter().map(|&(a, b)| (a - s[0].0, b - s[0].0)).collect()
}

#[test]
fn int8_ranges_that_cannot_be_used_are_refused() {
    assert!(VectorCodesStep::<Chunks>::int8(Vec::new()).err().unwrap().contains("no values"));
    let e = VectorCodesStep::<Chunks>::int8(vec![0.0, f32::NAN]).err().unwrap();
    assert!(e.contains("not both finite"), "{e}");
    let e = VectorCodesStep::<Chunks>::int8(vec![1.0, 0.0]).err().unwrap();
    assert!(e.contains("minimum 1 is over its maximum 0"), "{e}");
    // Ranges of another width fail the document, by its width.
    let a = std::sync::Arc::new(open_at(&bert(), 0).unwrap());
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .add(SubwordAnnotator::open(&bert()).unwrap())
        .add(ChunkVectors(a))
        .add(VectorCodesStep::<Chunks>::int8([vec![-1.0; 8], vec![1.0; 8]].concat()).unwrap())
        .build()
        .unwrap();
    let e = analyzer.run(&mut docs()).remove(0).unwrap_err().to_string();
    assert!(e.contains("int8 ranges for width 8 given vectors of width 32"), "{e}");
}

#[test]
fn the_providers_take_output_dim_and_codes() {
    let cut = Edited::new(declare);
    let mut r: Registry<dyn turbo_analysis::annotate::Step> = Registry::default();
    r.register(SubwordsProvider);
    r.register(VectorsProvider);
    r.register(VectorCodesProvider);
    let ranges = cut.0.join("ranges.f32");
    let table: Vec<u8> = [[-0.5f32; 8], [0.5; 8]].concat().iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&ranges, table).unwrap();
    for (codes, location, kind) in
        [("ubinary", None, CodeKind::UBinary), ("int8", Some(ranges.clone()), CodeKind::Int8)]
    {
        let sub = Spec { location: Some(cut.0.clone()), options: Vec::new() };
        let vec = Spec { location: Some(cut.0.clone()), options: Vec::new() }.with("layers", "chunks");
        let code = Spec { location, options: Vec::new() }.with("codes", codes);
        let a = r
            .pipeline(&[("subwords", sub), ("vectors", vec.with("output_dim", "8")), ("vector-codes", code)], |b| {
                b.given(Key::<Sentences>::DEFAULT).given(Key::<Chunks>::DEFAULT)
            })
            .unwrap();
        let mut d = docs();
        a.run(&mut d).remove(0).unwrap();
        assert_eq!(d[0].get::<Vectors<Chunks>>().unwrap().dim, 8, "{codes}");
        let c = d[0].get::<VectorCodes<Chunks>>().unwrap();
        assert_eq!((c.kind, c.dim, c.data.len()), (kind, 8, 2 * kind.row_bytes(8)), "{codes}");
    }
    let refused = |name: &str, spec: Spec| r.create(name, &spec).err().expect("refused").to_string();
    let e = refused("vectors", Spec { location: Some(cut.0.clone()), options: Vec::new() }.with("output_dim", "12"));
    assert!(e.contains("output_dim 12"), "{e}");
    let e = refused("vectors", Spec { location: Some(cut.0.clone()), options: Vec::new() }.with("output_dim", "x"));
    assert!(e.contains("output_dim \"x\""), "{e}");
    let e = refused("vector-codes", Spec::default().with("codes", "int4"));
    assert!(e.contains("codes \"int4\""), "{e}");
    let e = refused("vector-codes", Spec::default().with("codes", "int8"));
    assert!(e.contains("int8 codes need a ranges file"), "{e}");
    let e = refused("vector-codes", Spec::default().with("layers", "tokens"));
    assert!(e.contains("layers \"tokens\""), "{e}");
}
