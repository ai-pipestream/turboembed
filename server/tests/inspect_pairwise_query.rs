//! `ProjectChunks` pairwise cosines (#2248) and query placement (#2249)
//! over real gRPC on the CPU reference with testdata/tiny-bert-bundle. The
//! stored vectors are the served model's own answers to `ModelInfer`, and
//! every expected number is computed here from them.

#[path = "kserve/mod.rs"]
mod kserve;

use kserve::*;
use turbo_analysis::export::export;
use turbo_analysis::{Document, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Sentences, Vectors};
use turbo_chunker::{ProjectionDims, VectorProjection};
use turbo_kserve::Server;
use turbo_kserve::inspect::pairwise::{DEFAULT_PAIRWISE_ITEMS, MAX_PAIRWISE_ITEMS};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::{
    CapabilitiesRequest, OffsetMode, PairwiseCosinesSpec, ProjectChunksRequest, ProjectChunksResponse,
};
use turbo_kserve::results::document::codecs;

type Analysis = AnalysisServiceClient<tonic::transport::Channel>;

const SENTENCES: [&str; 5] = [
    "The harbour ferry leaves at dawn. ",
    "Zoë sells smoked fish on the quay. ",
    "Rates rose by a quarter point. ",
    "東京 の 市場 は 朝 に 開く。 ",
    "The central bank left rates alone.",
];
const QUERY: &str = "what did the central bank do with rates";

async fn analysis(s: &Server) -> Analysis {
    AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap()
}

/// The served bundle's own identity, as `Capabilities` lists it.
async fn served_model(c: &mut Analysis) -> ModelRef {
    let listed = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let m = listed.bundles.iter().find(|b| b.name == NAME).unwrap().model.clone().unwrap();
    ModelRef { id: m.id, revision: m.revision, sha256: m.sha256 }
}

/// One whole-document chunk at level 1 (row 0), then one level 0 chunk per
/// sentence (rows 1 to 5), with `rows` as their vectors: row 0's vector is
/// the first sentence's again, so every level has a vector.
fn graph(rows: &[Vec<f32>], empty: &[u32], model: ModelRef) -> (Vec<u8>, Vectors<Chunks>) {
    let text: String = SENTENCES.concat();
    let mut doc = Document::new(text.as_str());
    let (mut sentences, mut chunks) = (Spans::new(ViewId::SOURCE), Spans::<Chunks>::new(ViewId::SOURCE));
    chunks.push(Span::new(0, text.len() as u32));
    let mut at = 0u32;
    for s in SENTENCES {
        let span = Span::new(at, at + s.trim_end().len() as u32);
        sentences.push(span);
        chunks.push(span);
        at += s.len() as u32;
    }
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    let n = SENTENCES.len();
    let mut ranges = vec![IxRange::new(0, n as u32)];
    ranges.extend((0..n as u32).map(|i| IxRange::new(i, i + 1)));
    let mut parent = vec![None];
    parent.extend((0..n).map(|_| Some(Ix::new(0))));
    let mut level = vec![1u8];
    level.extend(vec![0u8; n]);
    doc.add(
        &Key::DEFAULT,
        Chunks {
            spans: chunks,
            level,
            parent,
            sentences: ranges,
            cut: vec![Cut::DocumentStart; n + 1],
            cost: vec![0; n + 1],
            model_tokens: vec![8; n + 1],
            over_budget: vec![false; n + 1],
            prompt_overhead: 2,
        },
        Provenance::by("test"),
    )
    .unwrap();
    let dim = rows[0].len() as u32;
    let data: Vec<f32> = rows.iter().flatten().copied().collect();
    let make = || Vectors::<Chunks>::new(dim, data.clone(), empty.iter().map(|&i| Ix::new(i)).collect(), model.clone());
    doc.add(&Key::DEFAULT, make(), Provenance::by("test")).unwrap();
    (export(&doc, codecs()).unwrap().to_bytes().unwrap(), make())
}

fn request(graph: &[u8], pca_dims: u32) -> ProjectChunksRequest {
    ProjectChunksRequest { graph: graph.to_vec(), offsets: OffsetMode::Utf8 as i32, pca_dims, ..Default::default() }
}

async fn project(c: &mut Analysis, r: ProjectChunksRequest) -> ProjectChunksResponse {
    c.project_chunks(r).await.unwrap().into_inner()
}

/// The stored rows: row 0 is the first sentence's vector again.
async fn stored(c: &mut Client, normalize: &str) -> Vec<Vec<f32>> {
    let got = embed(c, with(texts(&SENTENCES), "normalize", st(normalize))).await;
    let mut rows = vec![got[0].clone()];
    rows.extend(got);
    rows
}

#[tokio::test]
async fn pairwise_cosines_are_the_stored_vectors_cosines() {
    let (s, mut infer) = serve(1).await;
    let mut c = analysis(&s).await;
    let model = served_model(&mut c).await;
    let mut rows = stored(&mut infer, "NORMALIZE_NONE").await;
    // Row 3 is listed empty and row 4 is a vector of zero length: neither
    // can be compared, so the level's items are rows 1, 2 and 5.
    rows[3].fill(0.0);
    rows[4].fill(0.0);
    let (bytes, vectors) = graph(&rows, &[3], model);

    let plain = project(&mut c, request(&bytes, 2)).await;
    assert_eq!(plain.pairwise, None);
    assert_eq!(plain.query, None);

    let mut r = request(&bytes, 2);
    r.pairwise = Some(PairwiseCosinesSpec { level: 0, max_items: 0 });
    let out = project(&mut c, r.clone()).await;
    assert_eq!(out.projection, plain.projection);
    let pairs = out.pairwise.unwrap();
    assert_eq!((pairs.level, pairs.items.as_slice()), (0, &[1u32, 2, 5][..]));
    let row = |i: u32| vectors.row(Ix::new(i));
    let want: Vec<f64> = vec![cosine(row(1), row(2)), cosine(row(1), row(5)), cosine(row(2), row(5))];
    assert_eq!(pairs.cosines.len(), want.len());
    for (got, want) in pairs.cosines.iter().zip(&want) {
        assert!((f64::from(*got) - want).abs() < 1e-6, "{got} vs {want}");
    }
    // The three are not one number said three times.
    assert!((want[0] - want[2]).abs() > 1e-4, "{want:?}");

    // The level above holds one item: listed, with no pair to compare.
    r.pairwise = Some(PairwiseCosinesSpec { level: 1, max_items: 1 });
    let top = project(&mut c, r.clone()).await.pairwise.unwrap();
    assert_eq!((top.level, top.items, top.cosines), (1, vec![0], Vec::new()));
    // A level nothing sits at compares nothing.
    r.pairwise = Some(PairwiseCosinesSpec { level: 7, max_items: 0 });
    let none = project(&mut c, r.clone()).await.pairwise.unwrap();
    assert!(none.items.is_empty() && none.cosines.is_empty());

    // Three comparable items against a cap of two: refused naming the cap.
    r.pairwise = Some(PairwiseCosinesSpec { level: 0, max_items: 2 });
    let e = c.project_chunks(r.clone()).await.unwrap_err();
    assert_eq!(refusal(&e), (tonic::Code::OutOfRange, 771, 0));
    assert!(e.message().contains("pairwise items is 3") && e.message().contains("max_items (2)"), "{}", e.message());
    r.pairwise = Some(PairwiseCosinesSpec { level: 0, max_items: 3 });
    assert_eq!(project(&mut c, r.clone()).await.pairwise.unwrap().cosines, pairs.cosines);
    // A cap over the server's own ceiling is refused whatever the level holds.
    assert_eq!((DEFAULT_PAIRWISE_ITEMS, MAX_PAIRWISE_ITEMS), (256, 512));
    r.pairwise = Some(PairwiseCosinesSpec { level: 0, max_items: MAX_PAIRWISE_ITEMS + 1 });
    let e = c.project_chunks(r).await.unwrap_err();
    assert_eq!(refusal(&e), (tonic::Code::OutOfRange, 771, 0));
    assert!(e.message().contains("max_pairwise_items"), "{}", e.message());

    // A graph with no vectors has nothing to compare.
    let text: String = SENTENCES.concat();
    let bare = export(&Document::new(text.as_str()), codecs()).unwrap().to_bytes().unwrap();
    let mut r = request(&bare, 2);
    r.pairwise = Some(PairwiseCosinesSpec { level: 0, max_items: 0 });
    let e = c.project_chunks(r).await.unwrap_err();
    assert_eq!(refusal(&e), (tonic::Code::InvalidArgument, 256, 0));
    s.stop().await;
}

#[tokio::test]
async fn a_query_lands_in_the_projections_own_basis() {
    let (s, mut infer) = serve(1).await;
    let mut c = analysis(&s).await;
    let model = served_model(&mut c).await;
    for normalize in ["NORMALIZE_L2", "NORMALIZE_NONE"] {
        let mut rows = stored(&mut infer, normalize).await;
        rows[3].fill(0.0);
        let (bytes, vectors) = graph(&rows, &[3], model.clone());
        // What the server should embed: the query role, normalized as the
        // stored rows are.
        let asked = with(with(texts(&[QUERY]), "prompt_role", st("PROMPT_QUERY")), "normalize", st(normalize));
        let query = embed(&mut infer, asked).await.remove(0);
        if normalize == "NORMALIZE_L2" {
            assert!((norm(&query) - 1.0).abs() < 1e-4);
        } else {
            assert!((norm(&query) - 1.0).abs() > 1e-2, "the bundle's raw vectors are not unit length");
        }
        for (pca_dims, dims) in [(2, ProjectionDims::D2), (3, ProjectionDims::D3)] {
            let mut r = request(&bytes, pca_dims);
            r.query = Some(QUERY.into());
            let out = project(&mut c, r).await;
            let points = out.projection.unwrap().points;
            let placed = out.query.unwrap();
            assert_eq!(placed.model_name, NAME);
            let basis = VectorProjection::compute_with_basis(&vectors, dims).unwrap();
            let want = basis.place(&query).unwrap();
            let got = [placed.x, placed.y, placed.z];
            for (axis, want) in want.iter().enumerate() {
                assert!((got[axis] - want).abs() < 1e-4, "{normalize} axis {axis}: {} vs {want}", got[axis]);
            }
            if pca_dims == 2 {
                assert_eq!(placed.z, 0.0);
            }
            assert!(got.iter().any(|x| x.abs() > 1e-4), "the query is not at the origin: {got:?}");
            assert_eq!(placed.cosines.len(), points.len());
            for (i, got) in placed.cosines.iter().enumerate() {
                let want = if i == 3 { 0.0 } else { cosine(&query, vectors.row(Ix::new(i as u32))) };
                assert!((f64::from(*got) - want).abs() < 1e-4, "{normalize} item {i}: {got} vs {want}");
            }
            // Rows 0 and 1 hold one vector, and the others their own.
            assert_eq!(placed.cosines[0], placed.cosines[1]);
            assert!((placed.cosines[1] - placed.cosines[2]).abs() > 1e-4, "{:?}", placed.cosines);
        }
    }
    s.stop().await;
}

/// A graph with sentence vectors and no chunk vectors: the sentences are
/// the items, all at level 0, for the pairs and for the query alike.
#[tokio::test]
async fn sentence_vectors_are_compared_and_hold_a_query() {
    let (s, mut infer) = serve(1).await;
    let mut c = analysis(&s).await;
    let model = served_model(&mut c).await;
    let rows = embed(&mut infer, with(texts(&SENTENCES), "normalize", st("NORMALIZE_L2"))).await;
    let text: String = SENTENCES.concat();
    let mut doc = Document::new(text.as_str());
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0u32;
    for s in SENTENCES {
        spans.push(Span::new(at, at + s.trim_end().len() as u32));
        at += s.len() as u32;
    }
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    let data: Vec<f32> = rows.iter().flatten().copied().collect();
    let make = || Vectors::<Sentences>::new(rows[0].len() as u32, data.clone(), Vec::new(), model.clone());
    doc.add(&Key::DEFAULT, make(), Provenance::by("test")).unwrap();
    let bytes = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let vectors = make();
    let row = |i: usize| vectors.row(Ix::new(i as u32));

    let mut r = request(&bytes, 2);
    r.pairwise = Some(PairwiseCosinesSpec { level: 0, max_items: 0 });
    r.query = Some(QUERY.into());
    let out = project(&mut c, r).await;
    assert!(out.projection.unwrap().points.iter().all(|p| p.level == 0));
    let pairs = out.pairwise.unwrap();
    assert_eq!(pairs.items, vec![0, 1, 2, 3, 4]);
    let want: Vec<f64> =
        (0..5).flat_map(|a| (a + 1..5).map(move |b| (a, b))).map(|(a, b)| cosine(row(a), row(b))).collect();
    assert_eq!(pairs.cosines.len(), want.len());
    for (got, want) in pairs.cosines.iter().zip(&want) {
        assert!((f64::from(*got) - want).abs() < 1e-6, "{got} vs {want}");
    }
    assert!((want[0] - want[9]).abs() > 1e-4, "{want:?}");

    let asked = with(with(texts(&[QUERY]), "prompt_role", st("PROMPT_QUERY")), "normalize", st("NORMALIZE_L2"));
    let query = embed(&mut infer, asked).await.remove(0);
    let placed = out.query.unwrap();
    let at = VectorProjection::compute_with_basis(&vectors, ProjectionDims::D2).unwrap().place(&query).unwrap();
    assert!((placed.x - at[0]).abs() < 1e-4 && (placed.y - at[1]).abs() < 1e-4, "{placed:?} vs {at:?}");
    assert!(placed.x.abs() > 1e-4 || placed.y.abs() > 1e-4);
    assert_eq!(placed.cosines.len(), 5);
    for (i, got) in placed.cosines.iter().enumerate() {
        assert!((f64::from(*got) - cosine(&query, row(i))).abs() < 1e-4, "item {i}");
    }
    s.stop().await;
}

#[tokio::test]
async fn a_query_is_refused_without_vectors_or_their_model() {
    let (s, mut infer) = serve(1).await;
    let mut c = analysis(&s).await;
    let model = served_model(&mut c).await;
    let rows = stored(&mut infer, "NORMALIZE_L2").await;

    // No vectors in the graph.
    let text: String = SENTENCES.concat();
    let bare = export(&Document::new(text.as_str()), codecs()).unwrap().to_bytes().unwrap();
    let mut r = request(&bare, 2);
    r.query = Some(QUERY.into());
    let e = c.project_chunks(r).await.unwrap_err();
    assert_eq!(refusal(&e), (tonic::Code::InvalidArgument, 256, 0));
    assert!(e.message().contains("no chunk or sentence vectors"), "{}", e.message());

    // Vectors of a model this server does not serve: by id, by revision,
    // by the bundle's digest (the same name retrained), and by width.
    let other = ModelRef { id: "somebody-else".into(), ..model.clone() };
    let revised = ModelRef { revision: "another-revision".into(), ..model.clone() };
    let retrained = ModelRef { sha256: "0".repeat(64), ..model.clone() };
    assert_ne!(retrained.sha256, model.sha256);
    let narrow: Vec<Vec<f32>> = rows.iter().map(|r| r[..7].to_vec()).collect();
    for (rows, model, said) in [
        (&rows, other, "somebody-else".to_string()),
        (&rows, revised, "@another-revision#".to_string()),
        (&rows, retrained, format!("#{} at", "0".repeat(64))),
        (&narrow, model.clone(), "at 7 dimensions".to_string()),
    ] {
        let (bytes, _) = graph(rows, &[], model);
        let mut r = request(&bytes, 2);
        r.query = Some(QUERY.into());
        let e = c.project_chunks(r).await.unwrap_err();
        assert_eq!(refusal(&e), (tonic::Code::NotFound, 1280, 0));
        assert!(e.message().contains(&said), "{}", e.message());
    }

    // An empty text and one past the limit.
    let (bytes, _) = graph(&rows, &[], model);
    for (text, grpc, code) in
        [("  ".to_string(), tonic::Code::InvalidArgument, 256), ("a ".repeat(4097), tonic::Code::OutOfRange, 771)]
    {
        let mut r = request(&bytes, 2);
        r.query = Some(text);
        assert_eq!(refusal(&c.project_chunks(r).await.unwrap_err()), (grpc, code, 0));
    }
    // Every stored vector empty: nothing to stand among.
    let zeros = vec![vec![0.0f32; rows[0].len()]; rows.len()];
    let (bytes, _) = graph(&zeros, &[0, 1, 2, 3, 4, 5], served_model(&mut c).await);
    let mut r = request(&bytes, 2);
    r.query = Some(QUERY.into());
    assert_eq!(refusal(&c.project_chunks(r).await.unwrap_err()), (tonic::Code::InvalidArgument, 256, 0));
    s.stop().await;
}
