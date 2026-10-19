use serde_json::{Value, json};
use std::{path::Path, sync::Arc, time::Instant};
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::*;
use turbo_chunker::terms::{
    Bm25, LocalCollection, NormalizedTermIndexer, Query, TermCounter, TermForm, TermIndexer, TermStats, Vocabulary,
};
use turbo_chunker::{Budget, Chunker, Costs, StructureBoundaries};
use turbo_geo::{BundledGazetteer, DocumentRegionAnnotator, GeocodeAnnotator, PopulationPriorGeocoder};
use turbo_geo::{Locations, MentionPolicy};
use turbo_normalize::{NormalizeStep, Whitespace};
use turbo_structure::{StructureAnnotator, StructureProjector, Syntax};

fn fail(s: impl std::fmt::Display) -> String {
    s.to_string()
}

fn count<L: Layer>(doc: &Document) -> usize {
    doc.get::<L>().map_or(0, Layer::len)
}

fn write_pretty_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(fail)?;
    bytes.push(b'\n');
    std::fs::write(path, bytes).map_err(fail)
}

fn sample_spans<L: Layer>(doc: &Document) -> Vec<Value> {
    let Some(layer) = doc.get::<L>() else {
        return vec![];
    };
    let Some(sp) = layer.text_spans() else {
        return vec![];
    };
    sp.start
        .iter()
        .zip(sp.end)
        .enumerate()
        .take(12)
        .map(|(i, (&a, &b))| {
            let span = Span::new(a, b);
            let source = doc.to_source_bytes(sp.view, span);
            json!({
                "row": i,
                "span": [a, b],
                "source_bytes": [source.start, source.end],
                "text": &doc.text(sp.view)[span.range()],
                "source_text": String::from_utf8_lossy(&doc.bytes()[source.range()]),
            })
        })
        .collect()
}

fn main() -> Result<(), String> {
    let output = std::env::args().nth(1).ok_or("output directory required")?;
    let bundle = std::path::PathBuf::from(std::env::var("TURBO_MINILM_BUNDLE").map_err(fail)?);
    let vectors = Arc::new(
        VectorAnnotator::open(
            &bundle,
            &VectorOptions {
                encoder: EncoderOptions { cpu_threads: 4, max_seq: 256, ..Default::default() },
                ..Default::default()
            },
        )
        .map_err(fail)?,
    );
    // Dictionary entries come from the project's actual bundled Natural Earth
    // data. No names from either evaluation document are added by hand.
    let entries: Vec<Vec<String>> = turbo_geo::BUNDLED_TABLE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .flat_map(|line| {
            let c: Vec<_> = line.split(';').collect();
            assert!(c.len() >= 4, "bundled gazetteer line has fewer than 4 columns");
            let mut names = vec![c[2]];
            names.extend(c[3].split('|').filter(|x| !x.is_empty()));
            names
                .into_iter()
                .map(|n| {
                    let mut spans = Vec::new();
                    turbo_tokenize::simple_spans(n, Default::default(), &mut spans);
                    spans.iter().map(|s| n[s.range()].to_owned()).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let dictionary = turbo_maxent::Dictionary::from_entries(false, entries.iter().map(Vec::as_slice));
    let mention_policy = MentionPolicy::ExcludeSingleAsciiLetters;
    let pipeline = Analyzer::builder()
        .step(Bound::new(StructureAnnotator::new(Syntax::Plain)).write(0, "source"))
        .step(Bound::new(NormalizeStep::new(Whitespace::keep_paragraphs())).write(0, "clean"))
        .step(Bound::new(StructureProjector).read(0, "clean").read(1, "source"))
        .step(Bound::new(turbo_sentdetect::RuleSentenceDetector::new("en")).read(0, "clean"))
        .step(Bound::new(turbo_tokenize::SimpleTokenizer::default()).read(0, "clean"))
        .step(Bound::new(SubwordAnnotator::open(&bundle).map_err(fail)?).read(0, "clean"))
        .step(
            Bound::new(turbo_namefind::dictionary::DictionaryNameFinder::new(dictionary, "location")).read(0, "clean"),
        )
        .step(
            Bound::new(
                GeocodeAnnotator::new(PopulationPriorGeocoder::new(BundledGazetteer::get()))
                    .with_mention_policy(mention_policy),
            )
            .read(0, "clean"),
        )
        .step(Bound::new(DocumentRegionAnnotator::new()).read(0, "clean"))
        .step(Bound::new(turbo_numeric::NumberScanner::new(NumberNotation::LatinUs)).read(0, "clean"))
        .step(Bound::new(turbo_numeric::TemporalExtractor::new()).read(0, "clean"))
        .step(Bound::new(turbo_pii::PiiDetector::pack(turbo_pii::Pack::AllStructured)).read(0, "clean"))
        .step(Bound::new(turbo_stem::StemAnnotator::new(turbo_stem::Algorithm::Porter)).read(0, "clean"))
        .step(Bound::new(turbo_abbrev::AbbreviationAnnotator).read(0, "clean"))
        .add(StructureBoundaries { costs: Costs::default() })
        .add(Chunker { budget: Budget { max: 200, target: 150, size_weight: 1, overhead: 2 } })
        .step(Bound::new(turbo_chunker::levels::LevelBuilder).read(0, "clean"))
        .add(SentenceChunkVectors(Arc::clone(&vectors)))
        .step(Bound::new(turbo_stem::TermAnnotator::for_language("en").unwrap()).read(0, "clean"))
        .add(NormalizedTermIndexer { form: TermForm::Folded })
        .add(TermCounter::<Chunks>::new())
        .build()
        .map_err(fail)?;
    let term_profile_probe = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .add(turbo_stem::TermAnnotator::for_language("en").ok_or("English profile missing")?)
        .add(TermIndexer)
        .build()
        .err()
        .map(|e| e.to_string());
    let base = std::env::var("TURBO_MASC_ROOT").map_err(fail)?;
    let paths = [
        ("dublin", "written/travel-guides/IntroDublin.txt", "Dublin Liffey"),
        ("pulmonary-fibrosis", "written/technical/rr166.txt", "pulmonary fibrosis"),
    ];
    let mut results = vec![];
    let mut reference_documents = vec![];
    for (name, relative, query) in paths {
        let input = Path::new(&base).join(relative);
        let bytes = std::fs::read(&input).map_err(fail)?;
        let source_digest = turbo_maxent::sha256(&bytes).to_hex();
        let expected_digest = match name {
            "dublin" => "ec416a95af0a514f8f192e70fbce208986f37f67da19ef300acec44dacfed660",
            _ => "11054580c6b94fe3c62085cabfae4a1c7c15343c6de197c670e5da37ac403933",
        };
        assert_eq!(source_digest, expected_digest, "MASC input differs from the reviewed fixture");
        let mut docs = [turbo_encoding::open(Arc::<[u8]>::from(bytes), None).map_err(fail)?];
        let started = Instant::now();
        pipeline.run(&mut docs).remove(0).map_err(fail)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let doc = &docs[0];
        let tokens = doc.get::<Tokens>().unwrap();
        let clean = doc.get_key(&Key::<View>::named("clean")).unwrap();
        // Every token must lead to its actual original bytes. No normalized
        // offset is treated as a source offset.
        for (_, s) in tokens.spans.iter() {
            let source = doc.to_source_bytes(tokens.spans.view(), s);
            assert_eq!(&doc.bytes()[source.range()], &clean.text().as_bytes()[s.range()]);
        }
        let layers: Vec<Value> = doc
            .layers()
            .map(|key| {
                let p = doc.provenance(key).unwrap();
                json!({
                    "key": key.to_string(),
                    "status": format!("{:?}", doc.status(key).unwrap()),
                    "annotator": p.annotator,
                    "version": p.version,
                    "config": format!("{:016x}", p.config),
                    "model": p.model.as_ref().map(|m| json!({"id": m.id, "revision": m.revision, "sha256": m.sha256})),
                    "execution": p.execution.as_ref().map(|e| json!({"backend": e.backend, "device": e.device, "precision": e.precision})),
                    "reads": p.reads.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "made_on": doc.made_on(key).iter().map(ToString::to_string).collect::<Vec<_>>(),
                })
            })
            .collect();
        let chunkvec = doc.get::<Vectors<Chunks>>().unwrap();
        let sentvec = doc.get::<Vectors<Sentences>>().unwrap();
        assert_eq!(chunkvec.model.id, "sentence-transformers/all-MiniLM-L6-v2");
        for i in 0..chunkvec.len() {
            if !chunkvec.is_empty(Ix::new(i as u32)) {
                let norm = chunkvec.row(Ix::new(i as u32)).iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
                assert!((norm - 1.0).abs() < 1e-4);
                assert_eq!(chunkvec.dim, 384);
            }
        }
        let chunks = doc.get::<Chunks>().unwrap();
        let stats = doc.get::<TermStats<Chunks>>().unwrap();
        assert!(chunks.over_budget.iter().zip(&chunks.level).all(|(&over, &level)| level > 0 || !over));
        let vocabulary = doc.get::<Vocabulary>().unwrap();
        let local = LocalCollection::only(stats, vocabulary, |i| chunks.level[i.get()] == 0);
        let lower = query.to_lowercase();
        let qwords: Vec<_> = lower.split_whitespace().collect();
        let rank = Query::new(Bm25::LUCENE, &local, vocabulary, &qwords)
            .rank(stats, stats.units().filter(|i| chunks.level[i.get()] == 0));
        let ranked: Vec<Value> = rank
            .iter()
            .take(3)
            .map(|(ix, score)| {
                let s = chunks.spans.get(*ix);
                json!({"chunk": ix.get(), "score": score, "span": [s.start, s.end], "text": &clean.text()[s.range()]})
            })
            .collect();
        let abbr = doc.get::<Abbreviations>().unwrap();
        let edges = doc.get::<Edges<Abbreviations, Abbreviations>>().unwrap();
        let edge_samples: Vec<Value> = edges
            .iter()
            .take(10)
            .map(|(_, e)| {
                let a = abbr.spans.get(e.from);
                let b = abbr.spans.get(e.to);
                json!({
                    "from": &clean.text()[a.range()],
                    "to": &clean.text()[b.range()],
                    "kind": edges.kinds.label(e.kind),
                    "score": e.score,
                })
            })
            .collect();
        let entities = doc.get::<Entities>().unwrap();
        let locations = doc.get::<Locations>().unwrap();
        let (unfiltered_locations,) = GeocodeAnnotator::new(PopulationPriorGeocoder::new(BundledGazetteer::get()))
            .annotate((clean, entities))
            .map_err(fail)?;
        let rejected_letters = unfiltered_locations.len() - locations.len();
        for (i, span) in locations.spans.iter() {
            let s = clean.text()[span.range()].trim();
            assert!(!(s.len() == 1 && s.as_bytes()[0].is_ascii_alphabetic()), "location {}", i.get());
        }
        let expected = if name == "dublin" { (44, 14, 1, 1) } else { (5, 17, 17, 74) };
        assert_eq!((locations.len(), rejected_letters, count::<Structure>(doc), edges.len()), expected);
        let location_samples: Vec<Value> = (0..locations.len())
            .take(12)
            .map(|i| {
                let (s, e, confidence) = locations.resolution(Ix::new(i as u32));
                json!({"mention": &clean.text()[s.range()], "entry": format!("{e:?}"), "confidence": confidence})
            })
            .collect();
        let counts = json!({
            "layers": layers.len(),
            "structure": count::<Structure>(doc),
            "sentences": count::<Sentences>(doc),
            "tokens": tokens.len(),
            "subwords": count::<Subwords>(doc),
            "entities": count::<Entities>(doc),
            "locations": locations.len(),
            "regions": count::<Regions>(doc),
            "numbers": count::<Numbers>(doc),
            "temporal": count::<Temporals>(doc),
            "pii": count::<Pii>(doc),
            "abbreviations": abbr.len(),
            "abbreviation_edges": edges.len(),
            "chunks_all_levels": chunks.len(),
            "leaf_chunks": chunks.level.iter().filter(|&&l| l == 0).count(),
            "levels": count::<Levels>(doc),
            "sentence_vectors": sentvec.len(),
            "chunk_vectors": chunkvec.len(),
            "vector_dimension": chunkvec.dim,
            "vocabulary": vocabulary.len(),
            "rejected_single_letter_mentions": rejected_letters,
            "over_budget_leaves": chunks.over_budget.iter().zip(&chunks.level).filter(|&(over, level)| *over && *level == 0).count(),
            "over_budget_parents": chunks.over_budget.iter().zip(&chunks.level).filter(|&(over, level)| *over && *level > 0).count(),
        });
        let vec_provenance = doc.provenance(Key::<Vectors<Chunks>>::DEFAULT.id()).unwrap();
        let model_info = vec_provenance
            .model
            .as_ref()
            .map(|m| json!({"id": m.id, "revision": m.revision, "sha256": m.sha256}))
            .unwrap();
        let execution_info = vec_provenance
            .execution
            .as_ref()
            .map(|e| json!({"backend": e.backend, "device": e.device, "precision": e.precision}))
            .unwrap();
        let item = json!({
            "name": name,
            "input": input,
            "source_sha256": source_digest,
            "source_bytes": doc.bytes().len(),
            "clean_bytes": clean.text().len(),
            "elapsed_ms": elapsed_ms,
            "token_source_mapping_checks": tokens.len(),
            "counts": counts,
            "model": model_info,
            "execution": execution_info,
            "sentence_samples": sample_spans::<Sentences>(doc),
            "entity_samples": sample_spans::<Entities>(doc),
            "location_samples": location_samples,
            "graph_edge_samples": edge_samples,
            "layers": layers,
            "retrieval": {
                "query": query,
                "scope": "BM25 over this document's leaf chunks; English profile folded terms; ASCII queries lowercased",
                "top": ranked,
            },
            "not_exercised_layers": [
                "opennlp:pos",
                "opennlp:lemmas",
                "opennlp:dependencies",
                "coreference groups",
                "dependency relation extractions",
                "turbo:opinions",
                "turbo:pragmatics",
            ],
        });
        let reference_item = json!({
            "name": name,
            "source_sha256": source_digest,
            "source_bytes": doc.bytes().len(),
            "token_source_mapping_checks": tokens.len(),
            "counts": item["counts"],
            "model": item["model"],
            "execution": item["execution"],
        });
        println!(
            "{}",
            serde_json::to_string(&json!({"name": name, "counts": item["counts"], "elapsed_ms": elapsed_ms})).unwrap()
        );
        write_pretty_json(&Path::new(&output).join(format!("{name}-trace.json")), &item)?;
        results.push(item);
        reference_documents.push(reference_item);
    }
    assert!(term_profile_probe.is_none());
    let checkout_commit = std::env::var("TURBO_TRACE_COMMIT").map_err(fail)?;
    let output_data = json!({
        "checkout_commit": checkout_commit,
        "reference_base_commit": "272b9298",
        "reference_patch_commits": ["98bee392", "c958b17d", "cc62410b"],
        "pipeline_plan": pipeline.plan(),
        "normalized_term_integration_probe": term_profile_probe,
        "documents": results,
    });
    let reference_data = json!({
        "checkout_commit": checkout_commit,
        "reference_base_commit": "272b9298",
        "reference_patch_commits": ["98bee392", "c958b17d", "cc62410b"],
        "normalized_term_integration_probe": term_profile_probe,
        "documents": reference_documents,
    });
    write_pretty_json(&Path::new(&output).join("trace.json"), &output_data)?;
    write_pretty_json(&Path::new(&output).join("reference.json"), &reference_data)?;
    Ok(())
}
