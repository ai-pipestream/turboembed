//! Corpus encoding and benchmark dataset construction for the 4-arm SPLATE
//! cascade evaluation study (#711).

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use turbo::sparse::{MaxLogPool, SparseOptions as SparseCut, SparseTerm};
use turbo_analysis::{Annotator, Document, ModelRef, View};
use turbo_analysis_layers::Tokens;
use turbo_chunker::terms::{IndexedTerms, TermCounter, TermIndexer, TermStats, Vocabulary};
use turbo_tokenize::WhitespaceTokenizer;

use crate::Result;
use crate::embedder::unit;
use crate::metrics::{Qrel, Qrels};
use crate::set::{Doc, LoadedSet, Query};

/// Encoded representations of one document across all four retrieval arms.
pub struct EncodedDoc {
    pub id: String,
    pub text: String,
    pub bm25_stats: TermStats<View>,
    pub bm25_vocab: Vocabulary,
    pub token_ids: Vec<u32>,
    pub static_vec: Vec<f32>,
    pub token_vecs: Vec<f32>,
    pub splade_terms: Vec<SparseTerm>,
}

/// Encoded representations of one query across all four retrieval arms.
#[derive(Clone, Debug)]
pub struct EncodedQuery {
    pub id: String,
    pub text: String,
    pub bm25_terms: Vec<String>,
    pub token_ids: Vec<u32>,
    pub static_vec: Vec<f32>,
    pub token_vecs: Vec<f32>,
    pub splade_terms: Vec<SparseTerm>,
    /// Per-query base representation encoding time in nanoseconds for Arms 1..=4
    /// (before Arm 4's SPLATE linear adapter projection).
    pub encode_ns: [u64; 4],
}

/// Complete encoded corpus and query set ready for indexing and evaluation.
pub struct EncodedCorpus {
    pub set_name: String,
    pub dim: usize,
    pub vocab_size: usize,
    pub model: ModelRef,
    /// Row-major `[vocab_size, dim]` unit basis vectors for the vocabulary.
    pub vocab_basis: Vec<f32>,
    pub docs: Vec<EncodedDoc>,
    pub queries: Vec<EncodedQuery>,
}

const MAX_DOC_TOKENS: usize = 48;
const MAX_QUERY_TOKENS: usize = 16;

fn fnv1a_64(bytes: &[u8], seed: u64) -> u64 {
    turbo_analysis::Fnv1a::with_seed(turbo_analysis::FNV1A_OFFSET_BASIS ^ seed).bytes(bytes).finish()
}

fn stem_key(word: &str) -> &str {
    for suffix in [
        "ations", "ation", "ingly", "edly", "ments", "ment", "ities", "ity", "ness", "ings", "ing", "ers", "ies", "ed",
        "es", "ly", "er", "s",
    ] {
        if word.len() >= suffix.len() + 4 && word.ends_with(suffix) {
            return &word[..word.len() - suffix.len()];
        }
    }
    match word.char_indices().nth(5) {
        Some((byte_idx, _)) => &word[..byte_idx],
        None => word,
    }
}

fn seeded_coords(seed: u64, dim: usize, out: &mut [f32], weight: f32) {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    for slot in out.iter_mut().take(dim) {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let val = ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
        *slot += weight * val;
    }
}

fn is_stopword(w: &str) -> bool {
    const STOPS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "how",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
        "which", "with",
    ];
    STOPS.binary_search(&w).is_ok()
}

fn tokenize_bm25(text: &str) -> Result<(Vocabulary, IndexedTerms, TermStats<View>)> {
    let folded = text.to_ascii_lowercase();
    let doc = Document::new(folded);
    let view = doc.get::<View>().ok_or_else(|| "missing source view".to_owned())?;
    let (tokens,) = WhitespaceTokenizer::default().annotate((view, None)).map_err(|e| e.to_string())?;
    let (vocab, indexed) = TermIndexer.annotate((view, &tokens)).map_err(|e| e.to_string())?;
    let (stats,) = TermCounter::<View>::new().annotate((&tokens, &indexed, &vocab, view)).map_err(|e| e.to_string())?;
    Ok((vocab, indexed, stats))
}

fn extract_query_terms(text: &str) -> Result<Vec<String>> {
    let folded = text.to_ascii_lowercase();
    let doc = Document::new(folded);
    let view = doc.get::<View>().ok_or_else(|| "missing source view".to_owned())?;
    let (tokens,): (Tokens,) = WhitespaceTokenizer::default().annotate((view, None)).map_err(|e| e.to_string())?;
    let (vocab, indexed) = TermIndexer.annotate((view, &tokens)).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for tid in indexed.term.iter().flatten() {
        let w = vocab.get(*tid);
        if !w.is_empty() && !is_stopword(w) {
            out.push(w.to_owned());
        }
    }
    if out.is_empty() {
        for tid in indexed.term.iter().flatten() {
            out.push(vocab.get(*tid).to_owned());
        }
    }
    Ok(out)
}

/// Encodes a [`LoadedSet`] into synthetic BM25, static dense, contextual
/// token-vector, and learned sparse (SPLADE) representations over a shared
/// vocabulary of up to `max_vocab` terms and embedding dimension `dim`.
pub fn encode_loaded_set(
    set: &LoadedSet,
    dim: usize,
    max_vocab: usize,
    splade_doc_cut: &SparseCut,
    splade_query_cut: &SparseCut,
) -> Result<EncodedCorpus> {
    set.check()?;
    if dim == 0 || !dim.is_multiple_of(8) {
        return Err("dim must be a positive multiple of 8".into());
    }
    if max_vocab < 16 {
        return Err("max_vocab must be at least 16".into());
    }

    let mut raw_docs = Vec::with_capacity(set.docs.len());
    let mut df_map: BTreeMap<String, u32> = BTreeMap::new();
    for d in &set.docs {
        let full_text = match &d.title {
            Some(t) if !t.is_empty() => format!("{t}\n{}", d.text),
            _ => d.text.clone(),
        };
        let (vocab, indexed, stats) = tokenize_bm25(&full_text)?;
        for s in vocab.text.iter() {
            if s.len() >= 2 && !is_stopword(s) {
                *df_map.entry(s.to_owned()).or_insert(0) += 1;
            }
        }
        let mut seq = Vec::new();
        for tid in indexed.term.iter().flatten() {
            let w = vocab.get(*tid);
            if w.len() >= 2 && !is_stopword(w) {
                seq.push(w.to_owned());
            }
        }
        raw_docs.push((d.id.clone(), full_text, vocab, stats, seq));
    }

    let mut raw_queries = Vec::with_capacity(set.queries.len());
    for q in &set.queries {
        let t0 = Instant::now();
        let q_terms = extract_query_terms(&q.text)?;
        let bm25_ns = t0.elapsed().as_nanos() as u64;
        for w in &q_terms {
            if w.len() >= 2 && !is_stopword(w) {
                df_map.entry(w.clone()).or_insert(1);
            }
        }
        raw_queries.push((q.id.clone(), q.text.clone(), q_terms, bm25_ns));
    }

    let mut vocab_entries: Vec<(String, u32)> = df_map.into_iter().collect();
    vocab_entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    vocab_entries.truncate(max_vocab);
    vocab_entries.sort_by(|a, b| a.0.cmp(&b.0));
    let vocab_size = vocab_entries.len().max(16);

    let mut term_to_id: HashMap<String, u32> = HashMap::with_capacity(vocab_entries.len());
    let mut term_idf: Vec<f32> = vec![1.0f32; vocab_size];
    let mut stem_groups: HashMap<String, Vec<u32>> = HashMap::new();
    let mut vocab_basis = vec![0.0f32; vocab_size * dim];
    let n_docs_f = (set.docs.len() as f32).max(1.0);

    for (idx, (word, df)) in vocab_entries.iter().enumerate() {
        let tid = idx as u32;
        term_to_id.insert(word.clone(), tid);
        let idf = (1.0 + (n_docs_f - *df as f32 + 0.5) / (*df as f32 + 0.5)).ln().max(0.25);
        term_idf[idx] = idf;
        let stem = stem_key(word).to_owned();
        stem_groups.entry(stem.clone()).or_default().push(tid);
        let row = &mut vocab_basis[idx * dim..(idx + 1) * dim];
        seeded_coords(fnv1a_64(word.as_bytes(), 0x1001), dim, row, 0.45);
        seeded_coords(fnv1a_64(stem.as_bytes(), 0x2002), dim, row, 0.55);
        unit(row);
    }
    for idx in vocab_entries.len()..vocab_size {
        let row = &mut vocab_basis[idx * dim..(idx + 1) * dim];
        seeded_coords(fnv1a_64(&idx.to_le_bytes(), 0x3003), dim, row, 1.0);
        unit(row);
    }

    type EncodedText = (Vec<u32>, Vec<f32>, Vec<f32>, Vec<SparseTerm>, [u64; 3]);
    let build_representations = |words: &[String], max_toks: usize, cut: &SparseCut| -> Result<EncodedText> {
        let t_tok = Instant::now();
        let mut tids: Vec<u32> = words
            .iter()
            .filter_map(|w| {
                term_to_id
                    .get(w.as_str())
                    .copied()
                    .or_else(|| Some((fnv1a_64(w.as_bytes(), 0x55) as usize % vocab_size) as u32))
            })
            .take(max_toks)
            .collect();
        if tids.is_empty() {
            tids.push(0);
        }
        let n = tids.len();
        let mut token_vecs = vec![0.0f32; n * dim];
        for i in 0..n {
            let out = &mut token_vecs[i * dim..(i + 1) * dim];
            let base = &vocab_basis[tids[i] as usize * dim..(tids[i] as usize + 1) * dim];
            for (dst, &src) in out.iter_mut().zip(base) {
                *dst = 0.84 * src;
            }
            if i > 0 {
                let prev = &vocab_basis[tids[i - 1] as usize * dim..(tids[i - 1] as usize + 1) * dim];
                for (dst, &src) in out.iter_mut().zip(prev) {
                    *dst += 0.08 * src;
                }
            }
            if i + 1 < n {
                let next = &vocab_basis[tids[i + 1] as usize * dim..(tids[i + 1] as usize + 1) * dim];
                for (dst, &src) in out.iter_mut().zip(next) {
                    *dst += 0.08 * src;
                }
            }
            unit(out);
        }
        let tok_ns = t_tok.elapsed().as_nanos() as u64;

        let t_static = Instant::now();
        let mut static_vec = vec![0.0f32; dim];
        for (pos, tok) in token_vecs.chunks_exact(dim).enumerate() {
            let w = term_idf[tids[pos] as usize];
            for (d, &v) in static_vec.iter_mut().zip(tok) {
                *d += w * v;
            }
        }
        unit(&mut static_vec);
        let static_ns = tok_ns + t_static.elapsed().as_nanos() as u64;

        let t_splade = Instant::now();
        let mut pool = MaxLogPool::new(vocab_size);
        let mut logits = vec![0.0f32; vocab_size];
        for (pos, w) in words.iter().take(max_toks).enumerate() {
            logits.fill(0.0);
            let tid = tids[pos] as usize;
            let pos_decay = 1.0 / (1.0 + 0.015 * pos as f32);
            let idf_gain = term_idf[tid];
            logits[tid] = (1.5 + 0.55 * idf_gain) * pos_decay;
            let stem = stem_key(w);
            if let Some(neighbors) = stem_groups.get(stem) {
                for &nb in neighbors.iter().take(6) {
                    let nb_ix = nb as usize;
                    if nb_ix != tid {
                        let nb_gain = (0.75 + 0.35 * term_idf[nb_ix]) * pos_decay;
                        logits[nb_ix] = logits[nb_ix].max(nb_gain);
                    }
                }
            }
            pool.observe(&logits).map_err(|e| e.to_string())?;
        }
        let splade_terms = pool.terms(cut).map_err(|e| e.to_string())?;
        let splade_ns = tok_ns + t_splade.elapsed().as_nanos() as u64;
        Ok((tids, static_vec, token_vecs, splade_terms, [static_ns, splade_ns, tok_ns]))
    };

    let mut docs = Vec::with_capacity(raw_docs.len());
    for (id, text, bm25_vocab, bm25_stats, seq) in raw_docs {
        let (token_ids, static_vec, token_vecs, splade_terms, _) =
            build_representations(&seq, MAX_DOC_TOKENS, splade_doc_cut)?;
        docs.push(EncodedDoc { id, text, bm25_stats, bm25_vocab, token_ids, static_vec, token_vecs, splade_terms });
    }

    let mut queries = Vec::with_capacity(raw_queries.len());
    for (id, text, bm25_terms, bm25_ns) in raw_queries {
        let (token_ids, static_vec, token_vecs, splade_terms, [static_ns, splade_ns, tok_ns]) =
            build_representations(&bm25_terms, MAX_QUERY_TOKENS, splade_query_cut)?;
        let encode_ns = [bm25_ns, bm25_ns + static_ns, bm25_ns + splade_ns, bm25_ns + tok_ns];
        queries.push(EncodedQuery { id, text, bm25_terms, token_ids, static_vec, token_vecs, splade_terms, encode_ns });
    }

    let model = ModelRef {
        id: "turbo-eval-splate".into(),
        revision: "v1".into(),
        sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
    };
    Ok(EncodedCorpus { set_name: set.name.clone(), dim, vocab_size, model, vocab_basis, docs, queries })
}

type TopicSpec = (&'static str, &'static [(&'static str, &'static str)], &'static [&'static str]);

const TOPICS: [TopicSpec; 8] = [
    (
        "cardiology",
        &[
            ("hypertension", "hypertensive"),
            ("vascular", "vasculature"),
            ("arterial", "arteries"),
            ("myocardial", "myocardium"),
        ],
        &["systolic", "endothelium", "ventricle", "ischemia", "stenosis", "coronary"],
    ),
    (
        "immunology",
        &[
            ("vaccination", "vaccinated"),
            ("antibody", "antibodies"),
            ("pathogen", "pathogenic"),
            ("lymphocyte", "lymphocytes"),
        ],
        &["antigen", "cytokine", "mucosal", "neutralizing", "humoral", "adjuvant"],
    ),
    (
        "genomics",
        &[
            ("transcription", "transcriptional"),
            ("chromatin", "chromosomal"),
            ("methylation", "methylated"),
            ("sequencing", "sequenced"),
        ],
        &["promoter", "enhancer", "epigenetic", "nuclease", "isoform", "splicing"],
    ),
    (
        "neurology",
        &[
            ("synaptic", "synapses"),
            ("dopamine", "dopaminergic"),
            ("cortical", "cortex"),
            ("neuropathy", "neuropathic"),
        ],
        &["axonal", "myelin", "plasticity", "hippocampal", "striatal", "glial"],
    ),
    (
        "oncology",
        &[
            ("carcinoma", "carcinogenic"),
            ("metastasis", "metastatic"),
            ("angiogenesis", "angiogenic"),
            ("apoptosis", "apoptotic"),
        ],
        &["oncogene", "sarcoma", "chemotherapy", "remission", "biopsy", "stromal"],
    ),
    (
        "virology",
        &[
            ("replication", "replicating"),
            ("capsid", "capsids"),
            ("glycoprotein", "glycoproteins"),
            ("zoonotic", "zoonosis"),
        ],
        &["polymerase", "envelope", "virion", "tropism", "latency", "interferon"],
    ),
    (
        "pharmacology",
        &[
            ("bioavailability", "bioavailable"),
            ("clearance", "clearing"),
            ("metabolite", "metabolites"),
            ("inhibitor", "inhibitors"),
        ],
        &["hepatic", "cytochrome", "pharmacokinetic", "affinity", "dosing", "half-life"],
    ),
    (
        "epidemiology",
        &[
            ("incidence", "incident"),
            ("prevalence", "prevalent"),
            ("mortality", "mortalities"),
            ("longitudinal", "longitudinally"),
        ],
        &["cohort", "confounding", "stratified", "hazard", "surveillance", "outbreak"],
    ),
];

/// Builds a self-contained multi-topic benchmark dataset with graded relevance
/// (`grade = 2` for morphological/inflected + contextual phrase matches,
/// `grade = 1` for partial lexical matches, and hard-negative distractor documents).
pub fn synthetic_benchmark_set(docs_per_topic: usize, queries_per_topic: usize) -> LoadedSet {
    let mut docs = Vec::new();
    let mut queries = Vec::new();
    let mut qrels = Vec::new();
    let dpt = docs_per_topic.max(4);
    let qpt = queries_per_topic.clamp(1, 4);

    for (t_idx, &(domain, morph_pairs, terms)) in TOPICS.iter().enumerate() {
        for i in 0..dpt {
            let doc_id = format!("d_{domain}_{i:03}");
            let (base_w, inflected_w) = morph_pairs[i % morph_pairs.len()];
            let (next_base, _next_inflected) = morph_pairs[(i + 1) % morph_pairs.len()];
            let t1 = terms[i % terms.len()];
            let t2 = terms[(i + 1) % terms.len()];
            let other_domain = TOPICS[(t_idx + 1) % TOPICS.len()].0;
            let _ = inflected_w;
            let text = if i % 4 == 0 {
                // Grade 2: contains `base_w`, `next_base`, `t1`, `t2`.
                format!(
                    "Clinical study of {domain} {base_w} and {next_base} regulation via {t1} and {t2} signaling pathways."
                )
            } else if i % 4 == 1 {
                // Grade 1: contains `base_w` and `t1` only.
                format!("Experimental analysis of {domain} {base_w} response and {t1} modulation during progression.")
            } else if i % 4 == 2 {
                // Hard negative distractor: shares `domain` and `t1` without the target stem pair.
                format!(
                    "Observational review comparing {domain} {t1} levels with {other_domain} baselines under control protocols."
                )
            } else {
                // Distractor sharing `domain` and `t2`.
                format!(
                    "Background control cohort measuring routine {domain} {t2} markers without specific {other_domain} intervention."
                )
            };
            docs.push(Doc { id: doc_id, title: Some(format!("study {i}")), text });
        }
        for (q_i, &(_base_w, inflected_w)) in morph_pairs.iter().take(qpt).enumerate() {
            let q_id = format!("q_{domain}_{q_i}");
            let (_, next_inflected) = morph_pairs[(q_i + 1) % morph_pairs.len()];
            let t1 = terms[q_i % terms.len()];
            let t2 = terms[(q_i + 1) % terms.len()];
            // Query uses `domain`, the inflected forms (`inflected_w`, `next_inflected`), and `t1`/`t2`.
            queries
                .push(Query { id: q_id.clone(), text: format!("{domain} {inflected_w} {next_inflected} {t1} {t2}") });
            for i in 0..dpt {
                if i % morph_pairs.len() == q_i {
                    let doc_id = format!("d_{domain}_{i:03}");
                    if i % 4 == 0 {
                        qrels.push(Qrel { query: q_id.clone(), doc: doc_id, grade: 2 });
                    } else if i % 4 == 1 {
                        qrels.push(Qrel { query: q_id.clone(), doc: doc_id, grade: 1 });
                    }
                }
            }
        }
    }
    LoadedSet {
        name: format!("splate-synth-{}d-{}q", docs.len(), queries.len()),
        sha256: "a".repeat(64),
        docs,
        queries,
        qrels: Qrels::new(qrels).expect("valid synthetic qrels"),
        spans: None,
        unplaced: Vec::new(),
        absent: Vec::new(),
    }
}
