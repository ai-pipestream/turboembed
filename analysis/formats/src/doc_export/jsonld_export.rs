//! Deterministic JSON-LD / UIMA-style [`DocumentFormatter`].

use std::fmt::Write as _;
use std::io::Write;

use turbo_analysis::{Document, Edges, Ix, Layer as _};
use turbo_analysis_layers::{
    Categories, Chunks, Dependencies, Entities, Language, Lemmas, MorphFeatures, Pii, PosTags, Sentences, Stems,
    Tokens, WordDependencies, WordLemmas, WordPosTags, Words,
};
use turbo_stopword_layers::Stopwords;

use super::{DocumentFormatter, ExportOffsetMode, FormatExportError, OutputFormatDescriptor};

const REQUIRED_LAYERS: &[&str] = &[];

/// Deterministic JSON-LD / UIMA-style document layer and provenance exporter.
///
/// Emits a canonical JSON-LD object with `@context`, `@id`, `@type`, `docId`,
/// `offsetMode`, `sourceText`, `provenance` records in layer order, and typed
/// `layers` arrays whose spans are projected to [`turbo_analysis::ViewId::SOURCE`]
/// and converted to the caller's [`ExportOffsetMode`].
#[derive(Copy, Clone, Debug, Default)]
pub struct JsonLdFormatter;

impl DocumentFormatter for JsonLdFormatter {
    fn descriptor(&self) -> OutputFormatDescriptor {
        OutputFormatDescriptor {
            format_id: "jsonld",
            display_name: "JSON-LD / UIMA Typed Layer Graph",
            media_type: "application/ld+json; charset=utf-8",
            file_extension: "jsonld",
            binary: false,
            lossless: true,
            required_layers: REQUIRED_LAYERS,
            description: "Deterministic JSON-LD / UIMA-style typed layer and provenance export with ExportOffsetMode spans.",
        }
    }

    fn format_document(
        &self,
        doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
        out: &mut dyn Write,
    ) -> Result<(), FormatExportError> {
        let mut buf = String::new();
        let source_text = doc.source();

        buf.push_str("{\n");
        buf.push_str("  \"@context\": \"https://turboembed.ai/ns/jsonld/v1\",\n");
        buf.push_str("  \"@type\": \"turbo:AnnotatedDocument\",\n");
        let _ = writeln!(buf, "  \"@id\": {},", json_string(&format!("urn:turbo:document:{doc_id}")));
        let _ = writeln!(buf, "  \"docId\": {},", json_string(doc_id));
        let _ = writeln!(buf, "  \"offsetMode\": {},", json_string(offsets.as_str()));
        let _ = writeln!(buf, "  \"sourceText\": {},", json_string(source_text));

        // Provenance inventory in deterministic layer order.
        buf.push_str("  \"provenance\": [\n");
        let layer_keys: Vec<_> = doc.layers().collect();
        for (idx, key) in layer_keys.iter().enumerate() {
            let prov = doc.provenance(*key);
            let by = prov.map_or("", |p| p.annotator);
            let version = prov.map_or("", |p| p.version);
            let comma = if idx + 1 < layer_keys.len() { "," } else { "" };
            let _ = writeln!(
                buf,
                "    {{\"layer\": {}, \"instance\": {}, \"by\": {}, \"version\": {}}}{comma}",
                json_string(&key.layer.to_string()),
                json_string(key.name),
                json_string(by),
                json_string(version),
            );
        }
        buf.push_str("  ],\n");

        buf.push_str("  \"layers\": {\n");
        let mut sections: Vec<String> = Vec::new();

        if let Some(lang) = doc.get::<Language>() {
            let mut s = String::from("    \"turbo:language\": [");
            for (i, (code, score)) in lang.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                let _ = write!(
                    s,
                    "{{\"@type\": \"turbo:Language\", \"code\": {}, \"score\": {}}}",
                    json_string(code),
                    format_json_f64(score),
                );
            }
            s.push(']');
            sections.push(s);
        }

        if let Some(cats) = doc.get::<Categories>() {
            let mut s = String::from("    \"turbo:categories\": [");
            for (i, row) in cats.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                let _ = write!(
                    s,
                    "{{\"@type\": \"turbo:Category\", \"best\": {}, \"score\": {}}}",
                    json_string(row.best),
                    format_json_f64(row.best_score()),
                );
            }
            s.push(']');
            sections.push(s);
        }

        if let Some(sents) = doc.get::<Sentences>() {
            let view = sents.spans.view();
            let mut s = String::from("    \"opennlp:sentences\": [");
            for (i, (_, span)) in sents.spans.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                let src = doc.to_source(view, span);
                let (start, end) = offsets.convert_source_span(doc, src)?;
                let text = &source_text[src.range()];
                let _ = write!(
                    s,
                    "{{\"@type\": \"opennlp:Sentence\", \"index\": {i}, \"start\": {start}, \"end\": {end}, \"text\": {}}}",
                    json_string(text),
                );
            }
            s.push(']');
            sections.push(s);
        }

        if let Some(tokens) = doc.get::<Tokens>() {
            let view = tokens.spans.view();
            let pos = doc.get::<PosTags>();
            let lemmas = doc.get::<Lemmas>();
            let stems = doc.get::<Stems>();
            let feats = doc.get::<MorphFeatures<Tokens>>();
            let deps = doc.get::<Dependencies<Tokens>>();
            let stopwords = doc.get::<Stopwords>();

            let mut s = String::from("    \"opennlp:tokens\": [");
            for (t_ix, span) in tokens.spans.iter() {
                let i = t_ix.get();
                if i > 0 {
                    s.push_str(", ");
                }
                let src = doc.to_source(view, span);
                let (start, end) = offsets.convert_source_span(doc, src)?;
                let text = &source_text[src.range()];
                let sent_idx = tokens.sentence.as_ref().map_or(0, |sc| sc[i].get());
                let pos_tag = pos.map_or("", |p| p.tagset.label(p.tags[i]));
                let lemma_val = lemmas.map_or("", |l| l.lemma.get(i));
                let stem_val = stems.map_or("", |st| st.stem.get(i));
                let feat_val = feats.map_or("", |f| f.feats.get(i));
                let stop_val = stopwords.is_some_and(|sw| sw.stop.get(i).copied().unwrap_or(false));
                let (head_json, rel_val) = match deps {
                    Some(d) => match d.head[i] {
                        Some(h) => (h.get().to_string(), d.relations.label(d.relation[i])),
                        None => ("null".to_owned(), d.relations.label(d.relation[i])),
                    },
                    None => ("null".to_owned(), ""),
                };
                let _ = write!(
                    s,
                    "{{\"@type\": \"opennlp:Token\", \"index\": {i}, \"sentence\": {sent_idx}, \"start\": {start}, \"end\": {end}, \"text\": {}, \"pos\": {}, \"lemma\": {}, \"stem\": {}, \"feats\": {}, \"head\": {head_json}, \"deprel\": {}, \"stopword\": {stop_val}}}",
                    json_string(text),
                    json_string(pos_tag),
                    json_string(lemma_val),
                    json_string(stem_val),
                    json_string(feat_val),
                    json_string(rel_val),
                );
            }
            s.push(']');
            sections.push(s);

            if let Some(words) = doc.get::<Words>() {
                let w_pos = doc.get::<WordPosTags>();
                let w_lemmas = doc.get::<WordLemmas>();
                let w_feats = doc.get::<MorphFeatures<Words>>();
                let w_deps = doc.get::<WordDependencies>();
                let view_text = doc.text(view);
                let mut ws = String::from("    \"turbo:words\": [");
                for i in 0..words.len() {
                    if i > 0 {
                        ws.push_str(", ");
                    }
                    let w_ix = Ix::<Words>::new(i as u32);
                    let form = words.form(w_ix, view_text, tokens);
                    let t_range = words.tokens[i];
                    let pos_tag = w_pos.map_or("", |p| p.tagset.label(p.tags[i]));
                    let lemma_val = w_lemmas.map_or("", |l| l.lemma.get(i));
                    let feat_val = w_feats.map_or("", |f| f.feats.get(i));
                    let (head_json, rel_val) = match w_deps {
                        Some(d) => match d.head[i] {
                            Some(h) => (h.get().to_string(), d.relations.label(d.relation[i])),
                            None => ("null".to_owned(), d.relations.label(d.relation[i])),
                        },
                        None => ("null".to_owned(), ""),
                    };
                    let _ = write!(
                        ws,
                        "{{\"@type\": \"turbo:Word\", \"index\": {i}, \"tokenStart\": {}, \"tokenEnd\": {}, \"form\": {}, \"pos\": {}, \"lemma\": {}, \"feats\": {}, \"head\": {head_json}, \"deprel\": {}}}",
                        t_range.start.get(),
                        t_range.end.get(),
                        json_string(form),
                        json_string(pos_tag),
                        json_string(lemma_val),
                        json_string(feat_val),
                        json_string(rel_val),
                    );
                }
                ws.push(']');
                sections.push(ws);
            }
        }

        if let Some(entities) = doc.get::<Entities>() {
            let view = entities.spans.view();
            let mut s = String::from("    \"opennlp:entities\": [");
            for (e_ix, env) in entities.spans.iter() {
                let row = e_ix.get();
                if row > 0 {
                    s.push_str(", ");
                }
                let src_env = doc.to_source(view, env);
                let (start, end) = offsets.convert_source_span(doc, src_env)?;
                let mut frags_json = String::from("[");
                let mut parts = Vec::new();
                for (f_i, frag) in entities.spans.fragments(e_ix).enumerate() {
                    if f_i > 0 {
                        frags_json.push_str(", ");
                    }
                    let src_frag = doc.to_source(view, frag);
                    let (fs, fe) = offsets.convert_source_span(doc, src_frag)?;
                    let _ = write!(frags_json, "{{\"start\": {fs}, \"end\": {fe}}}");
                    parts.push(&source_text[src_frag.range()]);
                }
                frags_json.push(']');
                let text = parts.join(" ");
                let label = entities.kinds.label(entities.kind[row]);
                let score = entities.score[row];
                let _ = write!(
                    s,
                    "{{\"@type\": \"opennlp:Entity\", \"index\": {row}, \"kind\": {}, \"start\": {start}, \"end\": {end}, \"fragments\": {frags_json}, \"text\": {}, \"score\": {}}}",
                    json_string(label),
                    json_string(&text),
                    format_json_f64(score),
                );
            }
            s.push(']');
            sections.push(s);
        }

        if let Some(relations) = doc.get::<Edges<Entities, Entities>>() {
            let mut s = String::from("    \"turbo:edges<opennlp:entities,opennlp:entities>\": [");
            for (r_ix, edge) in relations.iter() {
                let row = r_ix.get();
                if row > 0 {
                    s.push_str(", ");
                }
                let label = relations.kinds.label(edge.kind);
                let _ = write!(
                    s,
                    "{{\"@type\": \"turbo:RelationEdge\", \"from\": {}, \"to\": {}, \"kind\": {}, \"score\": {}}}",
                    edge.from.get(),
                    edge.to.get(),
                    json_string(label),
                    format_json_f64(edge.score),
                );
            }
            s.push(']');
            sections.push(s);
        }

        if let Some(pii) = doc.get::<Pii>() {
            let view = pii.spans.view();
            let mut s = String::from("    \"turbo:pii\": [");
            for (i, m) in pii.mentions().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                let src = doc.to_source(view, m.span);
                let (start, end) = offsets.convert_source_span(doc, src)?;
                let _ = write!(
                    s,
                    "{{\"@type\": \"turbo:PiiMention\", \"kind\": {}, \"confidence\": {}, \"start\": {start}, \"end\": {end}, \"normalized\": {}}}",
                    json_string(m.kind.name()),
                    json_string(m.confidence.name()),
                    json_string(m.normalized),
                );
            }
            s.push(']');
            sections.push(s);
        }

        if let Some(chunks) = doc.get::<Chunks>() {
            let view = chunks.spans.view();
            let mut s = String::from("    \"turbo:chunks\": [");
            for (c_ix, span) in chunks.spans.iter() {
                let row = c_ix.get();
                if row > 0 {
                    s.push_str(", ");
                }
                let src = doc.to_source(view, span);
                let (start, end) = offsets.convert_source_span(doc, src)?;
                let _ =
                    write!(s, "{{\"@type\": \"turbo:Chunk\", \"index\": {row}, \"start\": {start}, \"end\": {end}}}",);
            }
            s.push(']');
            sections.push(s);
        }

        buf.push_str(&sections.join(",\n"));
        if !sections.is_empty() {
            buf.push('\n');
        }
        buf.push_str("  }\n}\n");

        out.write_all(buf.as_bytes())?;
        Ok(())
    }
}

fn format_json_f64(v: f64) -> String {
    if v.is_finite() { format!("{v:.6}") } else { "null".to_owned() }
}

/// Formats `s` as an RFC 8259 JSON string literal with all control characters,
/// backslashes, and quotes escaped.
pub fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
