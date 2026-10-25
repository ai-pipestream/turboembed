//! Token-level tabular (`tsv` and `csv`) [`DocumentFormatter`] implementations.

use std::fmt::Write as _;
use std::io::Write;

use turbo_analysis::{Document, Ix, Span};
use turbo_analysis_layers::{Entities, Lemmas, PosTags, Sentences, Stems, Terms, Tokens};
use turbo_stopword_layers::Stopwords;

use super::{DocumentFormatter, ExportOffsetMode, FormatExportError, OutputFormatDescriptor, sanitize_single_line};

const REQUIRED_LAYERS: &[&str] = &["opennlp:tokens"];
const HEADER_COLUMNS: [&str; 10] =
    ["sentence", "token", "start", "end", "text", "pos", "lemma", "stem", "entity", "stopword"];

/// Tab-separated token table (`tsv`) exporter.
///
/// Emits header `sentence\ttoken\tstart\tend\ttext\tpos\tlemma\tstem\tentity\tstopword\n`
/// followed by one LF-terminated row per token, sanitizing `\t`, `\r`, and `\n`
/// inside field values to spaces.
#[derive(Copy, Clone, Debug, Default)]
pub struct TsvFormatter;

/// RFC 4180 comma-separated token table (`csv`) exporter.
///
/// Emits header `sentence,token,start,end,text,pos,lemma,stem,entity,stopword\r\n`
/// followed by one CRLF-terminated row per token, quoting fields that contain
/// `,`, `"`, `\r`, or `\n` and doubling embedded `"` characters.
#[derive(Copy, Clone, Debug, Default)]
pub struct CsvFormatter;

struct TokenRow {
    sentence: usize,
    token: usize,
    start: u32,
    end: u32,
    text: String,
    pos: String,
    lemma: String,
    stem: String,
    entity: String,
    stopword: bool,
}

impl DocumentFormatter for TsvFormatter {
    fn descriptor(&self) -> OutputFormatDescriptor {
        OutputFormatDescriptor {
            format_id: "tsv",
            display_name: "Token TSV Table",
            media_type: "text/tab-separated-values; charset=utf-8",
            file_extension: "tsv",
            binary: false,
            lossless: false,
            required_layers: REQUIRED_LAYERS,
            description: "Tab-separated token table with sentence index, source offsets, text, POS, lemma, stem, BIO entity tag, and stopword flag.",
        }
    }

    fn format_document(
        &self,
        _doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
        out: &mut dyn Write,
    ) -> Result<(), FormatExportError> {
        let rows = collect_token_rows(doc, offsets)?;
        let mut buf = String::new();
        buf.push_str(&HEADER_COLUMNS.join("\t"));
        buf.push('\n');
        for r in rows {
            let _ = writeln!(
                buf,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                r.sentence,
                r.token,
                r.start,
                r.end,
                sanitize_single_line(&r.text),
                sanitize_single_line(&r.pos),
                sanitize_single_line(&r.lemma),
                sanitize_single_line(&r.stem),
                sanitize_single_line(&r.entity),
                r.stopword,
            );
        }
        out.write_all(buf.as_bytes())?;
        Ok(())
    }
}

impl DocumentFormatter for CsvFormatter {
    fn descriptor(&self) -> OutputFormatDescriptor {
        OutputFormatDescriptor {
            format_id: "csv",
            display_name: "Token CSV Table (RFC 4180)",
            media_type: "text/csv; charset=utf-8",
            file_extension: "csv",
            binary: false,
            lossless: false,
            required_layers: REQUIRED_LAYERS,
            description: "RFC 4180 CSV token table with CRLF line endings, doubled quote escaping, source offsets, POS, lemma, stem, entity, and stopword columns.",
        }
    }

    fn format_document(
        &self,
        _doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
        out: &mut dyn Write,
    ) -> Result<(), FormatExportError> {
        let rows = collect_token_rows(doc, offsets)?;
        let mut buf = String::new();
        buf.push_str(&HEADER_COLUMNS.join(","));
        buf.push_str("\r\n");
        for r in rows {
            let fields = [
                r.sentence.to_string(),
                r.token.to_string(),
                r.start.to_string(),
                r.end.to_string(),
                r.text,
                r.pos,
                r.lemma,
                r.stem,
                r.entity,
                r.stopword.to_string(),
            ];
            for (i, field) in fields.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                write_csv_field(&mut buf, field);
            }
            buf.push_str("\r\n");
        }
        out.write_all(buf.as_bytes())?;
        Ok(())
    }
}

fn write_csv_field(out: &mut String, field: &str) {
    if field.bytes().any(|b| matches!(b, b',' | b'"' | b'\r' | b'\n')) {
        out.push('"');
        for ch in field.chars() {
            if ch == '"' {
                out.push_str("\"\"");
            } else {
                out.push(ch);
            }
        }
        out.push('"');
    } else {
        out.push_str(field);
    }
}

fn collect_token_rows(doc: &Document, offsets: ExportOffsetMode) -> Result<Vec<TokenRow>, FormatExportError> {
    let tokens = doc.get::<Tokens>().ok_or(FormatExportError::MissingLayer("opennlp:tokens"))?;
    let tok_view = tokens.spans.view();
    let source_text = doc.source();

    let sentences = doc.get::<Sentences>();
    let pos = doc.get::<PosTags>();
    let lemmas = doc.get::<Lemmas>();
    let stems = doc.get::<Stems>();
    let terms = doc.get::<Terms>();
    let entities = doc.get::<Entities>();
    let stopwords = doc.get::<Stopwords>();

    let tok_len = tokens.spans.len();
    let mut bio_tags = vec!["O".to_owned(); tok_len];
    if let Some(ents) = entities {
        let ent_view = ents.spans.view();
        for (e_ix, _span) in ents.spans.iter() {
            let row = e_ix.get();
            let label = ents.kinds.label(ents.kind[row]);
            let range = ents.tokens[row];
            let start_t = range.start.get();
            let end_t = range.end.get().min(tok_len);
            let frags_src: Vec<Span> = ents.spans.fragments(e_ix).map(|frag| doc.to_source(ent_view, frag)).collect();
            let mut began = false;
            for (t_idx, slot) in bio_tags[start_t..end_t].iter_mut().enumerate() {
                let tok_src = doc.to_source(tok_view, tokens.spans.get(Ix::new((start_t + t_idx) as u32)));
                let overlaps_frag = frags_src.iter().any(|f| f.start < tok_src.end && f.end > tok_src.start);
                if overlaps_frag && slot == "O" {
                    let prefix = if !began {
                        began = true;
                        "B"
                    } else {
                        "I"
                    };
                    *slot = format!("{prefix}-{label}");
                }
            }
        }
    }

    let sent_src_spans: Vec<Span> = sentences
        .filter(|_| tokens.sentence.is_none())
        .map(|sents| {
            let s_view = sents.spans.view();
            sents.spans.iter().map(|(_, sp)| doc.to_source(s_view, sp)).collect()
        })
        .unwrap_or_default();

    let mut rows = Vec::with_capacity(tok_len);
    for (t_ix, tok_span) in tokens.spans.iter() {
        let i = t_ix.get();
        let src_span = doc.to_source(tok_view, tok_span);
        let (start, end) = offsets.convert_source_span(doc, src_span)?;
        let text = source_text[src_span.range()].to_owned();

        let sentence = if let Some(sent_col) = &tokens.sentence {
            sent_col[i].get()
        } else if !sent_src_spans.is_empty() {
            let idx = sent_src_spans.partition_point(|s_src| s_src.end <= src_span.start);
            sent_src_spans
                .get(idx)
                .filter(|s_src| src_span.start >= s_src.start && src_span.end <= s_src.end)
                .map_or(0, |_| idx)
        } else {
            0
        };

        let pos_str = pos.map_or_else(String::new, |p| p.tagset.label(p.tags[i]).to_owned());
        let lemma_str = if let Some(l) = lemmas {
            l.lemma.get(i).to_owned()
        } else if let Some(t) = terms
            && let Some(tl) = &t.lemma
        {
            tl.get(i).to_owned()
        } else {
            String::new()
        };
        let stem_str = if let Some(s) = stems {
            s.stem.get(i).to_owned()
        } else if let Some(t) = terms
            && let Some(ts) = &t.stem
        {
            ts.get(i).to_owned()
        } else {
            String::new()
        };
        let stopword = stopwords.is_some_and(|sw| sw.stop.get(i).copied().unwrap_or(false));

        rows.push(TokenRow {
            sentence,
            token: i,
            start,
            end,
            text,
            pos: pos_str,
            lemma: lemma_str,
            stem: stem_str,
            entity: bio_tags[i].clone(),
            stopword,
        });
    }

    Ok(rows)
}
