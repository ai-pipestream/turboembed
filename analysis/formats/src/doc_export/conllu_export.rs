//! Universal Dependencies v2 10-column CoNLL-U [`DocumentFormatter`].

use std::fmt::Write as _;
use std::io::Write;

use turbo_analysis::{Document, Ix, IxRange, Key, Layer as _, Span};
use turbo_analysis_layers::{
    Dependencies, Lemmas, MorphFeatures, PosTags, Sentences, Tokens, WordDependencies, WordLemmas, WordPosTags, Words,
};

use super::{
    DocumentFormatter, ExportOffsetMode, FormatExportError, OutputFormatDescriptor, get_bound_layer,
    sanitize_single_line,
};
use crate::error::FormatError;

const REQUIRED_LAYERS: &[&str] = &["opennlp:tokens"];

/// Universal Dependencies v2 10-column CoNLL-U exporter.
///
/// Emits `# newdoc id = <doc_id>`, `# sent_id = <doc_id>-<n>` (or `<n>` when
/// `doc_id` is empty), `# text = <sanitized>`, multiword token `i-j` range
/// lines when [`Words`] expands a surface token into multiple syntactic words,
/// `FORM`, `LEMMA` (from [`WordLemmas`] or [`Lemmas`]), `UPOS` (from
/// [`WordPosTags`] or [`PosTags`]), `FEATS` (from [`MorphFeatures<Words>`] or
/// [`MorphFeatures<Tokens>`]), `HEAD`/`DEPREL` (from [`WordDependencies`] or
/// [`Dependencies<Tokens>`]), and `SpaceAfter=No` / `SpacesAfter=...` in `MISC`
/// computed from source-projected token spans.
#[derive(Copy, Clone, Debug, Default)]
pub struct ConlluFormatter;

impl DocumentFormatter for ConlluFormatter {
    fn descriptor(&self) -> OutputFormatDescriptor {
        OutputFormatDescriptor {
            format_id: "conllu",
            display_name: "CoNLL-U (Universal Dependencies v2)",
            media_type: "text/x-conllu; charset=utf-8",
            file_extension: "conllu",
            binary: false,
            lossless: false,
            required_layers: REQUIRED_LAYERS,
            description: "Universal Dependencies v2 10-column CoNLL-U with multiword tokens, lemmas, POS, morph features, dependencies, and SpaceAfter=No.",
        }
    }

    fn format_document(
        &self,
        doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
        out: &mut dyn Write,
    ) -> Result<(), FormatExportError> {
        let tokens = doc.get::<Tokens>().ok_or(FormatExportError::MissingLayer("opennlp:tokens"))?;
        let tok_key = Key::<Tokens>::DEFAULT.id();
        let tok_len = tokens.spans.len();
        let explicit_words = get_bound_layer::<Words>(doc, tok_key, None)?;
        let has_explicit_words = explicit_words.is_some();
        let default_words;
        let words = match explicit_words {
            Some(w) => w,
            None => {
                default_words = Words::from_tokens(tokens);
                &default_words
            }
        };

        let tok_view = tokens.spans.view();
        let view_text = doc.text(tok_view);
        let source_text = doc.source();

        let word_key = Key::<Words>::DEFAULT.id();
        let word_len = words.len();
        let word_pos =
            if has_explicit_words { get_bound_layer::<WordPosTags>(doc, word_key, Some(word_len))? } else { None };
        let tok_pos = get_bound_layer::<PosTags>(doc, tok_key, Some(tok_len))?;
        let word_lemmas =
            if has_explicit_words { get_bound_layer::<WordLemmas>(doc, word_key, Some(word_len))? } else { None };
        let tok_lemmas = get_bound_layer::<Lemmas>(doc, tok_key, Some(tok_len))?;
        let word_feats = if has_explicit_words {
            get_bound_layer::<MorphFeatures<Words>>(doc, word_key, Some(word_len))?
        } else {
            None
        };
        let tok_feats = get_bound_layer::<MorphFeatures<Tokens>>(doc, tok_key, Some(tok_len))?;
        let word_deps =
            if has_explicit_words { get_bound_layer::<WordDependencies>(doc, word_key, Some(word_len))? } else { None };
        let tok_deps = get_bound_layer::<Dependencies<Tokens>>(doc, tok_key, Some(tok_len))?;

        if tokens.sentence.is_some() {
            let _ = get_bound_layer::<Tokens>(doc, Key::<Sentences>::DEFAULT.id(), None)?;
        }

        let sentence_ranges: Vec<(Span, IxRange<Tokens>)> = match doc.get::<Sentences>() {
            Some(sents) if tokens.sentence.is_some() => (0..sents.spans.len() as u32)
                .map(|i| {
                    let s_ix = Ix::new(i);
                    let src_sent = doc.to_source(sents.spans.view(), sents.spans.get(s_ix));
                    (src_sent, tokens.of(s_ix))
                })
                .collect(),
            Some(sents) if !sents.spans.is_empty() => (0..sents.spans.len() as u32)
                .map(|i| {
                    let s_ix = Ix::new(i);
                    let sent_span = sents.spans.get(s_ix);
                    let src_sent = doc.to_source(sents.spans.view(), sent_span);
                    let (starts, ends) = tokens.spans.columns();
                    let t_start = ends
                        .partition_point(|&e| doc.to_source(tok_view, Span::new(e, e)).start <= src_sent.start)
                        as u32;
                    let t_end = starts
                        .partition_point(|&s| doc.to_source(tok_view, Span::new(s, s)).start < src_sent.end)
                        .max(t_start as usize) as u32;
                    (src_sent, IxRange::new(t_start, t_end))
                })
                .collect(),
            _ => vec![(Span::new(0, source_text.len() as u32), IxRange::new(0, tokens.spans.len() as u32))],
        };

        let mut buf = String::new();
        let clean_doc_id = sanitize_single_line(doc_id);
        if !clean_doc_id.is_empty() {
            let _ = writeln!(buf, "# newdoc id = {clean_doc_id}");
        }

        let mut wrote_any_sentence = false;
        for (s_idx, (src_sent_span, tok_range)) in sentence_ranges.into_iter().enumerate() {
            if tok_range.is_empty() {
                continue;
            }
            if !wrote_any_sentence && clean_doc_id.is_empty() {
                // No newdoc header when doc_id is empty.
            }
            wrote_any_sentence = true;

            if clean_doc_id.is_empty() {
                let _ = writeln!(buf, "# sent_id = {}", s_idx + 1);
            } else {
                let _ = writeln!(buf, "# sent_id = {clean_doc_id}-{}", s_idx + 1);
            }
            let sent_surface = &source_text[src_sent_span.range()];
            let _ = writeln!(buf, "# text = {}", sanitize_single_line(sent_surface));

            let w_start = words.tokens.partition_point(|r| r.end <= tok_range.start);
            let w_end = w_start + words.tokens[w_start..].partition_point(|r| r.start < tok_range.end);

            let mut w_idx = w_start;
            while w_idx < w_end {
                let t_range = words.tokens[w_idx];
                let group_end = w_idx + words.tokens[w_idx..w_end].iter().take_while(|&&r| r == t_range).count();
                let first_tok = tokens.spans.get(t_range.start);
                let last_tok = tokens.spans.get(Ix::new(t_range.end.get() as u32 - 1));
                let tok_view_span = Span::new(first_tok.start, last_tok.end);
                let tok_src_span = doc.to_source(tok_view, tok_view_span);
                let surface = sanitize_conllu_field(&source_text[tok_src_span.range()]);

                let next_tok_ix = t_range.end;
                let gap = if next_tok_ix < tok_range.end {
                    let next_src = doc.to_source(tok_view, tokens.spans.get(next_tok_ix));
                    if next_src.start >= tok_src_span.end {
                        &source_text[tok_src_span.end as usize..next_src.start as usize]
                    } else {
                        ""
                    }
                } else {
                    " "
                };
                let space_misc = match gap {
                    " " => None,
                    "" => Some("SpaceAfter=No".to_owned()),
                    _ => Some(format!("SpacesAfter={}", escape_conllu_spaces(gap))),
                };

                let (off_start, off_end) = offsets.convert_source_span(doc, tok_src_span)?;
                let surface_misc = build_misc(space_misc.as_deref(), offsets, off_start, off_end);

                if group_end - w_idx > 1 {
                    let start_1 = (w_idx - w_start) + 1;
                    let end_1 = group_end - w_start;
                    let _ = writeln!(buf, "{start_1}-{end_1}\t{surface}\t_\t_\t_\t_\t_\t_\t_\t{surface_misc}");
                }

                for cur in w_idx..group_end {
                    let id_1 = (cur - w_start) + 1;
                    let w_ix = Ix::<Words>::new(cur as u32);
                    let raw_form = words.form(w_ix, view_text, tokens);
                    let form = sanitize_conllu_field(raw_form);
                    let tok_idx = t_range.start.get();

                    let lemma_raw = if let Some(wl) = word_lemmas {
                        wl.lemma.get(cur)
                    } else if !has_explicit_words && let Some(tl) = tok_lemmas {
                        tl.lemma.get(tok_idx)
                    } else {
                        "_"
                    };
                    let lemma = sanitize_conllu_field_or_underscore(lemma_raw);

                    let upos_raw = if let Some(wp) = word_pos {
                        wp.tagset.label(wp.tags[cur])
                    } else if !has_explicit_words && let Some(tp) = tok_pos {
                        tp.tagset.label(tp.tags[tok_idx])
                    } else {
                        "_"
                    };
                    let upos = sanitize_conllu_field_or_underscore(upos_raw);

                    let feat_raw = if let Some(wf) = word_feats {
                        wf.feats.get(cur)
                    } else if !has_explicit_words && let Some(tf) = tok_feats {
                        tf.feats.get(tok_idx)
                    } else {
                        "_"
                    };
                    let feat = sanitize_conllu_field_or_underscore(feat_raw);

                    let (head_s, deprel_raw) = if let Some(wd) = word_deps {
                        let h = match wd.head[cur] {
                            None => 0usize,
                            Some(h_ix) if (w_start..w_end).contains(&h_ix.get()) => (h_ix.get() - w_start) + 1,
                            Some(h_ix) => {
                                return Err(FormatError::HeadOutsideSentence { word: cur, head: h_ix.get() }.into());
                            }
                        };
                        (h.to_string(), wd.relations.label(wd.relation[cur]))
                    } else if !has_explicit_words && let Some(td) = tok_deps {
                        let t_start = tok_range.start.get();
                        let t_end = tok_range.end.get();
                        let h = match td.head[tok_idx] {
                            None => 0usize,
                            Some(h_ix) if (t_start..t_end).contains(&h_ix.get()) => (h_ix.get() - t_start) + 1,
                            Some(h_ix) => {
                                return Err(FormatError::HeadOutsideSentence { word: tok_idx, head: h_ix.get() }.into());
                            }
                        };
                        (h.to_string(), td.relations.label(td.relation[tok_idx]))
                    } else {
                        ("_".to_owned(), "_")
                    };
                    let deprel = sanitize_conllu_field_or_underscore(deprel_raw);

                    let word_misc = if group_end - w_idx == 1 {
                        surface_misc.clone()
                    } else if let Some(sub_span) = words.subspan.as_ref().and_then(|s| s[cur]) {
                        let sub_src = doc.to_source(tok_view, sub_span);
                        let (ws, we) = offsets.convert_source_span(doc, sub_src)?;
                        build_misc(None, offsets, ws, we)
                    } else {
                        "_".to_owned()
                    };

                    let _ =
                        writeln!(buf, "{id_1}\t{form}\t{lemma}\t{upos}\t_\t{feat}\t{head_s}\t{deprel}\t_\t{word_misc}");
                }

                w_idx = group_end;
            }
            buf.push('\n');
        }

        out.write_all(buf.as_bytes())?;
        Ok(())
    }
}

fn sanitize_conllu_field(s: &str) -> String {
    let clean = sanitize_single_line(s);
    if clean.is_empty() { "_".to_owned() } else { clean }
}

fn sanitize_conllu_field_or_underscore(s: &str) -> String {
    let clean = sanitize_single_line(s);
    let trimmed = clean.trim();
    if trimmed.is_empty() { "_".to_owned() } else { trimmed.to_owned() }
}

fn build_misc(space_part: Option<&str>, offsets: ExportOffsetMode, start: u32, end: u32) -> String {
    match (space_part, offsets) {
        (None, ExportOffsetMode::Utf8) => "_".to_owned(),
        (Some(sp), ExportOffsetMode::Utf8) => sp.to_owned(),
        (None, ExportOffsetMode::Utf16) => format!("TokenRangeUtf16={start}:{end}"),
        (Some(sp), ExportOffsetMode::Utf16) => format!("{sp}|TokenRangeUtf16={start}:{end}"),
        (None, ExportOffsetMode::CodePoint) => format!("TokenRangeCp={start}:{end}"),
        (Some(sp), ExportOffsetMode::CodePoint) => format!("{sp}|TokenRangeCp={start}:{end}"),
    }
}

fn escape_conllu_spaces(gap: &str) -> String {
    let mut out = String::with_capacity(gap.len() * 2);
    for c in gap.chars() {
        match c {
            ' ' => out.push_str("\\s"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '|' => out.push_str("\\p"),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out
}
