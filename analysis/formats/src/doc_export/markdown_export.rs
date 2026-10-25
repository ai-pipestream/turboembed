//! Structured Markdown report [`DocumentFormatter`].

use std::fmt::Write as _;
use std::io::Write;

use turbo_analysis::{Document, Ix, Key, Span};
use turbo_analysis_layers::{
    Aspects, Categories, Chunks, EmotionDimension, EmotionLayer, Entities, Language, Opinions, Pii, PolarityScores,
    Sentences, Tokens,
};

use super::{
    DocumentFormatter, ExportOffsetMode, FormatExportError, OutputFormatDescriptor, get_bound_layer,
    sanitize_single_line,
};

const REQUIRED_LAYERS: &[&str] = &[];

/// Structured Markdown report (`markdown`) exporter.
///
/// Emits `# Document <doc_id>`, detected language, document-level categories,
/// per-sentence blockquotes, entity bullets with source offsets, sentiment and
/// emotion summaries, PII summary, and chunks, escaping Markdown control
/// characters cleanly.
#[derive(Copy, Clone, Debug, Default)]
pub struct MarkdownFormatter;

impl DocumentFormatter for MarkdownFormatter {
    fn descriptor(&self) -> OutputFormatDescriptor {
        OutputFormatDescriptor {
            format_id: "markdown",
            display_name: "Markdown Analysis Report",
            media_type: "text/markdown; charset=utf-8",
            file_extension: "md",
            binary: false,
            lossless: false,
            required_layers: REQUIRED_LAYERS,
            description: "Structured Markdown report with detected language, categories, sentence blockquotes, entity bullets, and sentiment/emotion summaries.",
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
        let title_id = if doc_id.trim().is_empty() { "untitled" } else { doc_id.trim() };
        let _ = writeln!(buf, "# Document {}\n", escape_markdown(&sanitize_single_line(title_id)));
        let _ = writeln!(buf, "- **Offset Mode**: `{}`", offsets.as_str());
        let _ = writeln!(buf, "- **Source Byte Length**: `{}`", source_text.len());

        if let Some(lang) = doc.get::<Language>()
            && let Some((code, score)) = lang.best()
        {
            let _ = writeln!(buf, "- **Detected Language**: `{}` ({score:.4})", escape_markdown(code));
        }

        if let Some(cats) = doc.get::<Categories>()
            && let Some(row) = cats.get(0)
        {
            if !row.best.is_empty() {
                let _ = writeln!(buf, "- **Best Category**: `{}` ({:.4})", escape_markdown(row.best), row.best_score());
            }
            let ranked = row.ranked();
            if !ranked.is_empty() {
                buf.push_str("\n## Categories\n\n");
                for (label, score) in ranked {
                    let _ = writeln!(buf, "- `{}`: {:.4}", escape_markdown(label), score);
                }
            }
        }

        buf.push_str("\n## Sentences\n\n");
        if let Some(sents) = doc.get::<Sentences>()
            && !sents.spans.is_empty()
        {
            let s_view = sents.spans.view();
            for (s_ix, s_span) in sents.spans.iter() {
                let src_span = doc.to_source(s_view, s_span);
                let (start, end) = offsets.convert_source_span(doc, src_span)?;
                let text = sanitize_single_line(&source_text[src_span.range()]);
                let _ = writeln!(buf, "### Sentence {} ({start}..{end})\n", s_ix.get() + 1);
                let _ = writeln!(buf, "> {}\n", escape_markdown(&text));
            }
        } else {
            let full_span = Span::new(0, source_text.len() as u32);
            let (start, end) = offsets.convert_source_span(doc, full_span)?;
            let text = sanitize_single_line(source_text);
            let _ = writeln!(buf, "### Sentence 1 ({start}..{end})\n");
            let _ = writeln!(buf, "> {}\n", escape_markdown(&text));
        }

        if let Some(entities) = doc.get::<Entities>()
            && !entities.spans.is_empty()
        {
            buf.push_str("## Entities\n\n");
            let e_view = entities.spans.view();
            for (e_ix, _env) in entities.spans.iter() {
                let row = e_ix.get();
                let label = entities.kinds.label(entities.kind[row]);
                let score = entities.score[row];
                let mut frag_strs = Vec::new();
                let mut text_parts = Vec::new();
                for frag in entities.spans.fragments(e_ix) {
                    let src_frag = doc.to_source(e_view, frag);
                    let (s, e) = offsets.convert_source_span(doc, src_frag)?;
                    frag_strs.push(format!("{s}..{e}"));
                    text_parts.push(sanitize_single_line(&source_text[src_frag.range()]));
                }
                let joined_text = text_parts.join(" ");
                let joined_frags = frag_strs.join(";");
                let _ = writeln!(
                    buf,
                    "- **{}** [{}]: {} (score: {score:.4})",
                    escape_markdown(label),
                    joined_frags,
                    escape_markdown(&joined_text),
                );
            }
            buf.push('\n');
        }

        let opinions = get_bound_layer::<Opinions>(doc, Key::<Tokens>::DEFAULT.id(), None)?;
        let aspects = get_bound_layer::<Aspects>(doc, Key::<Tokens>::DEFAULT.id(), None)?;
        let emotions = doc.get::<EmotionLayer>();
        let has_sentiment = opinions.is_some_and(|o| !o.tokens.is_empty())
            || aspects.is_some_and(|a| !a.tokens.is_empty())
            || emotions.is_some_and(|e| !e.target.is_empty());

        if has_sentiment {
            buf.push_str("## Sentiment & Emotion Summary\n\n");
            if let Some(ops) = opinions {
                let tokens = doc.get::<Tokens>();
                for (_ix, op) in ops.iter() {
                    let surface = if let Some(toks) = tokens
                        && !op.tokens.is_empty()
                        && op.tokens.end.get() <= toks.spans.len()
                    {
                        let first = toks.spans.get(op.tokens.start);
                        let last = toks.spans.get(Ix::new(op.tokens.end.get() as u32 - 1));
                        let src = doc.to_source(toks.spans.view(), Span::new(first.start, last.end));
                        sanitize_single_line(&source_text[src.range()])
                    } else {
                        format!("tokens {}..{}", op.tokens.start.get(), op.tokens.end.get())
                    };
                    let _ = writeln!(
                        buf,
                        "- **Opinion**: {} (polarity: {:.4}, intensity: {:.4}, negated: `{}`)",
                        escape_markdown(&surface),
                        op.polarity,
                        op.intensity,
                        op.negated
                    );
                }
            }
            if let Some(asp) = aspects {
                for i in 0..asp.tokens.len() {
                    let pol = PolarityScores::new(asp.positive[i], asp.negative[i], asp.neutral[i]);
                    let cat = asp.category[i].map_or("general", |t| asp.categories.label(t));
                    let _ = writeln!(
                        buf,
                        "- **Aspect** (`{}`): {} (pos: {:.4}, neg: {:.4}, neu: {:.4})",
                        escape_markdown(cat),
                        pol.label().name(),
                        pol.positive,
                        pol.negative,
                        pol.neutral,
                    );
                }
            }
            if let Some(em) = emotions {
                for i in 0..em.target.len() {
                    let ix = Ix::new(i as u32);
                    let scores = em.scores_of(ix);
                    let mut best_dim = EmotionDimension::Neutral;
                    let mut best_val = -1.0_f64;
                    for dim in EmotionDimension::ALL {
                        let v = scores.0[dim.index()];
                        if v > best_val {
                            best_val = v;
                            best_dim = dim;
                        }
                    }
                    let _ = writeln!(buf, "- **Emotion Target {}**: `{}` ({best_val:.4})", i + 1, best_dim.name(),);
                }
            }
            buf.push('\n');
        }

        if let Some(pii) = doc.get::<Pii>()
            && !pii.spans.is_empty()
        {
            buf.push_str("## PII Summary\n\n");
            let p_view = pii.spans.view();
            for m in pii.mentions() {
                let src_span = doc.to_source(p_view, m.span);
                let (s, e) = offsets.convert_source_span(doc, src_span)?;
                let _ = writeln!(
                    buf,
                    "- **{}** [{s}..{e}]: `{}` (confidence: `{}`)",
                    escape_markdown(m.kind.name()),
                    escape_markdown(&sanitize_single_line(m.normalized)),
                    m.confidence.name(),
                );
            }
            buf.push('\n');
        }

        if let Some(chunks) = doc.get::<Chunks>()
            && !chunks.spans.is_empty()
        {
            buf.push_str("## Chunks\n\n");
            let c_view = chunks.spans.view();
            for (c_ix, c_span) in chunks.spans.iter() {
                let src_span = doc.to_source(c_view, c_span);
                let (s, e) = offsets.convert_source_span(doc, src_span)?;
                let _ = writeln!(buf, "- **Chunk {}** [{s}..{e}]", c_ix.get() + 1);
            }
        }

        out.write_all(buf.as_bytes())?;
        Ok(())
    }
}

/// Escapes Markdown syntax characters (`\`, `` ` ``, `*`, `_`, `[`, `]`, `<`,
/// `>`, `#`, `|`, `~`) so arbitrary document text cannot inject unintended
/// formatting or HTML elements.
pub fn escape_markdown(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if matches!(ch, '\\' | '`' | '*' | '_' | '[' | ']' | '<' | '>' | '#' | '|' | '~') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}
