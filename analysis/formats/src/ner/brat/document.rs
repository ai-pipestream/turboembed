//! Brat documents and the directories they are read from.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use turbo_jvm::hash_map::put_order;
use turbo_unicode::WhitespaceMode;

use super::annotation::{BratAnnotation, Parsed, parse_annotations};
use crate::error::FormatError;
use crate::masc::java::decode_utf8;
use crate::source::check_len;

/// A brat document (`BratDocument`): a text and its annotations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BratDocument {
    id: String,
    text: String,
    annotations: Vec<BratAnnotation>,
}

impl BratDocument {
    /// The document `id` (the path of its files without `.txt` and
    /// `.ann`) of the text file `txt` and the annotation file `ann`, as
    /// `BratDocument.parseDocument` reads them: both decoded as UTF-8 by
    /// the JDK's decoder, which replaces malformed bytes, and the
    /// annotations kept by id. A second annotation with an id keeps the
    /// first one's place in the map and replaces it; an annotator note is
    /// attached to the annotation it names, the last note of an
    /// annotation replacing the others, and dropped when there is none.
    ///
    /// # Errors
    ///
    /// What the annotation stream throws on a line of `ann`
    /// ([`super::super::NerError::BratAnnotation`],
    /// [`super::super::NerError::BratIndex`],
    /// [`super::super::NerError::BratSpan`]), and [`FormatError::TooLong`]
    /// for a text longer than spans can index.
    pub fn parse(
        id: impl Into<String>,
        txt: &[u8],
        ann: &[u8],
        whitespace: WhitespaceMode,
    ) -> Result<BratDocument, FormatError> {
        let id = id.into();
        let text = decode_utf8(txt);
        check_len(&text)?;
        let parsed = parse_annotations(&decode_utf8(ann), &id, whitespace)?;
        // `annotationMap`, a `HashMap` by id: the order it iterates in
        let mut by_id: HashMap<String, usize> = HashMap::new();
        let mut annotations: Vec<BratAnnotation> = Vec::new();
        let mut notes = Vec::new();
        for p in parsed {
            match p {
                Parsed::Annotation(a) => match by_id.get(&a.id) {
                    Some(&i) => annotations[i] = a,
                    None => {
                        by_id.insert(a.id.clone(), annotations.len());
                        annotations.push(a);
                    }
                },
                Parsed::Note(n) => notes.push(n),
            }
        }
        for n in notes {
            if let Some(&i) = by_id.get(&n.attached_to) {
                annotations[i].note = n.note;
            }
        }
        let (order, _) = put_order(annotations.iter().map(|a| a.id.as_str()));
        let mut slots: Vec<Option<BratAnnotation>> = annotations.into_iter().map(Some).collect();
        let annotations = order.into_iter().map(|i| slots[i].take().expect("each key once")).collect();
        Ok(BratDocument { id, text, annotations })
    }

    /// The document's id: the path of its files without the extension.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The document's text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The annotations, notes left out, in the order OpenNLP's map
    /// iterates them (`getAnnotations`).
    pub fn annotations(&self) -> &[BratAnnotation] {
        &self.annotations
    }

    /// The annotation with id `id` (`getAnnotation`).
    pub fn annotation(&self, id: &str) -> Option<&BratAnnotation> {
        self.annotations.iter().find(|a| a.id == id)
    }

    /// Converts the text-bound (`T`) annotations of this brat document into
    /// an [`turbo_analysis_layers::Entities`] layer over `tokens`, keeping
    /// multi-fragment annotations as discontinuous items in `Spans<Entities>`
    /// and preserving nested or overlapping entities.
    pub fn to_entities(
        &self,
        tokens: &turbo_analysis_layers::Tokens,
    ) -> Result<turbo_analysis_layers::Entities, String> {
        use std::fmt::Write as _;
        use std::sync::Arc;
        use turbo_analysis::{IxRange, Layer, Span, Spans};
        use turbo_analysis_layers::{Entities, TagSet};
        use turbo_unicode::utf16::utf16_to_byte;

        let text = &self.text;
        let view = tokens.spans.view();
        let (tok_starts, tok_ends) = tokens.spans.columns();

        struct RawEntity<'a> {
            kind: &'a str,
            frags: Vec<Span>,
        }

        let mut raw: Vec<RawEntity<'_>> = Vec::new();
        let mut labels: Vec<&str> = Vec::new();
        for a in &self.annotations {
            let super::annotation::AnnotationValue::Span { fragments, .. } = &a.value else {
                continue;
            };
            if fragments.is_empty() {
                continue;
            }
            if !labels.contains(&a.kind.as_str()) {
                labels.push(a.kind.as_str());
            }
            let mut frags: Vec<Span> = Vec::with_capacity(fragments.len());
            for f in fragments {
                let start_u16 = usize::try_from(f.start)
                    .map_err(|_| format!("annotation {} has negative start {}", a.id, f.start))?;
                let end_u16 =
                    usize::try_from(f.end).map_err(|_| format!("annotation {} has negative end {}", a.id, f.end))?;
                let start_b = utf16_to_byte(text, start_u16).ok_or_else(|| {
                    let mut msg = String::new();
                    let _ = write!(msg, "annotation {} start {} is not on a character boundary", a.id, f.start);
                    msg
                })? as u32;
                let end_b = utf16_to_byte(text, end_u16).ok_or_else(|| {
                    let mut msg = String::new();
                    let _ = write!(msg, "annotation {} end {} is not on a character boundary", a.id, f.end);
                    msg
                })? as u32;
                frags.push(Span::new(start_b, end_b));
            }
            frags.sort_by(|x, y| x.start.cmp(&y.start).then(x.end.cmp(&y.end)));
            // Coalesce touching or overlapping fragments within the same annotation.
            let mut merged: Vec<Span> = Vec::with_capacity(frags.len());
            for f in frags {
                if let Some(last) = merged.last_mut()
                    && f.start <= last.end
                {
                    last.end = last.end.max(f.end);
                } else {
                    merged.push(f);
                }
            }
            raw.push(RawEntity { kind: a.kind.as_str(), frags: merged });
        }

        // Order entities by envelope start so `Entities::check` holds.
        raw.sort_by(|a, b| {
            a.frags[0].start.cmp(&b.frags[0].start).then(b.frags.last().unwrap().end.cmp(&a.frags.last().unwrap().end))
        });

        let kinds = TagSet::new("brat", &labels);
        let mut spans = Spans::with_capacity(view, raw.len());
        let mut tok_ranges = Vec::with_capacity(raw.len());
        let mut kind_col = Vec::with_capacity(raw.len());
        let mut score_col = Vec::with_capacity(raw.len());

        for item in raw {
            let first_start = item.frags[0].start;
            let last_end = item.frags.last().unwrap().end;
            let t_start = tok_ends.partition_point(|&e| e <= first_start) as u32;
            let t_end = tok_starts.partition_point(|&s| s < last_end).max(t_start as usize) as u32;
            spans.try_push_fragments(&item.frags).map_err(|e| e.to_string())?;
            tok_ranges.push(IxRange::new(t_start, t_end));
            kind_col.push(kinds.tag(item.kind).expect("label collected above"));
            score_col.push(1.0);
        }

        let entities =
            Entities { spans, tokens: tok_ranges, kind: kind_col, kinds: Arc::clone(&kinds), score: score_col };
        entities.check()?;
        Ok(entities)
    }
}

/// Formats an [`turbo_analysis_layers::Entities`] layer over `text` in brat
/// standoff `.ann` syntax (`T1\t<kind> <start> <end>[;<start> <end>]\t<covered>\n`),
/// converting UTF-8 byte spans to UTF-16 code-unit offsets and joining
/// discontinuous fragments' covered text with a single space.
pub fn write_brat_entities(text: &str, entities: &turbo_analysis_layers::Entities) -> String {
    use std::fmt::Write as _;
    use turbo_unicode::utf16::byte_to_utf16;

    let mut out = String::new();
    for (ix, _) in entities.spans.iter() {
        let i = ix.get();
        let label = entities.kinds.label(entities.kind[i]);
        let _ = write!(out, "T{}\t{label} ", i + 1);
        for (f_idx, frag) in entities.spans.fragments(ix).enumerate() {
            if f_idx > 0 {
                out.push(';');
            }
            let u_start = byte_to_utf16(text, frag.start as usize).expect("fragment start on char boundary");
            let u_end = byte_to_utf16(text, frag.end as usize).expect("fragment end on char boundary");
            let _ = write!(out, "{u_start} {u_end}");
        }
        out.push('\t');
        for (f_idx, frag) in entities.spans.fragments(ix).enumerate() {
            if f_idx > 0 {
                out.push(' ');
            }
            out.push_str(&text[frag.range()]);
        }
        out.push('\n');
    }
    out
}

/// The ids of the documents of the brat corpus directory `dir`, in the
/// order `BratDocumentStream` reads them: each file whose absolute path
/// ends with `.ann` and that has a `.txt` file beside it, the files of a
/// directory in Java's string order of their names. With `recursive`,
/// the subdirectories of a directory are read after its files, the last
/// subdirectory by name first (the stream keeps them on a stack).
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`] when `dir` is not a directory, with
/// the message of the stream's `IOException`, and what reading a
/// directory gives.
pub fn brat_documents(dir: &Path, recursive: bool) -> io::Result<Vec<PathBuf>> {
    let dir = std::path::absolute(dir)?;
    if !dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Input corpus directory must be a directory according to File.isDirectory()!",
        ));
    }
    let mut ids = Vec::new();
    let mut stack = vec![dir];
    while let Some(dir) = stack.pop() {
        let mut entries: Vec<(String, PathBuf)> = std::fs::read_dir(&dir)?
            .map(|e| e.map(|e| (e.file_name().to_string_lossy().into_owned(), e.path())))
            .collect::<io::Result<_>>()?;
        entries.sort_by(|a, b| turbo_jvm::chars::compare(&a.0, &b.0));
        for (name, path) in entries {
            if path.is_file() {
                if let Some(base) = name.strip_suffix(".ann") {
                    let id = dir.join(base);
                    let txt = dir.join(format!("{base}.txt"));
                    if txt.is_file() {
                        ids.push(id);
                    }
                }
            } else if recursive && path.is_dir() {
                stack.push(path);
            }
        }
    }
    Ok(ids)
}

/// The documents of the brat corpus directory `dir` ([`brat_documents`]),
/// each read from its two files and parsed, or what parsing it gave.
///
/// # Errors
///
/// What listing the directory or reading a file gives.
pub fn read_brat_corpus(
    dir: &Path,
    recursive: bool,
    whitespace: WhitespaceMode,
) -> io::Result<Vec<Result<BratDocument, FormatError>>> {
    let mut out = Vec::new();
    for id in brat_documents(dir, recursive)? {
        let name = id.to_string_lossy().into_owned();
        let txt = std::fs::read(format!("{name}.txt"))?;
        let ann = std::fs::read(format!("{name}.ann"))?;
        out.push(BratDocument::parse(name, &txt, &ann, whitespace));
    }
    Ok(out)
}
