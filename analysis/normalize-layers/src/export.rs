//! The document graph codecs for this crate's layers, so what the
//! normalizers found crosses the export boundary with the rest of
//! the graph.

use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, Reader, Writer, get_spans, put_spans,
};
use turbo_analysis::{Document, Layer, Span, Spans};
use turbo_analysis_layers::Strs;

use crate::{
    ArtifactKind, Artifacts, AssetEncoding, AssetFormat, Assets, Corrections, Emoji, EmojiKind, Noise, NoiseSeverity,
    PlaceholderKind, Placeholders, ProtectedSpans,
};

/// The kinds in wire order: a variant crosses as its position in its
/// table, so a position outside the table is refused rather than
/// read as some other kind.
const PLACEHOLDER_KINDS: [PlaceholderKind; 9] = [
    PlaceholderKind::Url,
    PlaceholderKind::Email,
    PlaceholderKind::Hashtag,
    PlaceholderKind::Mention,
    PlaceholderKind::RetweetMarker,
    PlaceholderKind::Emoticon,
    PlaceholderKind::Number,
    PlaceholderKind::Emoji,
    PlaceholderKind::Asset,
];

const EMOJI_KINDS: [EmojiKind; 6] =
    [EmojiKind::Basic, EmojiKind::Keycap, EmojiKind::Flag, EmojiKind::Modifier, EmojiKind::Tag, EmojiKind::Zwj];

const NOISE_SEVERITIES: [NoiseSeverity; 4] =
    [NoiseSeverity::Misspelled, NoiseSeverity::Damaged, NoiseSeverity::Gibberish, NoiseSeverity::Binaryish];

const ASSET_ENCODINGS: [AssetEncoding; 3] = [AssetEncoding::Base64, AssetEncoding::Base64Url, AssetEncoding::Hex];

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn done(r: &Reader<'_>, what: &str) -> Result<(), ExportError> {
    if r.is_done() { Ok(()) } else { Err(malformed(format!("trailing bytes in a {what} payload"))) }
}

fn want_len(got: usize, want: usize, what: &str) -> Result<(), ExportError> {
    if got != want { Err(malformed(format!("{got} {what} for {want} items"))) } else { Ok(()) }
}

fn put_variant<E: Copy + PartialEq>(w: &mut Writer, table: &[E], value: E, what: &str) -> Result<(), ExportError> {
    let at = table.iter().position(|v| *v == value).ok_or_else(|| malformed(format!("a {what} outside its table")))?;
    w.u8(u8::try_from(at).map_err(|_| malformed(format!("a {what} table longer than a u8 can index")))?);
    Ok(())
}

fn get_variant<E: Copy>(r: &mut Reader<'_>, table: &[E], what: &str) -> Result<E, ExportError> {
    let at = r.u8()? as usize;
    table.get(at).copied().ok_or_else(|| malformed(format!("a {what} discriminant of {at}")))
}

fn put_variants<E: Copy + PartialEq>(w: &mut Writer, table: &[E], values: &[E], what: &str) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for &v in values {
        put_variant(w, table, v, what)?;
    }
    Ok(())
}

fn get_variants<E: Copy>(r: &mut Reader<'_>, table: &[E], what: &str) -> Result<Vec<E>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(get_variant(r, table, what)?);
    }
    Ok(values)
}

fn put_strs(w: &mut Writer, strs: &Strs) -> Result<(), ExportError> {
    w.usize(strs.len())?;
    for s in strs.iter() {
        w.str(s)?;
    }
    Ok(())
}

fn get_strs(r: &mut Reader<'_>) -> Result<Strs, ExportError> {
    let mut strs = Strs::default();
    for _ in 0..r.usize()? {
        strs.push(&r.string()?);
    }
    Ok(strs)
}

/// Every fragment inside the text of the view the spans are in, on
/// UTF-8 boundaries, in the document built so far.
fn check_spans<L>(spans: &Spans<L>, doc: &Document, cx: &ImportCx) -> Result<(), ExportError> {
    let text = doc.text(spans.view());
    let invalid = |detail: String| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail };
    for (i, _) in spans.iter() {
        for frag in spans.fragments(i) {
            check_span(frag, text).map_err(invalid)?;
        }
    }
    Ok(())
}

fn check_span(span: Span, text: &str) -> Result<(), String> {
    let (s, e) = (span.start as usize, span.end as usize);
    if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
        return Err(format!(
            "span {}..{} is not inside view text of length {} on UTF-8 boundaries",
            span.start,
            span.end,
            text.len()
        ));
    }
    Ok(())
}

fn checked<L: Layer>(layer: L, cx: &ImportCx) -> Result<L, ExportError> {
    layer.check().map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
    Ok(layer)
}

impl LayerCodec for Placeholders {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_variants(&mut w, &PLACEHOLDER_KINDS, &self.kind, "placeholder kind")?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let kind = get_variants(&mut r, &PLACEHOLDER_KINDS, "placeholder kind")?;
        want_len(kind.len(), spans.len(), "kinds")?;
        done(&r, "placeholders")?;
        check_spans(&spans, doc, cx)?;
        checked(Placeholders { spans, kind }, cx)
    }
}

impl LayerCodec for Emoji {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_variants(&mut w, &EMOJI_KINDS, &self.kind, "emoji kind")?;
        put_strs(&mut w, &self.name)?;
        put_strs(&mut w, &self.region)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let kind = get_variants(&mut r, &EMOJI_KINDS, "emoji kind")?;
        let name = get_strs(&mut r)?;
        let region = get_strs(&mut r)?;
        let n = spans.len();
        want_len(kind.len(), n, "kinds")?;
        want_len(name.len(), n, "names")?;
        want_len(region.len(), n, "regions")?;
        done(&r, "emoji")?;
        check_spans(&spans, doc, cx)?;
        checked(Emoji { spans, kind, name, region }, cx)
    }
}

impl LayerCodec for Artifacts {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_variants(&mut w, &ArtifactKind::ALL, &self.kind, "artifact kind")?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let kind = get_variants(&mut r, &ArtifactKind::ALL, "artifact kind")?;
        want_len(kind.len(), spans.len(), "kinds")?;
        done(&r, "artifacts")?;
        check_spans(&spans, doc, cx)?;
        checked(Artifacts { spans, kind }, cx)
    }
}

impl LayerCodec for Noise {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_variants(&mut w, &NOISE_SEVERITIES, &self.severity, "noise severity")?;
        w.usize(self.score.len())?;
        for &score in &self.score {
            w.f64(score);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let severity = get_variants(&mut r, &NOISE_SEVERITIES, "noise severity")?;
        let mut score = Vec::new();
        for _ in 0..r.usize()? {
            score.push(r.f64()?);
        }
        let n = spans.len();
        want_len(severity.len(), n, "severities")?;
        want_len(score.len(), n, "scores")?;
        done(&r, "noise")?;
        check_spans(&spans, doc, cx)?;
        // The layer's own check holds every score to (0, 1], which
        // also refuses a NaN the wire could spell.
        checked(Noise { spans, severity, score }, cx)
    }
}

impl LayerCodec for Corrections {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_strs(&mut w, &self.suggestion)?;
        w.usize(self.edit_distance.len())?;
        for &d in &self.edit_distance {
            w.u8(d);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let suggestion = get_strs(&mut r)?;
        let mut edit_distance = Vec::new();
        for _ in 0..r.usize()? {
            edit_distance.push(r.u8()?);
        }
        let n = spans.len();
        want_len(suggestion.len(), n, "suggestions")?;
        want_len(edit_distance.len(), n, "edit distances")?;
        done(&r, "corrections")?;
        check_spans(&spans, doc, cx)?;
        checked(Corrections { spans, suggestion, edit_distance }, cx)
    }
}

fn put_format(w: &mut Writer, format: AssetFormat) -> Result<(), ExportError> {
    // A sniffed format crosses as the name and media type pair that
    // names it, which `AssetFormat::from_names` inverts exactly; a
    // declared one has neither and crosses as its own tag.
    match format {
        AssetFormat::Declared => w.u8(0),
        other => {
            w.u8(1);
            w.str(other.name())?;
            w.str(other.media_type())?;
        }
    }
    Ok(())
}

fn get_format(r: &mut Reader<'_>) -> Result<AssetFormat, ExportError> {
    Ok(match r.u8()? {
        0 => AssetFormat::Declared,
        1 => {
            let (name, media_type) = (r.string()?, r.string()?);
            AssetFormat::from_names(&name, &media_type)
                .ok_or_else(|| malformed(format!("an asset format named {name:?} of media type {media_type:?}")))?
        }
        tag => return Err(malformed(format!("an asset format tag of {tag}"))),
    })
}

fn put_opt_u32s(w: &mut Writer, values: &[Option<u32>]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        match v {
            None => w.u8(0),
            Some(v) => {
                w.u8(1);
                w.u32(*v);
            }
        }
    }
    Ok(())
}

fn get_opt_u32s(r: &mut Reader<'_>) -> Result<Vec<Option<u32>>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(match r.u8()? {
            0 => None,
            1 => Some(r.u32()?),
            tag => return Err(malformed(format!("an optional size tag of {tag}"))),
        });
    }
    Ok(values)
}

impl LayerCodec for Assets {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.payload.len())?;
        for &payload in &self.payload {
            w.span(payload);
        }
        w.usize(self.format.len())?;
        for &format in &self.format {
            put_format(&mut w, format)?;
        }
        put_strs(&mut w, &self.declared_type)?;
        put_variants(&mut w, &ASSET_ENCODINGS, &self.encoding, "asset encoding")?;
        w.usize(self.decoded_len.len())?;
        for &len in &self.decoded_len {
            w.u64(len);
        }
        put_opt_u32s(&mut w, &self.width)?;
        put_opt_u32s(&mut w, &self.height)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut payload = Vec::new();
        for _ in 0..r.usize()? {
            payload.push(r.span()?);
        }
        let mut format = Vec::new();
        for _ in 0..r.usize()? {
            format.push(get_format(&mut r)?);
        }
        let declared_type = get_strs(&mut r)?;
        let encoding = get_variants(&mut r, &ASSET_ENCODINGS, "asset encoding")?;
        let mut decoded_len = Vec::new();
        for _ in 0..r.usize()? {
            decoded_len.push(r.u64()?);
        }
        let width = get_opt_u32s(&mut r)?;
        let height = get_opt_u32s(&mut r)?;
        let n = spans.len();
        want_len(payload.len(), n, "payloads")?;
        want_len(format.len(), n, "formats")?;
        want_len(declared_type.len(), n, "declared types")?;
        want_len(encoding.len(), n, "encodings")?;
        want_len(decoded_len.len(), n, "decoded lengths")?;
        want_len(width.len(), n, "widths")?;
        want_len(height.len(), n, "heights")?;
        done(&r, "assets")?;
        check_spans(&spans, doc, cx)?;
        let text = doc.text(spans.view());
        for &p in &payload {
            check_span(p, text).map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
        }
        // The layer's own check holds each payload inside its span,
        // the declared media types to a type and subtype, and the
        // dimensions to both-or-neither and positive.
        checked(Assets { spans, payload, format, declared_type, encoding, decoded_len, width, height }, cx)
    }
}

impl LayerCodec for ProtectedSpans {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        done(&r, "protected spans")?;
        check_spans(&spans, doc, cx)?;
        checked(ProtectedSpans { spans }, cx)
    }
}

/// Registers this crate's layer codecs on `codecs`.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs
        .register::<Placeholders>()
        .register::<Emoji>()
        .register::<Artifacts>()
        .register::<Noise>()
        .register::<Corrections>()
        .register::<Assets>()
        .register::<ProtectedSpans>();
}

/// A registry holding this crate's codecs and nothing else.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
