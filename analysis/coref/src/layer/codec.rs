//! The document graph codecs for the coreference layers: [`Mentions`],
//! [`Speakers`], and the chains and antecedent links over mentions
//! (`Groups<Mentions>` and `Edges<Mentions, Mentions>`, whose generic
//! codecs live in `turbo-analysis`).
//!
//! Every index column names the instance it indexes and is checked on
//! import against that instance as restored: a sentence, token, head or
//! entity index past its layer, a token range that runs backwards, or an
//! enum byte outside its enum is refused before the layer is built, and
//! `Document::add` then re-proves [`Mentions`]' own invariants.

use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, NodeRef, Reader, Writer, get_spans, get_tagset,
    put_spans, put_tagset,
};
use turbo_analysis::{Document, Edges, Groups, Ix, IxRange, Key, Layer, Tag};
use turbo_analysis_layers::{Entities, Sentences, Tokens};

use super::{Agreement, MentionKind, Mentions, Origin, Speakers};

/// Registers the codecs for [`Mentions`], [`Speakers`],
/// `Groups<Mentions>` and `Edges<Mentions, Mentions>` on `codecs`.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs
        .register::<Mentions>()
        .register::<Speakers>()
        .register::<Groups<Mentions>>()
        .register::<Edges<Mentions, Mentions>>();
}

/// The shared layer codecs with the coreference codecs added.
pub fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    register_codecs(&mut codecs);
    codecs
}

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn put_ref(w: &mut Writer, layer: &LayerRef) -> Result<(), ExportError> {
    w.str(&layer.layer)?;
    w.str(&layer.instance)
}

fn get_ref(r: &mut Reader<'_>) -> Result<LayerRef, ExportError> {
    Ok(LayerRef { layer: r.string()?, instance: r.string()? })
}

/// Reads one index into `target` and requires it inside the restored instance.
fn get_ix<T>(r: &mut Reader<'_>, cx: &ImportCx, target: &LayerRef) -> Result<Ix<T>, ExportError> {
    let index = r.u32()?;
    if index == u32::MAX {
        return Err(malformed("an index uses the reserved u32::MAX value"));
    }
    cx.require_node(&NodeRef { layer: target.clone(), index })?;
    Ok(Ix::new(index))
}

fn want_len(got: usize, want: usize, what: &str) -> Result<(), ExportError> {
    if got == want { Ok(()) } else { Err(malformed(format!("{got} {what} for {want} mentions"))) }
}

fn put_bytes(w: &mut Writer, bytes: impl ExactSizeIterator<Item = u8>) -> Result<(), ExportError> {
    w.usize(bytes.len())?;
    for b in bytes {
        w.u8(b);
    }
    Ok(())
}

fn get_bytes<T>(r: &mut Reader<'_>, what: &str, decode: impl Fn(u8) -> Option<T>) -> Result<Vec<T>, ExportError> {
    let n = r.usize()?;
    let mut out = Vec::new();
    for _ in 0..n {
        let b = r.u8()?;
        out.push(decode(b).ok_or_else(|| malformed(format!("{what} byte {b}")))?);
    }
    Ok(out)
}

impl LayerCodec for Mentions {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ref(&mut w, &cx.resolve(&Key::<Sentences>::DEFAULT))?;
        w.usize(self.sentence.len())?;
        for s in &self.sentence {
            w.u32(s.get() as u32);
        }
        put_ref(&mut w, &cx.resolve(&Key::<Tokens>::DEFAULT))?;
        w.usize(self.tokens.len())?;
        for (range, head) in self.tokens.iter().zip(&self.head) {
            w.u32(range.start.get() as u32);
            w.u32(range.end.get() as u32);
            w.u32(head.get() as u32);
        }
        match &self.entity {
            None => w.u8(0),
            Some(entity) => {
                w.u8(1);
                put_ref(&mut w, &cx.resolve(&Key::<Entities>::DEFAULT))?;
                w.usize(entity.len())?;
                for e in entity {
                    match e {
                        None => w.u8(0),
                        Some(e) => {
                            w.u8(1);
                            w.u32(e.get() as u32);
                        }
                    }
                }
            }
        }
        put_bytes(&mut w, self.kind.iter().map(|k| k.byte()))?;
        put_bytes(&mut w, self.origin.iter().map(|o| o.byte()))?;
        put_bytes(&mut w, self.agreement.iter().map(|a| a.bits()))?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = spans.len();
        let sentences = get_ref(&mut r)?;
        cx.expect_target::<Sentences>(&sentences)?;
        let mut sentence = Vec::new();
        for _ in 0..r.usize()? {
            sentence.push(get_ix(&mut r, cx, &sentences)?);
        }
        want_len(sentence.len(), n, "sentences")?;
        let tokens_ref = get_ref(&mut r)?;
        cx.expect_target::<Tokens>(&tokens_ref)?;
        let token_count = cx.layer_len(&tokens_ref)?;
        let (mut tokens, mut head) = (Vec::new(), Vec::new());
        for _ in 0..r.usize()? {
            let (start, end) = (r.u32()?, r.u32()?);
            if start > end || end as usize > token_count || end == u32::MAX {
                return Err(malformed(format!("a token range {start}..{end} outside {token_count} tokens")));
            }
            tokens.push(IxRange::new(start, end));
            head.push(get_ix(&mut r, cx, &tokens_ref)?);
        }
        want_len(tokens.len(), n, "token ranges")?;
        let entity = match r.u8()? {
            0 => None,
            1 => {
                let entities = get_ref(&mut r)?;
                cx.expect_target::<Entities>(&entities)?;
                let mut col = Vec::new();
                for _ in 0..r.usize()? {
                    col.push(match r.u8()? {
                        0 => None,
                        1 => Some(get_ix(&mut r, cx, &entities)?),
                        tag => return Err(malformed(format!("an entity slot tag of {tag}"))),
                    });
                }
                want_len(col.len(), n, "entity slots")?;
                Some(col)
            }
            tag => return Err(malformed(format!("an optional column tag of {tag}"))),
        };
        let kind = get_bytes(&mut r, "a mention kind", MentionKind::from_byte)?;
        let origin = get_bytes(&mut r, "a mention origin", Origin::from_byte)?;
        let agreement = get_bytes(&mut r, "an agreement", Agreement::from_bits)?;
        want_len(kind.len(), n, "kinds")?;
        want_len(origin.len(), n, "origins")?;
        want_len(agreement.len(), n, "agreements")?;
        if !r.is_done() {
            return Err(malformed("trailing bytes in a mentions payload"));
        }
        let layer = Mentions { spans, sentence, tokens, head, entity, kind, origin, agreement };
        layer.check().map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
        Ok(layer)
    }
}

impl LayerCodec for Speakers {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_tagset(&mut w, &self.speakers)?;
        w.usize(self.speaker.len())?;
        for t in &self.speaker {
            w.u16(t.0);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let speakers = get_tagset(&mut r)?;
        let mut speaker = Vec::new();
        for _ in 0..r.usize()? {
            let t = r.u16()?;
            if t as usize >= speakers.len() {
                return Err(malformed(format!("speaker {t} outside {}", speakers.name)));
            }
            speaker.push(Tag(t));
        }
        if speaker.len() != spans.len() {
            return Err(malformed(format!("{} speakers for {} turns", speaker.len(), spans.len())));
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a speakers payload"));
        }
        Ok(Speakers { spans, speaker, speakers })
    }
}
