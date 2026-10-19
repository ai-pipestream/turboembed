//! The document graph codec for this crate's own layer, so resolved
//! locations cross the export boundary with the rest of the graph.
//! This crate's other writes (`Containment`, `Regions`) are shared
//! layer types whose codecs register with the shared layers.

use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, Reader, Writer, get_spans, put_spans,
};
use turbo_analysis::{Document, Layer};

use crate::entry::{AttributeValue, EntryParts, GazetteerEntry, GeoBoundingBox, GeoPoint};
use crate::geocode::Locations;

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn invalid(cx: &ImportCx, detail: String) -> ExportError {
    ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail }
}

fn put_strings(w: &mut Writer, values: &[String]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.str(v)?;
    }
    Ok(())
}

fn get_strings(r: &mut Reader<'_>) -> Result<Vec<String>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.string()?);
    }
    Ok(values)
}

fn put_opt_str(w: &mut Writer, value: Option<&str>) -> Result<(), ExportError> {
    match value {
        None => w.u8(0),
        Some(v) => {
            w.u8(1);
            w.str(v)?;
        }
    }
    Ok(())
}

fn get_opt_str(r: &mut Reader<'_>) -> Result<Option<String>, ExportError> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(r.string()?),
        tag => return Err(malformed(format!("an optional text tag of {tag}"))),
    })
}

fn put_entry(w: &mut Writer, entry: &GazetteerEntry) -> Result<(), ExportError> {
    let parts = entry.parts();
    w.str(&parts.source)?;
    w.str(&parts.record_id)?;
    w.str(&parts.name)?;
    put_strings(w, &parts.alternate_names)?;
    w.f64(parts.location.latitude());
    w.f64(parts.location.longitude());
    match parts.bounding_box {
        None => w.u8(0),
        Some(b) => {
            w.u8(1);
            w.f64(b.west());
            w.f64(b.south());
            w.f64(b.east());
            w.f64(b.north());
        }
    }
    put_opt_str(w, parts.country_code.as_deref())?;
    put_strings(w, &parts.containment)?;
    w.u64(parts.population as u64);
    put_opt_str(w, parts.feature_class.as_deref())?;
    w.usize(parts.attributes.len())?;
    for (key, value) in &parts.attributes {
        w.str(key)?;
        w.str(value.value())?;
        w.str(value.source())?;
        w.str(value.notes())?;
    }
    Ok(())
}

fn get_entry(r: &mut Reader<'_>, cx: &ImportCx) -> Result<GazetteerEntry, ExportError> {
    let source = r.string()?;
    let record_id = r.string()?;
    let name = r.string()?;
    let alternate_names = get_strings(r)?;
    // The checked constructors settle the ranges: a latitude or
    // longitude outside its interval, a NaN, or a box with its south
    // above its north never becomes a value.
    let location = GeoPoint::new(r.f64()?, r.f64()?).map_err(|e| invalid(cx, e.to_string()))?;
    let bounding_box = match r.u8()? {
        0 => None,
        1 => Some(GeoBoundingBox::new(r.f64()?, r.f64()?, r.f64()?, r.f64()?).map_err(|e| invalid(cx, e.to_string()))?),
        tag => return Err(malformed(format!("an optional bounding box tag of {tag}"))),
    };
    let country_code = get_opt_str(r)?;
    let containment = get_strings(r)?;
    let population = r.u64()? as i64;
    let feature_class = get_opt_str(r)?;
    let mut attributes = Vec::new();
    for _ in 0..r.usize()? {
        let key = r.string()?;
        let value =
            AttributeValue::new(r.string()?, r.string()?, r.string()?).map_err(|e| invalid(cx, e.to_string()))?;
        attributes.push((key, value));
    }
    // And the entry constructor settles the rest: no blank source,
    // record id, name, alternate or containment element, a country
    // code of two ASCII capitals, a non-negative population, and no
    // repeated attribute key.
    GazetteerEntry::new(EntryParts {
        source,
        record_id,
        name,
        alternate_names,
        location,
        bounding_box,
        country_code,
        containment,
        population,
        feature_class,
        attributes,
    })
    .map_err(|e| invalid(cx, e.to_string()))
}

impl LayerCodec for Locations {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.entry.len())?;
        for entry in &self.entry {
            put_entry(&mut w, entry)?;
        }
        w.usize(self.confidence.len())?;
        for &confidence in &self.confidence {
            w.f64(confidence);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut entry = Vec::new();
        for _ in 0..r.usize()? {
            entry.push(get_entry(&mut r, cx)?);
        }
        let mut confidence = Vec::new();
        for _ in 0..r.usize()? {
            confidence.push(r.f64()?);
        }
        let n = spans.len();
        if entry.len() != n || confidence.len() != n {
            return Err(malformed(format!(
                "{} entries and {} confidences for {n} locations",
                entry.len(),
                confidence.len()
            )));
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a locations payload"));
        }

        let text = doc.text(spans.view());
        for (i, _) in spans.iter() {
            for frag in spans.fragments(i) {
                let (s, e) = (frag.start as usize, frag.end as usize);
                if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
                    return Err(invalid(
                        cx,
                        format!(
                            "span {}..{} is not inside view text of length {} on UTF-8 boundaries",
                            frag.start,
                            frag.end,
                            text.len()
                        ),
                    ));
                }
            }
        }

        // The layer's own check holds every confidence to [0, 1],
        // which also refuses a NaN the wire could spell, and every
        // span to non-empty.
        let layer = Locations { spans, entry, confidence };
        layer.check().map_err(|detail| invalid(cx, detail))?;
        Ok(layer)
    }
}

/// Registers this crate's layer codec on `codecs`.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<Locations>();
}

/// A registry holding this crate's codec and nothing else.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
