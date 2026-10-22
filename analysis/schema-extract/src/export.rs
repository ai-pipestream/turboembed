//! `DocumentGraph` export and import for the record layer. The payload is
//! read as untrusted: every count is a row count checked against what
//! follows, every tag against its set, every record and support row
//! against the records and the restored source layers, and the result
//! against [`Layer::check`].

use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, Reader, Writer, get_spans, get_tagset, put_spans,
    put_tagset,
};
use turbo_analysis::{Document, Layer, LayerName, Tag};
use turbo_analysis_layers::{Numbers, Pii};
use turbo_geo::Locations;
use turbo_glossary_layers::GlossaryTerms;

use crate::layer::{Records, source_of};
use crate::schema::Source;

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn tag(r: &mut Reader<'_>, len: usize, what: &str) -> Result<Tag, ExportError> {
    let t = r.u16()?;
    if t as usize >= len {
        return Err(malformed(format!("a {what} outside its set")));
    }
    Ok(Tag(t))
}

/// The rows of the restored default instance of `source`; 0 when the
/// graph has none.
fn source_len(source: Source, cx: &ImportCx) -> usize {
    let name: &LayerName = match source {
        Source::Glossary => GlossaryTerms::NAME,
        Source::Pii => Pii::NAME,
        Source::Numbers => Numbers::NAME,
        Source::Locations => Locations::NAME,
    };
    let name = name.to_string();
    let instance = cx.resolved_instance(&name).to_owned();
    cx.layer_len(&LayerRef { layer: name, instance }).unwrap_or(0)
}

impl LayerCodec for Records {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        for set in [&self.types, &self.paths, &self.choices] {
            put_tagset(&mut w, set)?;
        }
        w.usize(self.record_type.len())?;
        for (t, &i) in self.record_type.iter().zip(&self.instance) {
            w.u16(t.0);
            w.u32(i);
        }
        w.usize(self.record.len())?;
        for i in 0..self.record.len() {
            w.u32(self.record[i]);
            w.u16(self.path[i].0);
            w.f64(self.score[i]);
            w.f64(self.logit[i]);
            w.u32(self.support_end[i]);
        }
        w.usize(self.support_source.len())?;
        for (&s, &row) in self.support_source.iter().zip(&self.support_row) {
            w.u8(s);
            w.u32(row);
        }
        w.usize(self.choice.len())?;
        for i in 0..self.choice.len() {
            w.u32(self.choice_record[i]);
            w.u16(self.choice_path[i].0);
            w.u16(self.choice[i].0);
            w.f64(self.choice_score[i]);
            w.f64(self.choice_logit[i]);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let (types, paths, choices) = (get_tagset(&mut r)?, get_tagset(&mut r)?, get_tagset(&mut r)?);
        let records = r.usize()?;
        let (mut record_type, mut instance) = (Vec::new(), Vec::new());
        for _ in 0..records {
            record_type.push(tag(&mut r, types.len(), "record type")?);
            instance.push(r.u32()?);
        }
        let values = r.usize()?;
        if values != spans.len() {
            return Err(malformed(format!("{values} values for {} spans", spans.len())));
        }
        let (mut record, mut path, mut score, mut logit, mut support_end) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..values {
            record.push(r.u32()?);
            path.push(tag(&mut r, paths.len(), "field path")?);
            score.push(r.f64()?);
            logit.push(r.f64()?);
            support_end.push(r.u32()?);
        }
        let supports = r.usize()?;
        let lens: Vec<usize> = (0..4u8).filter_map(source_of).map(|s| source_len(s, cx)).collect();
        let (mut support_source, mut support_row) = (Vec::new(), Vec::new());
        for _ in 0..supports {
            let (s, row) = (r.u8()?, r.u32()?);
            let len = *lens.get(s as usize).ok_or_else(|| malformed(format!("a support source code of {s}")))?;
            if row as usize >= len {
                return Err(malformed(format!("a support row past the {len} rows of its source")));
            }
            support_source.push(s);
            support_row.push(row);
        }
        let n = r.usize()?;
        let (mut choice_record, mut choice_path, mut choice, mut choice_score, mut choice_logit) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            choice_record.push(r.u32()?);
            choice_path.push(tag(&mut r, paths.len(), "field path")?);
            choice.push(tag(&mut r, choices.len(), "choice")?);
            choice_score.push(r.f64()?);
            choice_logit.push(r.f64()?);
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a records payload"));
        }
        let layer = Records {
            spans,
            record,
            path,
            score,
            logit,
            support_end,
            support_source,
            support_row,
            record_type,
            instance,
            choice_record,
            choice_path,
            choice,
            choice_score,
            choice_logit,
            types,
            paths,
            choices,
        };
        layer.check().map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
        Ok(layer)
    }
}

/// Registers this crate's codecs: the record layer and its evidence.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<Records>().register::<turbo_analysis::Evidence<Records>>();
}

/// The standard codecs plus this crate's, for callers that build no
/// registry of their own.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    turbo_analysis_layers::export::register_codecs(&mut codecs);
    register_codecs(&mut codecs);
    codecs
}
