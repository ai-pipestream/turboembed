use super::*;

const GRANULARITIES: [Granularity; 5] =
    [Granularity::Day, Granularity::Week, Granularity::Month, Granularity::Quarter, Granularity::Year];

impl LayerCodec for Temporals {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.granularity.len())?;
        for g in &self.granularity {
            w.u8(GRANULARITIES.iter().position(|x| x == g).expect("a granularity in the table") as u8);
        }
        w.usize(self.origin.len())?;
        for o in &self.origin {
            w.u8(match o {
                Origin::Absolute => 0,
                Origin::Relative => 1,
            });
        }
        w.usize(self.year.len())?;
        for y in &self.year {
            w.u32(*y as u32);
        }
        w.usize(self.period.len())?;
        for p in &self.period {
            w.u8(*p);
        }
        w.usize(self.day.len())?;
        for d in &self.day {
            w.u8(*d);
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = spans.len();
        let mut granularity = Vec::new();
        for _ in 0..r.usize()? {
            granularity.push(
                *GRANULARITIES.get(r.u8()? as usize).ok_or_else(|| malformed("a granularity outside the table"))?,
            );
        }
        let mut origin = Vec::new();
        for _ in 0..r.usize()? {
            origin.push(match r.u8()? {
                0 => Origin::Absolute,
                1 => Origin::Relative,
                tag => return Err(malformed(format!("an origin tag of {tag}"))),
            });
        }
        let mut year = Vec::new();
        for _ in 0..r.usize()? {
            year.push(r.u32()? as i32);
        }
        let mut period = Vec::new();
        for _ in 0..r.usize()? {
            period.push(r.u8()?);
        }
        let mut day = Vec::new();
        for _ in 0..r.usize()? {
            day.push(r.u8()?);
        }
        want_len(granularity.len(), n, "granularities")?;
        want_len(origin.len(), n, "origins")?;
        want_len(year.len(), n, "years")?;
        want_len(period.len(), n, "periods")?;
        want_len(day.len(), n, "days")?;
        done(&r, "temporals")?;
        Ok(Temporals { spans, granularity, origin, year, period, day })
    }
}

pub(super) fn put_date(w: &mut Writer, date: Date) {
    w.u32(date.year() as u32);
    w.u8(date.month());
    w.u8(date.day());
}

pub(super) fn get_date(r: &mut Reader<'_>) -> Result<Date, ExportError> {
    let (year, month, day) = (r.u32()? as i32, r.u8()?, r.u8()?);
    Date::new(year, month, day).ok_or_else(|| malformed(format!("a date that does not exist: {year}-{month}-{day}")))
}

impl LayerCodec for DocumentDate {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        match self.date {
            None => w.u8(0),
            Some(date) => {
                w.u8(1);
                put_date(&mut w, date);
            }
        }
        match self.mention {
            None => w.u8(0),
            Some(m) => {
                w.u8(1);
                put_ref(&mut w, &cx.resolve(&Key::<Temporals>::DEFAULT))?;
                w.u32(m.get() as u32);
            }
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let date = match r.u8()? {
            0 => None,
            1 => Some(get_date(&mut r)?),
            tag => return Err(malformed(format!("an optional date tag of {tag}"))),
        };
        let mention = match r.u8()? {
            0 => None,
            1 => {
                let target = get_ref(&mut r)?;
                cx.expect_target::<Temporals>(&target)?;
                let index = r.u32()?;
                cx.require_node(&NodeRef { layer: target, index })?;
                Some(Ix::new(index))
            }
            tag => return Err(malformed(format!("an optional mention tag of {tag}"))),
        };
        done(&r, "document date")?;
        Ok(DocumentDate { date, mention })
    }
}

impl LayerCodec for LanguageSpans {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_tagset(&mut w, &self.languages)?;
        put_tags(&mut w, &self.language)?;
        w.usize(self.script.len())?;
        for s in &self.script {
            w.str(s.name())?;
        }
        put_f64s(&mut w, &self.confidence)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let languages = get_tagset(&mut r)?;
        let language = get_tags(&mut r, &languages)?;
        let mut script = Vec::new();
        for _ in 0..r.usize()? {
            let name = r.string()?;
            script.push(Script::from_name(&name).ok_or_else(|| malformed(format!("an unknown script {name:?}")))?);
        }
        let confidence = get_f64s(&mut r)?;
        want_len(language.len(), spans.len(), "languages")?;
        want_len(script.len(), spans.len(), "scripts")?;
        want_len(confidence.len(), spans.len(), "confidences")?;
        done(&r, "language spans")?;
        Ok(LanguageSpans { spans, language, languages, script, confidence })
    }
}
