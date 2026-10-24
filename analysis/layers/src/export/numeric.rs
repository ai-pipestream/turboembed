use super::*;

const NOTATIONS: [NumberNotation; 2] = [NumberNotation::LatinUs, NumberNotation::LatinEu];
const MULTIPLIERS: [Multiplier; 5] =
    [Multiplier::One, Multiplier::Thousand, Multiplier::Million, Multiplier::Billion, Multiplier::Trillion];
const MARKERS: [ScaleMarker; 3] = [ScaleMarker::None, ScaleMarker::Suffix, ScaleMarker::Word];
const BASES: [OrdinalBase; 3] = [OrdinalBase::Start, OrdinalBase::End, OrdinalBase::Current];
const TIME_KINDS: [TimeKind; 8] = [
    TimeKind::Date,
    TimeKind::Time,
    TimeKind::DateTime,
    TimeKind::DateRange,
    TimeKind::TimeRange,
    TimeKind::DateTimeRange,
    TimeKind::Duration,
    TimeKind::Set,
];
const TIME_GRAINS: [TimeGrain; 14] = [
    TimeGrain::Second,
    TimeGrain::Minute,
    TimeGrain::Hour,
    TimeGrain::PartOfDay,
    TimeGrain::Day,
    TimeGrain::Week,
    TimeGrain::Weekend,
    TimeGrain::Month,
    TimeGrain::Quarter,
    TimeGrain::Half,
    TimeGrain::Season,
    TimeGrain::Year,
    TimeGrain::Decade,
    TimeGrain::Century,
];
const RELATIONS: [Relation; 8] = [
    Relation::None,
    Relation::Before,
    Relation::After,
    Relation::Since,
    Relation::Until,
    Relation::More,
    Relation::Less,
    Relation::Approx,
];
const PORTIONS: [Portion; 6] =
    [Portion::Whole, Portion::Start, Portion::Mid, Portion::End, Portion::Approx, Portion::Undefined];
const PII_CONFIDENCES: [PiiConfidence; 3] =
    [PiiConfidence::Format, PiiConfidence::Checksum, PiiConfidence::LabelledChecksum];

impl LayerCodec for Numbers {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_u32s(&mut w, &self.numeral_end)?;
        put_bools(&mut w, &self.negative)?;
        put_strs(&mut w, &self.coefficient)?;
        put_u32s(&mut w, &self.scale)?;
        put_variants(&mut w, &MULTIPLIERS, &self.multiplier)?;
        put_variants(&mut w, &MARKERS, &self.marker)?;
        put_variants(&mut w, &NOTATIONS, &self.notation)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let numeral_end = get_u32s(&mut r)?;
        let negative = get_bools(&mut r)?;
        let coefficient = get_strs(&mut r)?;
        let scale = get_u32s(&mut r)?;
        let multiplier = get_variants(&mut r, &MULTIPLIERS, "multiplier")?;
        let marker = get_variants(&mut r, &MARKERS, "scale marker")?;
        let notation = get_variants(&mut r, &NOTATIONS, "number notation")?;
        let n = spans.len();
        want_len(numeral_end.len(), n, "numeral ends")?;
        want_len(negative.len(), n, "negatives")?;
        want_len(coefficient.len(), n, "coefficients")?;
        want_len(scale.len(), n, "scales")?;
        want_len(multiplier.len(), n, "multipliers")?;
        want_len(marker.len(), n, "markers")?;
        want_len(notation.len(), n, "notations")?;
        done(&r, "numbers")?;
        Ok(Numbers { spans, numeral_end, negative, coefficient, scale, multiplier, marker, notation })
    }
}

impl LayerCodec for Money {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_bools(&mut w, &self.negative)?;
        put_strs(&mut w, &self.coefficient)?;
        put_i32s(&mut w, &self.scale)?;
        w.usize(self.currency.len())?;
        for c in &self.currency {
            w.str(c.as_str())?;
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let negative = get_bools(&mut r)?;
        let coefficient = get_strs(&mut r)?;
        let scale = get_i32s(&mut r)?;
        let mut currency = Vec::new();
        for _ in 0..r.usize()? {
            let code = r.string()?;
            currency
                .push(Currency::parse(&code).ok_or_else(|| malformed(format!("an invalid currency code {code:?}")))?);
        }
        let n = spans.len();
        want_len(negative.len(), n, "negatives")?;
        want_len(coefficient.len(), n, "coefficients")?;
        want_len(scale.len(), n, "scales")?;
        want_len(currency.len(), n, "currencies")?;
        done(&r, "money")?;
        Ok(Money { spans, negative, coefficient, scale, currency })
    }
}

impl LayerCodec for Ordinals {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_variants(&mut w, &BASES, &self.base)?;
        put_i64s(&mut w, &self.offset)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let base = get_variants(&mut r, &BASES, "ordinal base")?;
        let offset = get_i64s(&mut r)?;
        want_len(base.len(), spans.len(), "bases")?;
        want_len(offset.len(), spans.len(), "offsets")?;
        done(&r, "ordinals")?;
        Ok(Ordinals { spans, base, offset })
    }
}

impl LayerCodec for Quantities {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_bools(&mut w, &self.negative)?;
        put_strs(&mut w, &self.coefficient)?;
        put_i32s(&mut w, &self.scale)?;
        put_strs(&mut w, &self.unit)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let negative = get_bools(&mut r)?;
        let coefficient = get_strs(&mut r)?;
        let scale = get_i32s(&mut r)?;
        let unit = get_strs(&mut r)?;
        let n = spans.len();
        want_len(negative.len(), n, "negatives")?;
        want_len(coefficient.len(), n, "coefficients")?;
        want_len(scale.len(), n, "scales")?;
        want_len(unit.len(), n, "units")?;
        done(&r, "quantities")?;
        Ok(Quantities { spans, negative, coefficient, scale, unit })
    }
}

impl LayerCodec for Times {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_variants(&mut w, &TIME_KINDS, &self.kind)?;
        put_variants(&mut w, &TIME_GRAINS, &self.grain)?;
        w.usize(self.modifier.len())?;
        for m in &self.modifier {
            put_variant(&mut w, &RELATIONS, m.relation);
            put_variant(&mut w, &PORTIONS, m.portion);
        }
        put_opt_i64s(&mut w, &self.start)?;
        put_opt_i64s(&mut w, &self.end)?;
        put_opt_i64s(&mut w, &self.duration_ms)?;
        put_bools(&mut w, &self.ambiguous)?;
        put_strs(&mut w, &self.timex)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let kind = get_variants(&mut r, &TIME_KINDS, "time kind")?;
        let grain = get_variants(&mut r, &TIME_GRAINS, "time grain")?;
        let mut modifier = Vec::new();
        for _ in 0..r.usize()? {
            modifier.push(Modifier {
                relation: get_variant(&mut r, &RELATIONS, "relation")?,
                portion: get_variant(&mut r, &PORTIONS, "portion")?,
            });
        }
        let start = get_opt_i64s(&mut r)?;
        let end = get_opt_i64s(&mut r)?;
        let duration_ms = get_opt_i64s(&mut r)?;
        let ambiguous = get_bools(&mut r)?;
        let timex = get_strs(&mut r)?;
        let n = spans.len();
        want_len(kind.len(), n, "kinds")?;
        want_len(grain.len(), n, "grains")?;
        want_len(modifier.len(), n, "modifiers")?;
        want_len(start.len(), n, "starts")?;
        want_len(end.len(), n, "ends")?;
        want_len(duration_ms.len(), n, "durations")?;
        want_len(ambiguous.len(), n, "ambiguous flags")?;
        want_len(timex.len(), n, "timex values")?;
        done(&r, "times")?;
        Ok(Times { spans, kind, grain, modifier, start, end, duration_ms, ambiguous, timex })
    }
}

impl LayerCodec for Pii {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.kind.len())?;
        for &k in &self.kind {
            if let PiiKind::Custom(name) = k
                && (name.is_empty() || name.len() > PiiKind::MAX_CUSTOM_NAME_BYTES || PiiKind::parse(name).is_some())
            {
                return Err(malformed(format!("invalid custom pii kind {name:?}")));
            }
            PiiKind::remember_custom(k);
            w.str(k.name())?;
        }
        put_variants(&mut w, &PII_CONFIDENCES, &self.confidence)?;
        put_strs(&mut w, &self.normalized)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut kind = Vec::new();
        for _ in 0..r.usize()? {
            let name = r.string()?;
            kind.push(
                PiiKind::parse(&name)
                    .or_else(|| PiiKind::intern_custom(&name))
                    .ok_or_else(|| malformed(format!("an unknown pii kind {name:?}")))?,
            );
        }
        let confidence = get_variants(&mut r, &PII_CONFIDENCES, "pii confidence")?;
        let normalized = get_strs(&mut r)?;
        want_len(kind.len(), spans.len(), "kinds")?;
        want_len(confidence.len(), spans.len(), "confidences")?;
        want_len(normalized.len(), spans.len(), "normalized values")?;
        done(&r, "pii")?;
        Ok(Pii { spans, kind, confidence, normalized })
    }
}

impl LayerCodec for Regions {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.country.len())?;
        for c in &self.country {
            w.str(c.as_str())?;
        }
        put_f64s(&mut w, &self.share)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let mut country = Vec::new();
        for _ in 0..r.usize()? {
            let code = r.string()?;
            country.push(RegionCode::parse(&code).map_err(|e| malformed(e.to_string()))?);
        }
        let share = get_f64s(&mut r)?;
        want_len(share.len(), country.len(), "shares")?;
        done(&r, "regions")?;
        Ok(Regions { country, share })
    }
}
