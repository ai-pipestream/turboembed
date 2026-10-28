//! The layers of the `formats` goldens, as the golden reader needs them.

use turbo_goldens::{ColumnSpec, ColumnType, GoldenAnchor, Shape};

const U32: ColumnType = ColumnType::U32;
const STR: ColumnType = ColumnType::Str;
pub(crate) const SAMPLE_TOKENS: &str = "turbo:sample-tokens";

pub(crate) fn shapes() -> Vec<Shape> {
    const WRITTEN: &[ColumnSpec] = &[ColumnSpec::new("written", STR)];
    const ERRORS: &[ColumnSpec] =
        &[ColumnSpec::new("after", U32), ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    const TOKENS: &[ColumnSpec] = &[ColumnSpec::new("sample", U32), ColumnSpec::new("text", STR)];
    const TAG: &[ColumnSpec] = &[ColumnSpec::new("tag", STR)];
    const LEMMA: &[ColumnSpec] = &[ColumnSpec::new("lemma", STR)];
    const CHUNK: &[ColumnSpec] = &[ColumnSpec::new("chunk", STR)];
    const SPANS: &[ColumnSpec] = &[
        ColumnSpec::new("sample", U32),
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("type", STR),
    ];
    const PHRASE_ERRORS: &[ColumnSpec] =
        &[ColumnSpec::new("sample", U32), ColumnSpec::new("class", STR), ColumnSpec::new("message", STR)];
    const TEXT: &[ColumnSpec] = &[ColumnSpec::new("text", STR)];
    const BYTES: &[ColumnSpec] =
        &[ColumnSpec::new("sample", U32), ColumnSpec::new("start", U32), ColumnSpec::new("end", U32)];
    const LABEL: &[ColumnSpec] = &[ColumnSpec::new("label", STR)];
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    let aligned =
        |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Aligned(SAMPLE_TOKENS.into()), columns };
    vec![
        doc("turbo:samples", WRITTEN),
        doc("turbo:read-errors", ERRORS),
        doc(SAMPLE_TOKENS, TOKENS),
        aligned("turbo:sample-tags", TAG),
        aligned("turbo:sample-lemmas", LEMMA),
        aligned("turbo:sample-chunks", CHUNK),
        doc("turbo:names", SPANS),
        doc("turbo:phrases", SPANS),
        doc("turbo:phrase-errors", PHRASE_ERRORS),
        doc("turbo:sample-texts", TEXT),
        doc("turbo:sample-spans", BYTES),
        doc("turbo:sample-labels", LABEL),
    ]
}
