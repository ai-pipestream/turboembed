use super::*;

pub(super) fn put_error(w: &mut Writer, error: &ErrorRecord) -> Result<(), ExportError> {
    use crate::bytes::SourceError;
    use crate::offsets::{OffsetError, OffsetUnit};
    match error {
        ErrorRecord::Missing(r) => {
            w.u8(0);
            put_layer_ref(w, r)?;
        }
        ErrorRecord::Duplicate(r) => {
            w.u8(1);
            put_layer_ref(w, r)?;
        }
        ErrorRecord::Clash(r) => {
            w.u8(2);
            put_layer_ref(w, r)?;
        }
        ErrorRecord::Bounds { layer, view, detail } => {
            w.u8(3);
            put_layer_ref(w, layer)?;
            w.str(view)?;
            w.str(detail)?;
        }
        ErrorRecord::Misaligned { layer, anchor, len, anchor_len } => {
            w.u8(4);
            put_layer_ref(w, layer)?;
            put_layer_ref(w, anchor)?;
            w.u32(*len);
            w.u32(*anchor_len);
        }
        ErrorRecord::Unaligned { layer, on, read } => {
            w.u8(5);
            put_layer_ref(w, layer)?;
            put_layer_ref(w, on)?;
            w.str(read)?;
        }
        ErrorRecord::Invalid { layer, detail } => {
            w.u8(6);
            put_layer_ref(w, layer)?;
            w.str(detail)?;
        }
        ErrorRecord::Pipeline(detail) => {
            w.u8(7);
            w.str(detail)?;
        }
        ErrorRecord::Annotator { annotator, detail } => {
            w.u8(8);
            w.str(annotator)?;
            w.str(detail)?;
        }
        ErrorRecord::Skipped(reason) => {
            w.u8(9);
            w.str(reason)?;
        }
        ErrorRecord::Offset { unit, at, reason } => {
            w.u8(10);
            w.u8(match unit {
                OffsetUnit::Utf16 => 0,
                OffsetUnit::CodePoint => 1,
            });
            w.u32(*at);
            match reason {
                OffsetError::PastEnd { len } => {
                    w.u8(0);
                    w.u32(*len);
                }
                OffsetError::InsideCharacter => w.u8(1),
            }
        }
        ErrorRecord::Source(e) => {
            w.u8(11);
            match e {
                SourceError::TooLong { len } => {
                    w.u8(0);
                    w.u64(*len);
                }
                SourceError::Misaligned(detail) => {
                    w.u8(1);
                    w.str(detail)?;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn get_error(r: &mut Reader<'_>) -> Result<ErrorRecord, ExportError> {
    use crate::bytes::SourceError;
    use crate::offsets::{OffsetError, OffsetUnit};
    Ok(match r.u8()? {
        0 => ErrorRecord::Missing(get_layer_ref(r)?),
        1 => ErrorRecord::Duplicate(get_layer_ref(r)?),
        2 => ErrorRecord::Clash(get_layer_ref(r)?),
        3 => ErrorRecord::Bounds { layer: get_layer_ref(r)?, view: r.string()?, detail: r.string()? },
        4 => {
            let layer = get_layer_ref(r)?;
            let anchor = get_layer_ref(r)?;
            ErrorRecord::Misaligned { layer, anchor, len: r.u32()?, anchor_len: r.u32()? }
        }
        5 => {
            let layer = get_layer_ref(r)?;
            let on = get_layer_ref(r)?;
            ErrorRecord::Unaligned { layer, on, read: r.string()? }
        }
        6 => ErrorRecord::Invalid { layer: get_layer_ref(r)?, detail: r.string()? },
        7 => ErrorRecord::Pipeline(r.string()?),
        8 => ErrorRecord::Annotator { annotator: r.string()?, detail: r.string()? },
        9 => ErrorRecord::Skipped(r.string()?),
        10 => {
            let unit = match r.u8()? {
                0 => OffsetUnit::Utf16,
                1 => OffsetUnit::CodePoint,
                tag => return Err(malformed(format!("an offset unit tag of {tag}"))),
            };
            let at = r.u32()?;
            let reason = match r.u8()? {
                0 => OffsetError::PastEnd { len: r.u32()? },
                1 => OffsetError::InsideCharacter,
                tag => return Err(malformed(format!("an offset reason tag of {tag}"))),
            };
            ErrorRecord::Offset { unit, at, reason }
        }
        11 => ErrorRecord::Source(match r.u8()? {
            0 => SourceError::TooLong { len: r.u64()? },
            1 => SourceError::Misaligned(r.string()?),
            tag => return Err(malformed(format!("a source error tag of {tag}"))),
        }),
        tag => return Err(malformed(format!("an error tag of {tag}"))),
    })
}
