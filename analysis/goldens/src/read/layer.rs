use turbo_analysis_layers::Strs;

use crate::compare::Shape;
use crate::escape::unescape_into;
use crate::format::{Column, ColumnData, ColumnType, GoldenAnchor, GoldenLayer, Header};

use super::{GoldenError, Lines, field_error, parse_f64, parse_u32, u32_field};

/// The most rows reserved for ahead of reading them. The declared count
/// is only a claim until the rows are there, so a file declaring u32::MAX
/// rows is refused when it runs out of lines, not by allocating for them.
const RESERVE_CAP: usize = 4096;

pub(super) fn read_layer(
    lines: &mut Lines<'_>,
    head: &str,
    header: &Header,
    shapes: &[Shape],
    earlier: &[GoldenLayer],
) -> Result<GoldenLayer, GoldenError> {
    let line = lines.line;
    let fields: Vec<&str> = head.split('\t').collect();
    if fields[0] != "layer" {
        return Err(GoldenError::Record { line, expected: "a layer record".into(), found: head.into() });
    }
    if fields.len() < 5 {
        return Err(GoldenError::Fields {
            line,
            expected: "layer <id> <anchor> <row count> <name:type>...".into(),
            found: fields.len(),
        });
    }
    let id = fields[1];
    let anchor = match fields[2] {
        "text" => GoldenAnchor::Text,
        "document" => GoldenAnchor::Document,
        a => match a.strip_prefix("aligned:") {
            Some(of) if !of.is_empty() => GoldenAnchor::Aligned(of.into()),
            _ => return Err(field_error(line, "layer anchor", "text, document or aligned:<layer id>", a)),
        },
    };
    let rows = u32_field(line, "layer row count", fields[3])? as usize;
    let mut specs = Vec::with_capacity(fields.len() - 4);
    for f in &fields[4..] {
        let spec = f.split_once(':').and_then(|(n, t)| Some((n, ColumnType::parse(t)?))).filter(|(n, _)| !n.is_empty());
        specs.push(spec.ok_or_else(|| field_error(line, "column", "<name>:<u32|f64|str>", f))?);
    }

    // The layer's type decides its anchor and columns.
    let Some(shape) = shapes.iter().find(|s| s.id == id) else {
        return Err(GoldenError::UnknownLayer {
            line,
            layer: id.into(),
            known: shapes.iter().map(|s| s.id.clone()).collect(),
        });
    };
    if earlier.iter().any(|l| l.id == id) {
        return Err(GoldenError::Duplicate { line, layer: id.into() });
    }
    if shape.anchor != anchor {
        return Err(GoldenError::Anchor {
            line,
            layer: id.into(),
            expected: shape.anchor.to_string(),
            found: anchor.to_string(),
        });
    }
    let found: Vec<String> = specs.iter().map(|(n, t)| format!("{n}:{t}")).collect();
    let expected: Vec<String> = shape.columns.iter().map(|c| c.to_string()).collect();
    if found != expected {
        return Err(GoldenError::Columns { line, layer: id.into(), expected, found });
    }
    if let GoldenAnchor::Aligned(of) = &anchor {
        let Some(target) = earlier.iter().find(|l| &l.id == of) else {
            return Err(GoldenError::AnchorLayer { line, layer: id.into(), anchor: of.clone() });
        };
        if target.len() != rows {
            return Err(GoldenError::AlignedRows {
                line,
                layer: id.into(),
                anchor: of.clone(),
                expected: target.len(),
                found: rows,
            });
        }
    }

    let mut columns: Vec<Column> = specs
        .iter()
        .map(|&(name, ty)| {
            let mut data = ColumnData::empty(ty);
            match &mut data {
                ColumnData::U32(v) => v.reserve(rows.min(RESERVE_CAP)),
                ColumnData::F64(v) => v.reserve(rows.min(RESERVE_CAP)),
                ColumnData::Str(_) => {}
            }
            Column { name: name.into(), data }
        })
        .collect();
    // Spans are checked as they are read, for the layers that have them.
    let span_cols = match anchor {
        GoldenAnchor::Text => {
            let start = specs.iter().position(|s| s.0 == "start");
            let end = specs.iter().position(|s| s.0 == "end");
            start.zip(end)
        }
        _ => None,
    };
    // A layer of UTF-16 spans (`turbo:utf16<L>`) has its spans in UTF-16
    // units of the input, in the order of the layer it is aligned to.
    let unit_cols = {
        let start = specs.iter().position(|s| s == &("start16", ColumnType::U32));
        let end = specs.iter().position(|s| s == &("end16", ColumnType::U32));
        start.zip(end)
    };
    let end_line = format!("end\t{id}");
    let mut buf = String::new();
    let mut last_start = 0;
    for r in 0..rows {
        let Some(row) = lines.next() else {
            return Err(GoldenError::Eof { line: lines.line + 1, expected: format!("row {r} of {id}") });
        };
        let line = lines.line;
        if let Err(e) = read_row(line, row, &mut columns, &mut buf) {
            // An early end line is the clearer error.
            return Err(if row == end_line {
                GoldenError::Rows { line, layer: id.into(), declared: rows, found: r }
            } else {
                e
            });
        }
        if let Some((s, e)) = span_cols {
            let (ColumnData::U32(starts), ColumnData::U32(ends)) = (&columns[s].data, &columns[e].data) else {
                unreachable!("start and end are u32 columns, as the shape says");
            };
            let (start, end) = (starts[r], ends[r]);
            let detail = if start > end {
                Some(format!("the span {start}..{end} ends before it starts"))
            } else if end > header.input.utf8_len {
                Some(format!("the span {start}..{end} ends past the input's {} bytes", header.input.utf8_len))
            } else if start < last_start {
                Some(format!("the span {start}..{end} starts before the row before it, expected text order"))
            } else {
                None
            };
            if let Some(detail) = detail {
                return Err(GoldenError::Span { line, layer: id.into(), detail });
            }
            last_start = start;
        }
        if let Some((s, e)) = unit_cols {
            let (ColumnData::U32(starts), ColumnData::U32(ends)) = (&columns[s].data, &columns[e].data) else {
                unreachable!("start16 and end16 are u32 columns, as found");
            };
            let (start, end) = (starts[r], ends[r]);
            let detail = if start > end {
                Some(format!("the UTF-16 span {start}..{end} ends before it starts"))
            } else if end > header.input.utf16_len {
                Some(format!(
                    "the UTF-16 span {start}..{end} ends past the input's {} UTF-16 units",
                    header.input.utf16_len
                ))
            } else {
                None
            };
            if let Some(detail) = detail {
                return Err(GoldenError::Span { line, layer: id.into(), detail });
            }
        }
    }
    let l = lines.expect(&format!("\"end\t{id}\""))?;
    if l != end_line {
        return Err(GoldenError::Record {
            line: lines.line,
            expected: format!("\"end\\t{id}\" after {rows} rows"),
            found: l.into(),
        });
    }
    Ok(GoldenLayer { id: id.into(), anchor, columns })
}

/// One row, a field per column, each appended to its column.
fn read_row(line: usize, row: &str, columns: &mut [Column], buf: &mut String) -> Result<(), GoldenError> {
    let n = row.split('\t').count();
    if n != columns.len() {
        let names: Vec<String> = columns.iter().map(|c| format!("<{}:{}>", c.name, c.data.ty())).collect();
        return Err(GoldenError::Fields { line, expected: names.join(" "), found: n });
    }
    for (field, col) in row.split('\t').zip(columns.iter_mut()) {
        match &mut col.data {
            ColumnData::U32(v) => v.push(
                parse_u32(field)
                    .ok_or_else(|| field_error(line, &col.name, "a u32 in decimal, no sign or leading zero", field))?,
            ),
            ColumnData::F64(v) => v.push(parse_f64(field).ok_or_else(|| {
                field_error(line, &col.name, "16 lowercase hex digits of bits, a space, a decimal", field)
            })?),
            ColumnData::Str(v) => push_str(line, &col.name, field, v, buf)?,
        }
    }
    Ok(())
}

fn push_str(line: usize, name: &str, field: &str, out: &mut Strs, buf: &mut String) -> Result<(), GoldenError> {
    if !field.contains('\\') && !field.bytes().any(|b| b < 0x20) {
        out.push(field);
        return Ok(());
    }
    buf.clear();
    unescape_into(field, buf).map_err(|error| GoldenError::Escape { line, field: name.into(), error })?;
    out.push(buf);
    Ok(())
}
