//! Delimited (CSV / TSV) catalog parsing and binary snapshot serialization
//! for [`KnowledgeBase`].

use crate::kb::{CatalogError, KbEntity, KnowledgeBase};

const SNAPSHOT_MAGIC: &[u8; 8] = b"TBKB\x01\x00\x00\x00";
const MAX_SNAPSHOT_LEN: usize = 64 * 1024 * 1024;

pub(crate) fn encode_kb(kb: &KnowledgeBase) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(SNAPSHOT_MAGIC);
    put_str(&mut out, kb.provider_id());
    put_u32(&mut out, kb.entities().len() as u32);
    for ent in kb.entities() {
        put_str(&mut out, &ent.id);
        put_str(&mut out, &ent.name);
        put_u32(&mut out, ent.aliases.len() as u32);
        for (alias, weight) in &ent.aliases {
            put_str(&mut out, alias);
            out.extend_from_slice(&weight.to_le_bytes());
        }
        put_u32(&mut out, ent.entity_types.len() as u32);
        for t in &ent.entity_types {
            put_str(&mut out, t);
        }
        put_str(&mut out, &ent.description);
        out.extend_from_slice(&ent.popularity_prior.to_le_bytes());
        put_u32(&mut out, ent.related_ids.len() as u32);
        for rel in &ent.related_ids {
            put_str(&mut out, rel);
        }
    }
    out
}

pub(crate) fn decode_kb(bytes: &[u8], expected_magic: &[u8; 8]) -> Result<KnowledgeBase, CatalogError> {
    if bytes.len() < expected_magic.len() || &bytes[..expected_magic.len()] != expected_magic {
        return Err(CatalogError::SnapshotError("invalid or missing TBKB magic header".into()));
    }
    if bytes.len() > MAX_SNAPSHOT_LEN {
        return Err(CatalogError::SnapshotError("snapshot exceeds maximum allowed byte size".into()));
    }
    let mut pos = expected_magic.len();
    let provider_id = get_str(bytes, &mut pos)?;
    let count = get_u32(bytes, &mut pos)? as usize;
    if count > bytes.len() - pos {
        return Err(CatalogError::SnapshotError("entity count exceeds remaining bytes".into()));
    }
    let mut entities = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let id = get_str(bytes, &mut pos)?;
        let name = get_str(bytes, &mut pos)?;
        let alias_count = get_u32(bytes, &mut pos)? as usize;
        if alias_count > bytes.len() - pos {
            return Err(CatalogError::SnapshotError("alias count exceeds remaining bytes".into()));
        }
        let mut aliases = Vec::with_capacity(alias_count.min(256));
        for _ in 0..alias_count {
            let alias = get_str(bytes, &mut pos)?;
            let weight = get_f64(bytes, &mut pos)?;
            aliases.push((alias, weight));
        }
        let type_count = get_u32(bytes, &mut pos)? as usize;
        if type_count > bytes.len() - pos {
            return Err(CatalogError::SnapshotError("type count exceeds remaining bytes".into()));
        }
        let mut entity_types = Vec::with_capacity(type_count.min(64));
        for _ in 0..type_count {
            entity_types.push(get_str(bytes, &mut pos)?);
        }
        let description = get_str(bytes, &mut pos)?;
        let popularity_prior = get_f64(bytes, &mut pos)?;
        let rel_count = get_u32(bytes, &mut pos)? as usize;
        if rel_count > bytes.len() - pos {
            return Err(CatalogError::SnapshotError("related_ids count exceeds remaining bytes".into()));
        }
        let mut related_ids = Vec::with_capacity(rel_count.min(256));
        for _ in 0..rel_count {
            related_ids.push(get_str(bytes, &mut pos)?);
        }
        entities.push(KbEntity { id, name, aliases, entity_types, description, popularity_prior, related_ids });
    }
    if pos != bytes.len() {
        return Err(CatalogError::SnapshotError(format!(
            "trailing {} bytes after snapshot payload",
            bytes.len() - pos
        )));
    }
    KnowledgeBase::from_entities(provider_id, entities)
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

fn get_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, CatalogError> {
    if *pos + 4 > bytes.len() {
        return Err(CatalogError::SnapshotError("unexpected EOF reading u32".into()));
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[*pos..*pos + 4]);
    *pos += 4;
    Ok(u32::from_le_bytes(buf))
}

fn get_f64(bytes: &[u8], pos: &mut usize) -> Result<f64, CatalogError> {
    if *pos + 8 > bytes.len() {
        return Err(CatalogError::SnapshotError("unexpected EOF reading f64".into()));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[*pos..*pos + 8]);
    *pos += 8;
    Ok(f64::from_le_bytes(buf))
}

fn get_str(bytes: &[u8], pos: &mut usize) -> Result<String, CatalogError> {
    let len = get_u32(bytes, pos)? as usize;
    if *pos + len > bytes.len() {
        return Err(CatalogError::SnapshotError("unexpected EOF reading string".into()));
    }
    let slice = &bytes[*pos..*pos + len];
    *pos += len;
    std::str::from_utf8(slice)
        .map(str::to_owned)
        .map_err(|_| CatalogError::SnapshotError("invalid UTF-8 in snapshot string".into()))
}

pub(crate) fn parse_delimited(provider_id: String, text: &str, delimiter: char) -> Result<KnowledgeBase, CatalogError> {
    let mut entities = Vec::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = split_delimited_line(raw_line, delimiter)
            .map_err(|detail| CatalogError::ParseError { line: line_no, detail })?;
        if entities.is_empty()
            && fields.len() >= 2
            && fields[0].trim().eq_ignore_ascii_case("id")
            && fields[1].trim().eq_ignore_ascii_case("name")
        {
            continue;
        }
        if fields.len() < 5 {
            return Err(CatalogError::ParseError {
                line: line_no,
                detail: format!(
                    "expected at least 5 columns (id, name, aliases, types, description[, prior, related_ids]), got {}",
                    fields.len()
                ),
            });
        }
        let id = fields[0].trim().to_owned();
        let name = fields[1].trim().to_owned();
        let aliases = parse_aliases(fields[2].trim(), line_no)?;
        let entity_types = split_list(fields[3].trim());
        let description = fields[4].trim().to_owned();
        let popularity_prior = if fields.len() >= 6 && !fields[5].trim().is_empty() {
            fields[5].trim().parse::<f64>().map_err(|_| CatalogError::ParseError {
                line: line_no,
                detail: format!("invalid prior float '{}'", fields[5].trim()),
            })?
        } else {
            0.5
        };
        let related_ids = if fields.len() >= 7 { split_list(fields[6].trim()) } else { Vec::new() };

        entities.push(KbEntity { id, name, aliases, entity_types, description, popularity_prior, related_ids });
    }
    KnowledgeBase::from_entities(provider_id, entities)
}

fn split_delimited_line(line: &str, delimiter: char) -> Result<Vec<String>, String> {
    if delimiter == '\t' {
        return Ok(line.split('\t').map(str::to_owned).collect());
    }
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;
    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == delimiter {
            fields.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
    }
    if in_quotes {
        return Err("unclosed quote in delimited record".into());
    }
    fields.push(cur);
    Ok(fields)
}

fn parse_aliases(raw: &str, line_no: usize) -> Result<Vec<(String, f64)>, CatalogError> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in raw.split(['|', ';']) {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        if let Some((alias_part, weight_part)) = item.rsplit_once(':')
            && let Ok(w) = weight_part.trim().parse::<f64>()
        {
            let alias = alias_part.trim();
            if !alias.is_empty() {
                if !w.is_finite() || w <= 0.0 {
                    return Err(CatalogError::ParseError {
                        line: line_no,
                        detail: format!("alias '{alias}' weight {w} must be finite and > 0"),
                    });
                }
                out.push((alias.to_owned(), w));
                continue;
            }
        }
        out.push((item.to_owned(), 1.0));
    }
    Ok(out)
}

fn split_list(raw: &str) -> Vec<String> {
    if raw.is_empty() {
        return Vec::new();
    }
    raw.split(['|', ';', ',']).map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).collect()
}
