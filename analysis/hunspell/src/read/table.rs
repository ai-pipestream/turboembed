//! Tables of the affix file whose first line gives the number of entries:
//! OpenNLP's `HunspellAffixTable`.

use crate::error::LoadError;
use crate::text::parse_int;

/// One entry of a table: its fields, the directive first, and its line.
pub(crate) struct TableEntry<'a> {
    pub(crate) fields: &'a [&'a str],
    pub(crate) line: usize,
}

/// The entries of the table `tag`, or `None` when no line starts with it.
/// The first such line must hold exactly a count; every later one is an
/// entry of `min` to `max` fields, and there must be exactly that many.
pub(crate) fn read<'a>(
    lines: &'a [Vec<&'a str>],
    tag: &'static str,
    min: usize,
    max: usize,
) -> Result<Option<Vec<TableEntry<'a>>>, LoadError> {
    let mut entries = Vec::new();
    let mut count: Option<usize> = None;
    for (i, fields) in lines.iter().enumerate() {
        if fields.first() != Some(&tag) {
            continue;
        }
        let line = i + 1;
        match count {
            None => {
                let n = (fields.len() == 2).then(|| parse_int(fields[1])).flatten();
                match n {
                    Some(n) if n >= 0 => count = Some(n as usize),
                    _ => return Err(LoadError::InvalidTableCount { tag, line }),
                }
            }
            Some(n) => {
                if fields.len() < min || fields.len() > max || entries.len() == n {
                    return Err(LoadError::InvalidTableEntry { tag, line });
                }
                entries.push(TableEntry { fields, line });
            }
        }
    }
    match count {
        None => Ok(None),
        Some(n) if entries.len() != n => Err(LoadError::TableCount { tag, expected: n as i32, found: entries.len() }),
        Some(_) => Ok(Some(entries)),
    }
}
