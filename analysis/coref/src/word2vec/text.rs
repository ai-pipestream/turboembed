//! Reading the word2vec and GloVe text formats.

use crate::MapWordVectors;

/// Reads word2vec text (a `count dimension` header) or GloVe text (no
/// header) into a [`MapWordVectors`]; `Err` on a ragged row, a value
/// that does not parse or is not finite, a count that disagrees with
/// the rows, or no rows.
pub fn read_text(text: &str) -> Result<MapWordVectors, String> {
    let mut lines = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()).peekable();
    let header = lines.peek().and_then(|(_, l)| parse_header(l));
    if header.is_some() {
        lines.next();
    }
    let mut rows = Vec::new();
    for (n, line) in lines {
        let mut fields = line.split_whitespace();
        let word = fields.next().expect("a non-blank line has a field");
        let values = fields
            .map(|f| f.parse::<f32>().map_err(|e| format!("line {}: value {f:?}: {e}", n + 1)))
            .collect::<Result<Vec<f32>, String>>()?;
        rows.push((n, word.to_owned(), values));
    }
    let dimension = match (header, rows.first()) {
        (_, None) => return Err("no vectors".into()),
        (Some((_, d)), _) => d,
        (None, Some((_, _, v))) => v.len(),
    };
    if dimension == 0 {
        return Err("vectors have dimension 0".into());
    }
    if let Some((count, _)) = header
        && count != rows.len()
    {
        return Err(format!("header counts {count} vectors, found {}", rows.len()));
    }
    let mut table = MapWordVectors::new(dimension);
    for (n, word, values) in rows {
        table.insert(word, values).map_err(|e| format!("line {}: {e}", n + 1))?;
    }
    Ok(table)
}

/// A line of exactly two unsigned integers.
fn parse_header(line: &str) -> Option<(usize, usize)> {
    let mut fields = line.split_whitespace();
    let count = fields.next()?.parse().ok()?;
    let dimension = fields.next()?.parse().ok()?;
    fields.next().is_none().then_some((count, dimension))
}
