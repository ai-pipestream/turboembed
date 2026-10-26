use std::collections::BTreeMap;
use std::path::Path;

pub struct Response {
    pub context: u32,
    pub text: String,
    pub sarcastic: bool,
}

pub fn load(path: &Path) -> Vec<Response> {
    let csv = std::fs::read_to_string(path).unwrap();
    let mut grouped: BTreeMap<(u32, u32, String), Vec<u8>> = BTreeMap::new();
    for row in records(&csv).into_iter().skip(1) {
        if row.len() != 9 {
            continue;
        }
        let (Ok(context), Ok(speaker), Ok(rating)) =
            (row[0].parse::<u32>(), row[5].parse::<u32>(), row[8].parse::<u8>())
        else {
            continue;
        };
        if row[3].trim().is_empty() {
            continue;
        }
        grouped.entry((context, speaker, row[3].clone())).or_default().push(rating);
    }
    grouped
        .into_iter()
        .map(|((context, _, text), mut ratings)| {
            ratings.sort_unstable();
            Response { context, text, sarcastic: ratings[ratings.len() / 2] >= 4 }
        })
        .collect()
}

fn records(csv: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = csv.chars().peekable();
    let mut quoted = false;
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => quoted = true,
            ',' => {
                row.push(std::mem::take(&mut field));
            }
            '\n' => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            '\r' => {}
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_reader_handles_quotes_newlines_and_empty_fields() {
        let rows = records("a,b,c\n1,\"two \"\"quoted\"\"\nlines\",3\n4,,5\n");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1][1], "two \"quoted\"\nlines");
        assert_eq!(rows[2][1], "");
    }
}
