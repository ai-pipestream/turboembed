//! What the tests share: where the corpora are, and a directory of its
//! own for a test to write a bundle in.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn testdata() -> PathBuf {
    repo().join("testdata")
}

/// One JSON string per line.
pub fn jsonl_strings(path: &Path) -> Vec<String> {
    let file = std::fs::read_to_string(path).unwrap();
    file.lines().map(|l| parse(l).unwrap().as_str().unwrap().to_owned()).collect()
}

/// testdata/tokenizer-texts.jsonl: one `{"text": ...}` per line.
pub fn tokenizer_texts() -> Vec<String> {
    let file = std::fs::read_to_string(testdata().join("tokenizer-texts.jsonl")).unwrap();
    file.lines().map(|l| parse(l).unwrap()["text"].as_str().unwrap().to_owned()).collect()
}

/// The directory of the JVM goldens' BPE inputs.
pub fn bpe_inputs_dir() -> PathBuf {
    repo().join("analysis/goldens/data/inputs/bpe")
}

/// The BPE inputs of the JVM goldens, by name, in name order.
pub fn bpe_inputs() -> Vec<(String, String)> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(bpe_inputs_dir()).unwrap().map(|e| e.unwrap().path()).collect();
    names.sort();
    names
        .into_iter()
        .map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), std::fs::read_to_string(&p).unwrap()))
        .collect()
}

/// A directory of its own, removed on drop. Tests run side by side.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> TempDir {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("turbo-bpe-trainer-{}-{name}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Number(n) if *n >= 0.0 && n.fract() == 0.0 && *n <= u64::MAX as f64 => Some(*n as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

static NULL: Json = Json::Null;

impl std::ops::Index<&str> for Json {
    type Output = Json;

    fn index(&self, key: &str) -> &Json {
        self.get(key).unwrap_or(&NULL)
    }
}

impl std::ops::Index<usize> for Json {
    type Output = Json;

    fn index(&self, at: usize) -> &Json {
        self.as_array().and_then(|a| a.get(at)).unwrap_or(&NULL)
    }
}

pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = JsonParser { text, at: 0 };
    let value = p.value()?;
    p.space();
    if p.at == text.len() { Ok(value) } else { Err(format!("trailing JSON at byte {}", p.at)) }
}

struct JsonParser<'a> {
    text: &'a str,
    at: usize,
}

impl JsonParser<'_> {
    fn space(&mut self) {
        while self.text.as_bytes().get(self.at).is_some_and(u8::is_ascii_whitespace) {
            self.at += 1;
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.space();
        match self.text.as_bytes().get(self.at).copied() {
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => self.string().map(Json::String),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(format!("expected JSON at byte {}", self.at)),
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.text[self.at..].starts_with(word) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(format!("expected {word} at byte {}", self.at))
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.at += 1;
        let mut out = String::new();
        loop {
            let c = self.text[self.at..].chars().next().ok_or("unterminated JSON string")?;
            self.at += c.len_utf8();
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let e = self.text[self.at..].chars().next().ok_or("unterminated JSON escape")?;
                    self.at += e.len_utf8();
                    match e {
                        '"' | '\\' | '/' => out.push(e),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => out.push(self.escape()?),
                        _ => return Err(format!("bad JSON escape at byte {}", self.at)),
                    }
                }
                '\u{0}'..='\u{1f}' => return Err("control character in JSON string".into()),
                _ => out.push(c),
            }
        }
    }

    fn hex(&mut self) -> Result<u16, String> {
        let end = self.at + 4;
        let digits = self.text.get(self.at..end).ok_or("short JSON Unicode escape")?;
        self.at = end;
        u16::from_str_radix(digits, 16).map_err(|_| "bad JSON Unicode escape".into())
    }

    fn escape(&mut self) -> Result<char, String> {
        let high = self.hex()?;
        let cp = if (0xD800..=0xDBFF).contains(&high) {
            if !self.text[self.at..].starts_with("\\u") {
                return Err("unpaired high surrogate".into());
            }
            self.at += 2;
            let low = self.hex()?;
            if !(0xDC00..=0xDFFF).contains(&low) {
                return Err("invalid low surrogate".into());
            }
            0x10000 + (((high - 0xD800) as u32) << 10) + (low - 0xDC00) as u32
        } else if (0xDC00..=0xDFFF).contains(&high) {
            return Err("unpaired low surrogate".into());
        } else {
            high as u32
        };
        char::from_u32(cp).ok_or_else(|| "invalid Unicode escape".into())
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        while self
            .text
            .as_bytes()
            .get(self.at)
            .is_some_and(|b| matches!(b, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'))
        {
            self.at += 1;
        }
        self.text[start..self.at].parse().map(Json::Number).map_err(|_| format!("bad JSON number at byte {start}"))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.at += 1;
        let mut out = Vec::new();
        self.space();
        if self.text.as_bytes().get(self.at) == Some(&b']') {
            self.at += 1;
            return Ok(Json::Array(out));
        }
        loop {
            out.push(self.value()?);
            self.space();
            match self.text.as_bytes().get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Array(out));
                }
                _ => return Err(format!("expected comma or array end at byte {}", self.at)),
            }
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.at += 1;
        let mut out = Vec::new();
        self.space();
        if self.text.as_bytes().get(self.at) == Some(&b'}') {
            self.at += 1;
            return Ok(Json::Object(out));
        }
        loop {
            self.space();
            let key = self.string()?;
            self.space();
            if self.text.as_bytes().get(self.at) != Some(&b':') {
                return Err(format!("expected colon at byte {}", self.at));
            }
            self.at += 1;
            out.push((key, self.value()?));
            self.space();
            match self.text.as_bytes().get(self.at) {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(out));
                }
                _ => return Err(format!("expected comma or object end at byte {}", self.at)),
            }
        }
    }
}
