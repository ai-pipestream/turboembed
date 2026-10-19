use super::*;
impl Unigram {
    /// The model from the tokenizer file, checked against what the
    /// manifest says of it. `unk_id` is the manifest's SPECIAL_UNK id.
    pub fn parse(file: &str, json: &Value, m: &manifest::Unigram, unk_id: u32) -> Result<Unigram> {
        let disagree = |what: &str| invalid(format!("{file}: {what} is not what manifest.json says"));
        let mut chars = m.metaspace.chars();
        let metaspace = chars
            .next()
            .filter(|_| chars.next().is_none())
            .ok_or_else(|| invalid(format!("{file}: metaspace must be one character")))?;
        let model = &json["model"];
        if model["type"] != "Unigram" {
            return Err(invalid(format!("{file}: model.type is {}, not Unigram", model["type"])));
        }
        if model["unk_id"] != unk_id {
            return Err(disagree("model.unk_id"));
        }
        if boolean(file, model, "byte_fallback")?.unwrap_or(false) {
            return Err(invalid(format!("{file}: model.byte_fallback: the core has no byte fallback")));
        }
        let raw = model["vocab"].as_array().ok_or_else(|| invalid(format!("{file}: model.vocab is not an array")))?;
        if raw.len() > i32::MAX as usize + 1 {
            return Err(invalid(format!("{file}: vocabulary ids exceed INT32_MAX")));
        }
        let mut pieces = Vec::with_capacity(raw.len());
        let mut scores = Vec::with_capacity(raw.len());
        for (id, entry) in raw.iter().enumerate() {
            let (piece, score) = match entry.as_array().map(Vec::as_slice) {
                Some([p, s]) => (p.as_str(), s.as_f64()),
                _ => (None, None),
            };
            let (Some(piece), Some(score)) = (piece, score) else {
                return Err(invalid(format!("{file}: model.vocab[{id}] is not [piece, score]")));
            };
            if piece.is_empty() {
                return Err(invalid(format!("{file}: model.vocab[{id}] is an empty piece")));
            }
            pieces.push(Box::<str>::from(piece));
            scores.push(score);
        }
        if (unk_id as usize) >= scores.len() {
            return Err(invalid(format!("{file}: model.unk_id {unk_id} is not under {} pieces", scores.len())));
        }
        let min_score = scores.iter().copied().fold(f64::INFINITY, f64::min);

        // The normalizer, its sequences flattened: the precompiled map,
        // the collapse of spaces, the spacing of punctuation, the collapse
        // of whitespace and the strip, each in that order exactly when the
        // manifest says so, and nothing else.
        let mut steps: Vec<&Value> = Vec::new();
        let normalizer = &json["normalizer"];
        let mut pending = if normalizer.is_null() { Vec::new() } else { vec![normalizer] };
        while let Some(n) = pending.pop() {
            match n["type"].as_str() {
                Some("Sequence") => {
                    let children = n["normalizers"]
                        .as_array()
                        .ok_or_else(|| invalid(format!("{file}: normalizer sequence must contain an array")))?;
                    pending.extend(children.iter().rev());
                }
                Some(_) => steps.push(n),
                None => return Err(invalid(format!("{file}: normalizer must have a string type"))),
            }
        }
        let mut charsmap = None;
        let (mut collapse_spaces, mut collapse_whitespace, mut strip) = (false, false, false);
        let mut punctuation = String::new();
        // Each step's place in the order; a step may not come before one
        // already seen.
        let mut stage = 0;
        for (i, n) in steps.iter().enumerate() {
            let replace = |pattern: &str, regex: bool| {
                n["type"] == "Replace" && n["pattern"][if regex { "Regex" } else { "String" }] == pattern
            };
            let at = match n["type"].as_str() {
                Some("Precompiled") if i == 0 => {
                    let b64 = n["precompiled_charsmap"].as_str().unwrap_or_default();
                    charsmap = Some(Charsmap::parse(file, b64)?);
                    0
                }
                Some("Replace") if replace(" {2,}", true) && n["content"] == " " && !collapse_spaces => {
                    collapse_spaces = true;
                    1
                }
                Some("Replace") if punctuation_step(n).is_some_and(|c| !punctuation.contains(c)) => {
                    punctuation.push(punctuation_step(n).expect("matched"));
                    2
                }
                Some("Replace") if replace("\\s+", true) && n["content"] == " " && !collapse_whitespace => {
                    collapse_whitespace = true;
                    3
                }
                Some("Strip") if n["strip_left"] == true && n["strip_right"] == true && !strip => {
                    strip = true;
                    4
                }
                _ => {
                    return Err(invalid(format!(
                        "{file}: normalizer step {i}, {}: the core runs the precompiled map, the collapse of \
                         spaces, the spacing of punctuation, the collapse of whitespace and the strip, in that order",
                        n["type"]
                    )));
                }
            };
            if at < stage {
                return Err(invalid(format!("{file}: normalizer step {i} is out of the core's order")));
            }
            stage = at;
        }
        let space_punctuation = !punctuation.is_empty();
        if space_punctuation && punctuation.len() != PUNCTUATION.len() {
            return Err(invalid(format!(
                "{file}: normalizer: punctuation spaced is {punctuation:?}, not every ASCII punctuation character"
            )));
        }
        if charsmap.is_some() != m.precompiled_charsmap
            || collapse_spaces != m.collapse_spaces
            || space_punctuation != m.space_punctuation
            || collapse_whitespace != m.collapse_whitespace
            || strip != m.strip
        {
            return Err(disagree("normalizer"));
        }

        // The pre-tokenizer: Metaspace, prepending to every piece of text
        // when the manifest says so, and splitting at the metaspace.
        let pre = &json["pre_tokenizer"];
        if pre["type"] != "Metaspace" {
            return Err(invalid(format!("{file}: pre_tokenizer is {}, not Metaspace", pre["type"])));
        }
        if pre["replacement"] != m.metaspace.as_str() {
            return Err(disagree("pre_tokenizer.replacement"));
        }
        let legacy_prefix = boolean(file, pre, "add_prefix_space")?;
        let prepend = match pre.get("prepend_scheme") {
            Some(v) if v == "always" => true,
            Some(v) if v == "never" => false,
            Some(v) => return Err(invalid(format!("{file}: unsupported pre_tokenizer.prepend_scheme {v}"))),
            None => legacy_prefix.unwrap_or(true),
        };
        if legacy_prefix.is_some_and(|legacy| legacy != prepend) {
            return Err(invalid(format!("{file}: conflicting metaspace prefix settings")));
        }
        if prepend != m.add_prefix_space {
            return Err(disagree("pre_tokenizer.add_prefix_space"));
        }
        if !boolean(file, pre, "split")?.unwrap_or(true) != m.whole_text {
            return Err(disagree("pre_tokenizer.split"));
        }

        let trie =
            Trie::new(&pieces).map_err(|p| invalid(format!("{file}: model.vocab: piece {p:?} is listed twice")))?;
        let max_score = scores.iter().fold((min_score - UNK_PENALTY).abs(), |a, b| a.max(b.abs()));
        Ok(Unigram {
            max_score,
            pieces,
            trie,
            scores,
            unk_score: min_score - UNK_PENALTY,
            unk_id,
            charsmap,
            collapse_spaces,
            space_punctuation,
            collapse_whitespace,
            strip,
            whole_text: m.whole_text,
            metaspace,
            add_prefix_space: m.add_prefix_space,
        })
    }
}
/// The ASCII punctuation characters, which `space_punctuation` spaces.
const PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// The character a Replace step spaces, when it is one of PUNCTUATION
/// replaced by itself between two spaces.
fn punctuation_step(n: &Value) -> Option<char> {
    let p = n["pattern"]["String"].as_str()?;
    let mut chars = p.chars();
    let c = chars.next().filter(|c| c.is_ascii_punctuation() && chars.next().is_none())?;
    (n["content"] == format!(" {c} ")).then_some(c)
}

fn boolean(file: &str, object: &Value, field: &str) -> Result<Option<bool>> {
    object
        .get(field)
        .map(|value| value.as_bool().ok_or_else(|| invalid(format!("{file}: {field} must be a boolean"))))
        .transpose()
}
