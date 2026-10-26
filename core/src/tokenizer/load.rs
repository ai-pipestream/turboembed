//! Bundle loading, tokenizer consistency, and reference checks.

use super::*;

impl Tokenizer {
    /// Build the tokenizer the bundle names and check it against the
    /// reference ids. This is where `turbo_tokenizer_create` stops.
    pub fn load(bundle: &Bundle) -> Result<Tokenizer> {
        let tok = Self::from_bundle(bundle)?;
        tok.check_reference(bundle)?;
        Ok(tok)
    }

    /// Build the tokenizer the bundle names without checking it against
    /// the reference ids: for the bundle tool, which writes those ids.
    pub fn unchecked(bundle: &Bundle) -> Result<Tokenizer> {
        Self::from_bundle(bundle)
    }

    pub(super) fn from_bundle(bundle: &Bundle) -> Result<Tokenizer> {
        let m = &bundle.manifest;
        let t = &m.tokenizer;
        let file = t.file.as_str();
        let bytes = bundle.read_verified(file)?;
        // A SentencePiece model is a protobuf message, not JSON.
        let json: Value = match &t.sentencepiece {
            Some(_) => Value::Null,
            None => serde_json::from_slice(&bytes).map_err(|e| invalid(format!("{file}: {e}")))?,
        };
        let unk = t.special_tokens.iter().find(|s| s.role == SpecialRole::Unk).expect("validated");

        let kind = match (&t.wordpiece, &t.unigram, &t.bpe) {
            (Some(w), ..) => {
                let disagree = |what: &str| invalid(format!("{file}: {what} is not what manifest.json says"));
                let model = &json["model"];
                if model["type"] != "WordPiece" {
                    return Err(invalid(format!("{file}: model.type is {}, not WordPiece", model["type"])));
                }
                if model["unk_token"] != unk.content.as_str() {
                    return Err(disagree("model.unk_token"));
                }
                if model["continuing_subword_prefix"] != w.continuing_prefix.as_str() {
                    return Err(disagree("model.continuing_subword_prefix"));
                }
                if model["max_input_chars_per_word"] != w.max_chars_per_word {
                    return Err(disagree("model.max_input_chars_per_word"));
                }

                let n = &json["normalizer"];
                let lowercase = n["lowercase"].as_bool();
                let strip = if n["strip_accents"].is_null() { lowercase } else { n["strip_accents"].as_bool() };
                let norm = t.normalizer.as_ref().expect("validated");
                if n["type"] != "BertNormalizer"
                    || n["clean_text"].as_bool() != Some(norm.clean_text)
                    || n["handle_chinese_chars"].as_bool() != Some(norm.split_cjk)
                    || lowercase != Some(norm.lowercase)
                    || strip != Some(norm.strip_accents)
                {
                    return Err(disagree("normalizer"));
                }
                if json["pre_tokenizer"]["type"] != "BertPreTokenizer" {
                    return Err(invalid(format!(
                        "{file}: pre_tokenizer is {}, not BertPreTokenizer",
                        json["pre_tokenizer"]["type"]
                    )));
                }

                let raw = model["vocab"]
                    .as_object()
                    .ok_or_else(|| invalid(format!("{file}: model.vocab is not an object")))?;
                let mut vocab = Vocab::with_capacity_and_hasher(raw.len(), Default::default());
                let mut seen = vec![false; raw.len()];
                for (piece, id) in raw {
                    let id = id.as_u64().filter(|&i| i < raw.len() as u64);
                    let Some(id) = id else {
                        return Err(invalid(format!(
                            "{file}: model.vocab[{piece:?}] is not an id under {}",
                            raw.len()
                        )));
                    };
                    if std::mem::replace(&mut seen[id as usize], true) {
                        return Err(invalid(format!("{file}: model.vocab: id {id} is used twice")));
                    }
                    vocab.insert(piece.clone(), id as u32);
                }
                // The category table is made here, at load, rather than
                // by the first text that is not ASCII.
                category('\0', OTHER);
                let mut continuing = Vocab::default();
                if !w.continuing_prefix.is_empty() {
                    for (piece, &id) in &vocab {
                        if let Some(rest) = piece.strip_prefix(w.continuing_prefix.as_str()) {
                            continuing.insert(rest.to_owned(), id);
                        }
                    }
                }
                Kind::WordPiece {
                    longest: vocab.keys().map(String::len).max().unwrap_or(0),
                    longest_continuing: continuing.keys().map(String::len).max().unwrap_or(0),
                    continuing,
                    vocab,
                    ascii: ascii_map(norm),
                    normalizer: norm.clone(),
                    continuing_prefix: w.continuing_prefix.clone(),
                    max_chars_per_word: w.max_chars_per_word as usize,
                }
            }
            (None, Some(u), _) => Kind::Unigram(Unigram::parse(file, &json, u, unk.id)?),
            (None, None, Some(b)) => {
                // A normalized token would be matched in the NFC text,
                // which the core does not do.
                if let Some(s) = t.special_tokens.iter().find(|s| b.nfc && s.normalized) {
                    return Err(invalid(format!(
                        "tokenizer.special_tokens: {:?} is normalized, and the core matches no token after NFC",
                        s.content
                    )));
                }
                Kind::Bpe(Bpe::parse(file, &json, b, unk.id)?)
            }
            (None, None, None) => {
                let spec = t.sentencepiece.as_ref().expect("validated");
                let sp = SentencePiece::parse(&bytes).map_err(|e| invalid(format!("{file}: {}", e.message)))?;
                let want = match spec.algorithm {
                    crate::manifest::SentencePieceAlgorithm::Unigram => Algorithm::Unigram,
                    crate::manifest::SentencePieceAlgorithm::Bpe => Algorithm::Bpe,
                };
                if sp.algorithm() != want {
                    return Err(invalid(format!(
                        "{file}: the model is {:?}, manifest.json says {:?}",
                        sp.algorithm(),
                        spec.algorithm
                    )));
                }
                if sp.unk_id() != unk.id {
                    return Err(invalid(format!(
                        "{file}: the model's unknown piece is id {}, manifest.json says {}",
                        sp.unk_id(),
                        unk.id
                    )));
                }
                Kind::SentencePiece(Box::new(sp))
            }
        };
        let vocab_size = kind.vocab_size();
        let table = m
            .architecture
            .as_ref()
            .map(|a| ("architecture", a.vocab_size))
            .or(m.static_embedding.as_ref().map(|st| ("static_embedding", st.vocab_size)));
        if let Some((block, rows)) = table
            && vocab_size as u64 > rows as u64
        {
            return Err(invalid(format!("{file}: {vocab_size} vocabulary entries, {block}.vocab_size is {rows}")));
        }

        // Every special token is in the vocabulary under its id, and the
        // added tokens upstream matches in raw text are exactly these.
        for s in &t.special_tokens {
            if kind.id(&s.content) != Some(s.id) {
                return Err(invalid(format!("{file}: model.vocab has no {:?} with id {}", s.content, s.id)));
            }
        }
        let added = json["added_tokens"].as_array().map(Vec::as_slice).unwrap_or_default();
        let sentencepiece = matches!(kind, Kind::SentencePiece(_));
        let mut seen_added = std::collections::HashSet::with_capacity(added.len());
        for (i, a) in added.iter().enumerate() {
            let content = a["content"].as_str().unwrap_or_default();
            if !seen_added.insert(content) {
                return Err(invalid(format!("{file}: added_tokens[{i}] duplicates {content:?}")));
            }
            let Some(s) = t.special_tokens.iter().find(|s| s.content == content) else {
                return Err(invalid(format!(
                    "{file}: added_tokens[{i}] {content:?} is not a special token in manifest.json"
                )));
            };
            if a["special"] != true
                || a["id"] != s.id
                || a["normalized"] != s.normalized
                || a["lstrip"] != s.lstrip
                || a["rstrip"] != s.rstrip
                || a["single_word"] != s.single_word
            {
                return Err(invalid(format!(
                    "{file}: added_tokens[{i}] {content:?} is matched differently from what manifest.json says"
                )));
            }
        }
        if !sentencepiece && added.len() != t.special_tokens.len() {
            return Err(invalid(format!(
                "{file}: {} added_tokens, manifest.json has {} special tokens",
                added.len(),
                t.special_tokens.len()
            )));
        }

        let id_of = |content: &str| t.special_tokens.iter().find(|s| s.content == content).map(|s| s.id as i32);
        let role = |r: SpecialRole| t.special_tokens.iter().find(|s| s.role == r).map_or(-1, |s| s.id as i32);
        // The library matches no special token in the text: a control
        // piece written there is its characters.
        let mut specials: Vec<Special> = t
            .special_tokens
            .iter()
            .filter(|s| !s.normalized && !sentencepiece)
            .map(|s| Special { content: s.content.clone(), id: s.id as i32, lstrip: s.lstrip, single_word: false })
            .collect();
        specials.sort_by_key(|a| std::cmp::Reverse(a.content.len()));
        // Whitespace around a normalized special token is dropped by the
        // pre-tokenizer either way, so lstrip and rstrip change no id.
        let mut normalized_specials: Vec<Special> = match &kind {
            Kind::WordPiece { normalizer, ascii, .. } => t
                .special_tokens
                .iter()
                .filter(|s| s.normalized)
                .map(|s| Special {
                    content: normalized(normalizer, ascii, &s.content),
                    id: s.id as i32,
                    lstrip: s.lstrip,
                    single_word: s.single_word,
                })
                .filter(|s| !s.content.is_empty())
                .collect(),
            // No normalizer: a normalized token is its content, matched
            // in each piece the raw tokens leave, after them as upstream.
            Kind::Bpe(_) => t
                .special_tokens
                .iter()
                .filter(|s| s.normalized)
                .map(|s| Special { content: s.content.clone(), id: s.id as i32, lstrip: s.lstrip, single_word: false })
                .collect(),
            Kind::Unigram(u) => t
                .special_tokens
                .iter()
                .filter(|s| s.normalized)
                .map(|s| Special {
                    content: u.normalize(&s.content),
                    id: s.id as i32,
                    lstrip: s.lstrip,
                    single_word: false,
                })
                .collect(),
            Kind::SentencePiece(_) => Vec::new(),
        };
        normalized_specials.sort_by_key(|a| std::cmp::Reverse(a.content.len()));
        let mut special_starts = [false; 256];
        for s in &specials {
            if let Some(&b) = s.content.as_bytes().first() {
                special_starts[b as usize] = true;
            }
        }
        let mut normalized_starts = [false; 256];
        for s in &normalized_specials {
            if let Some(&b) = s.content.as_bytes().first() {
                normalized_starts[b as usize] = true;
            }
        }
        let stat = m.static_embedding.as_ref().map(|_| StaticRules { median_chars: kind.median_chars() });
        let decoder = if sentencepiece { Decoder::SentencePiece } else { Decoder::parse(&json["decoder"]) };
        let mut pieces = vec![Box::<str>::from(""); vocab_size as usize];
        kind.each_piece(|piece, id| pieces[id as usize] = piece.into());
        let e = m.encoder();
        let rerank = m.rerank.clone().or_else(|| {
            let a = m.architecture.as_ref()?;
            if t.template.len() != 3 || t.template[1] != "$TEXT" {
                return None;
            }
            let pair_template = match a.family {
                crate::manifest::Family::Bert if a.token_types >= 2 => crate::manifest::PairTemplate::Bert,
                crate::manifest::Family::Roberta => crate::manifest::PairTemplate::Roberta,
                crate::manifest::Family::Bert | crate::manifest::Family::DebertaV2 => return None,
            };
            let specials = pair_template.specials();
            if e.max_seq <= specials + 1 {
                return None;
            }
            Some(crate::manifest::Rerank {
                max_seq: e.max_seq,
                max_batch: e.max_batch,
                query_tokens: ((e.max_seq - specials) / 2).max(1),
                pair_template,
            })
        });
        Ok(Tokenizer {
            rerank,
            kind,
            specials,
            normalized_specials,
            special_starts,
            normalized_starts,
            template: t.template.iter().map(|s| if s == "$TEXT" { None } else { id_of(s) }).collect(),
            truncation: t.truncation,
            max_seq: m.static_embedding.as_ref().map_or(e.max_seq, |st| st.max_length),
            prefix_query: e.prefix_query.clone(),
            prefix_document: e.prefix_document.clone(),
            pad_id: role(SpecialRole::Pad),
            bos_id: role(SpecialRole::Bos),
            eos_id: role(SpecialRole::Eos),
            unk_id: role(SpecialRole::Unk),
            stat,
            decoder,
            pieces,
            special_ids: t.special_tokens.iter().map(|s| s.id as i32).collect(),
            sha256: crate::bundle::sha256_hex(&bytes),
            manifest_sha256: bundle.manifest_sha256.clone(),
        })
    }

    /// Loader rule 5: encode every reference case and compare the ids
    /// exactly with the reference file's.
    pub(super) fn check_reference(&self, bundle: &Bundle) -> Result<()> {
        if bundle.manifest.qa.is_some() {
            return crate::qa::check_reference(bundle, self);
        }
        let m = &bundle.manifest;
        if m.rerank.is_some() {
            return self.check_pair_reference(bundle);
        }
        let file = m.reference.file.as_str();
        let bytes = bundle.read_verified(file)?;
        let st = safetensors::File::parse(file, &bytes)?;
        let cases = &m.reference.cases;
        let n = cases.len() as u64;
        let ids = st.get("ids", Dtype::I32, 2)?;
        let lengths = st.get("lengths", Dtype::I32, 1)?;
        let emb = st.get("embeddings", Dtype::F32, 2)?;
        let dim = m.encoder().dim as u64;
        if ids.shape[0] != n || lengths.shape[0] != n || emb.shape != [n, dim] {
            return Err(invalid(format!(
                "{file}: ids {:?}, lengths {:?} and embeddings {:?} are not [{n}, L], [{n}] and [{n}, {dim}]",
                ids.shape, lengths.shape, emb.shape
            )));
        }
        let width = ids.shape[1] as usize;
        let ids = ids.i32s();
        let lengths = lengths.i32s();

        let mut truncated = false;
        let opts = Encode {
            add_special_tokens: true,
            truncation: self.truncation,
            max_tokens: self.max_seq,
            prompt: PromptRole::None,
        };
        for (i, case) in cases.iter().enumerate() {
            let at = |why: String| invalid(format!("reference case {i} ({}): {why}", preview(&case.text)));
            let got = self.encode(&case.text, Encode { prompt: case.prompt_role, ..opts })?;
            truncated |= self.count(&case.text, case.prompt_role) > self.max_seq as usize;
            let row = &ids[i * width..(i + 1) * width];
            let len = lengths[i];
            if len < 0 || len as usize > width {
                return Err(at(format!("length {len} is outside the row width {width}")));
            }
            let want = &row[..len as usize];
            if let Some(p) = (0..got.len().min(want.len())).find(|&p| got[p] != want[p]) {
                return Err(at(format!("position {p}: the core gives {}, the reference has {}", got[p], want[p])));
            }
            if got.len() != want.len() {
                return Err(at(format!(
                    "position {}: the core gives {} tokens, the reference has {}",
                    got.len().min(want.len()),
                    got.len(),
                    want.len()
                )));
            }
            if let Some(p) = (want.len()..width).find(|&p| row[p] != self.pad_id) {
                return Err(at(format!("position {p}: padding is {}, not pad id {}", row[p], self.pad_id)));
            }
        }
        if !truncated {
            return Err(invalid(format!(
                "reference.cases: none is longer than embed.max_seq {}, so truncation is not checked",
                self.max_seq
            )));
        }
        Ok(())
    }
}
