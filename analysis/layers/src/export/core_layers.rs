use super::*;

macro_rules! spans_only {
    ($t:ty, $what:literal) => {
        impl LayerCodec for $t {
            fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
                let mut w = Writer::new();
                put_spans(&mut w, &self.spans, cx)?;
                Ok(w.finish())
            }
            fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
                let mut r = Reader::new(data);
                let spans = get_spans(&mut r, cx)?;
                done(&r, $what)?;
                Ok(Self { spans })
            }
            fn slice_body(data: &[u8], cx: &SliceCx) -> Result<SlicedBody, ExportError> {
                slice_spans_payload(data, cx).map(Some)
            }
        }
    };
}

spans_only!(Sentences, "sentences");
spans_only!(TokenVectorWindows, "token-vector windows");

impl LayerCodec for Tokens {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_opt_ix_col(&mut w, cx, &Key::<Sentences>::DEFAULT, &self.sentence)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let sentence = get_opt_ix_col(&mut r, cx)?;
        if let Some(s) = &sentence {
            want_len(s.len(), spans.len(), "sentence indexes")?;
        }
        done(&r, "tokens")?;
        Ok(Tokens { spans, sentence })
    }

    /// Clips the token spans and re-points the sentence column
    /// through the sentences remap; a surviving token whose
    /// sentence fell outside the slice refuses, since its column
    /// has no per-item empty case to fall back to.
    fn slice_body(data: &[u8], cx: &SliceCx) -> Result<SlicedBody, ExportError> {
        use turbo_analysis::{Span, Spans, ViewId};
        let mut r = Reader::new(data);
        let (instance, items) = get_spans_raw(&mut r)?;
        let range = cx.view_range(&instance)?;
        let mut spans = Spans::<Tokens>::new(ViewId::SOURCE);
        for fragments in &items {
            match &fragments[..] {
                [one] => {
                    spans.push(*one);
                }
                many => {
                    spans
                        .try_push_fragments(many)
                        .map_err(|e| malformed(format!("token fragments being sliced: {e}")))?;
                }
            }
        }
        let (sliced, remap) = spans.sliced(range);
        let sentence = match r.u8()? {
            0 => None,
            1 => {
                let target = get_ref(&mut r)?;
                let mut kept: Vec<u32> = Vec::new();
                for old in 0..r.usize()? {
                    let index = r.u32()?;
                    if remap.get(old).copied().flatten().is_some() {
                        let new = cx.remapped(&target, index)?.ok_or_else(|| ExportError::InvalidLayer {
                            layer: "opennlp:tokens".to_string(),
                            detail: "a token in the slice names a sentence outside it".to_string(),
                        })?;
                        kept.push(new);
                    }
                }
                Some((target, kept))
            }
            tag => {
                return Err(malformed(format!("an optional column tag of {tag}")));
            }
        };
        if !r.is_done() {
            return Err(malformed("trailing bytes in a tokens payload being sliced"));
        }
        let clipped: Vec<Vec<Span>> = sliced.iter().map(|(ix, _)| sliced.fragments(ix).collect()).collect();
        let mut w = Writer::new();
        put_spans_raw(&mut w, &instance, &clipped)?;
        match sentence {
            None => w.u8(0),
            Some((target, kept)) => {
                w.u8(1);
                put_ref(&mut w, &target)?;
                w.usize(kept.len())?;
                for index in kept {
                    w.u32(index);
                }
            }
        }
        Ok(Some((w.finish(), remap)))
    }
}

impl LayerCodec for Subwords {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.ids.len())?;
        for id in &self.ids {
            w.u32(*id as u32);
        }
        put_model(&mut w, &self.model)?;
        put_opt_ix_col(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.token)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut ids = Vec::new();
        for _ in 0..r.usize()? {
            ids.push(r.u32()? as i32);
        }
        want_len(ids.len(), spans.len(), "subword ids")?;
        let model = get_model(&mut r)?;
        let token = get_opt_ix_col(&mut r, cx)?;
        if let Some(t) = &token {
            want_len(t.len(), spans.len(), "token indexes")?;
        }
        done(&r, "subwords")?;
        Ok(Subwords { spans, ids, model, token })
    }
}

macro_rules! tag_column {
    ($t:ty, $what:literal) => {
        impl LayerCodec for $t {
            fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
                let mut w = Writer::new();
                put_tagset(&mut w, &self.tagset)?;
                put_tags(&mut w, &self.tags)?;
                put_opt_f64s(&mut w, &self.prob)?;
                Ok(w.finish())
            }
            fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
                let mut r = Reader::new(data);
                let tagset = get_tagset(&mut r)?;
                let tags = get_tags(&mut r, &tagset)?;
                let prob = get_opt_f64s(&mut r)?;
                if let Some(p) = &prob {
                    want_len(p.len(), tags.len(), "probabilities")?;
                }
                done(&r, $what)?;
                Ok(Self { tagset, tags, prob })
            }
        }
    };
}

tag_column!(PosTags, "pos tags");
tag_column!(ChunkTags, "chunk tags");
tag_column!(WordPosTags, "word pos tags");

macro_rules! strs_column {
    ($t:ty, $field:ident, $what:literal) => {
        impl LayerCodec for $t {
            fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
                let mut w = Writer::new();
                put_strs(&mut w, &self.$field)?;
                Ok(w.finish())
            }
            fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
                let mut r = Reader::new(data);
                let $field = get_strs(&mut r)?;
                done(&r, $what)?;
                Ok(Self { $field })
            }
        }
    };
}

strs_column!(Lemmas, lemma, "lemmas");
strs_column!(Stems, stem, "stems");
strs_column!(WordLemmas, lemma, "word lemmas");

impl LayerCodec for Terms {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_strs(&mut w, &self.surface)?;
        put_strs(&mut w, &self.folded)?;
        put_opt_strs(&mut w, &self.stem)?;
        put_opt_strs(&mut w, &self.lemma)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let surface = get_strs(&mut r)?;
        let folded = get_strs(&mut r)?;
        let stem = get_opt_strs(&mut r)?;
        let lemma = get_opt_strs(&mut r)?;
        want_len(folded.len(), surface.len(), "folded forms")?;
        if let Some(s) = &stem {
            want_len(s.len(), surface.len(), "stems")?;
        }
        if let Some(l) = &lemma {
            want_len(l.len(), surface.len(), "lemmas")?;
        }
        done(&r, "terms")?;
        Ok(Terms { surface, folded, stem, lemma })
    }
}

impl<A: WordOrTokenAnchor> LayerCodec for Dependencies<A> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_tagset(&mut w, &self.relations)?;
        w.usize(self.head.len())?;
        for h in &self.head {
            match h {
                None => w.u8(0),
                Some(h) => {
                    w.u8(1);
                    w.u32(h.get() as u32);
                }
            }
        }
        put_tags(&mut w, &self.relation)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let relations = get_tagset(&mut r)?;
        let n = r.usize()?;
        let mut head = Vec::new();
        for i in 0..n {
            head.push(match r.u8()? {
                0 => None,
                1 => {
                    let h = r.u32()?;
                    if h as usize >= n {
                        return Err(malformed(format!("dependency head {h} at {i} is out of bounds ({n})")));
                    }
                    if h as usize == i {
                        return Err(malformed(format!("dependency item {i} is its own head")));
                    }
                    Some(Ix::new(h))
                }
                tag => return Err(malformed(format!("an optional head tag of {tag}"))),
            });
        }
        let relation = get_tags(&mut r, &relations)?;
        want_len(relation.len(), n, "relations")?;
        done(&r, "dependencies")?;
        let layer = Dependencies { head, relation, relations };
        // Keep direct codec calls checked as well as Document insertion.
        layer.check().map_err(malformed)?;
        layer.check_forest().map_err(malformed)?;
        Ok(layer)
    }
}

impl LayerCodec for Entities {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ranges(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.tokens)?;
        put_tagset(&mut w, &self.kinds)?;
        put_tags(&mut w, &self.kind)?;
        put_f64s(&mut w, &self.score)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let tokens = get_ranges(&mut r, cx)?;
        let kinds = get_tagset(&mut r)?;
        let kind = get_tags(&mut r, &kinds)?;
        let score = get_f64s(&mut r)?;
        want_len(tokens.len(), spans.len(), "token ranges")?;
        want_len(kind.len(), spans.len(), "kinds")?;
        want_len(score.len(), spans.len(), "scores")?;
        done(&r, "entities")?;
        Ok(Entities { spans, tokens, kind, kinds, score })
    }
}

impl LayerCodec for PhraseChunks {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ranges(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.tokens)?;
        put_tagset(&mut w, &self.kinds)?;
        put_tags(&mut w, &self.kind)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let tokens = get_ranges(&mut r, cx)?;
        let kinds = get_tagset(&mut r)?;
        let kind = get_tags(&mut r, &kinds)?;
        want_len(tokens.len(), spans.len(), "token ranges")?;
        want_len(kind.len(), spans.len(), "kinds")?;
        done(&r, "phrase chunks")?;
        Ok(PhraseChunks { spans, tokens, kind, kinds })
    }
}
