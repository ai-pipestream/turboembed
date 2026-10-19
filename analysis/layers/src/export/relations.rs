use super::*;

impl LayerCodec for Matches {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ranges(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.tokens)?;
        put_tagset(&mut w, &self.labels)?;
        put_tags(&mut w, &self.label)?;
        put_u32s(&mut w, &self.rule)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let tokens = get_ranges(&mut r, cx)?;
        let labels = get_tagset(&mut r)?;
        let label = get_tags(&mut r, &labels)?;
        let rule = get_u32s(&mut r)?;
        want_len(tokens.len(), spans.len(), "token ranges")?;
        want_len(label.len(), spans.len(), "labels")?;
        want_len(rule.len(), spans.len(), "rules")?;
        done(&r, "matches")?;
        Ok(Matches { spans, tokens, label, labels, rule })
    }
}

impl LayerCodec for Levels {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ref(&mut w, &cx.resolve(&Key::<Chunks>::DEFAULT))?;
        put_ref(&mut w, &cx.resolve(&Key::<Structure>::DEFAULT))?;
        w.usize(self.unit.len())?;
        for u in &self.unit {
            match u {
                Unit::Chunk(i) => {
                    w.u8(0);
                    w.u32(i.get() as u32);
                }
                Unit::Section(i) => {
                    w.u8(1);
                    w.u32(i.get() as u32);
                }
                Unit::Document => w.u8(2),
            }
        }
        put_parents(&mut w, &self.parent)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let chunks_ref = get_ref(&mut r)?;
        cx.expect_target::<Chunks>(&chunks_ref)?;
        let structure_ref = get_ref(&mut r)?;
        cx.expect_target::<Structure>(&structure_ref)?;
        let mut unit = Vec::new();
        for _ in 0..r.usize()? {
            unit.push(match r.u8()? {
                0 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: chunks_ref.clone(), index })?;
                    Unit::Chunk(Ix::new(index))
                }
                1 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: structure_ref.clone(), index })?;
                    Unit::Section(Ix::new(index))
                }
                2 => Unit::Document,
                tag => return Err(malformed(format!("a unit tag of {tag}"))),
            });
        }
        let parent = get_parents(&mut r)?;
        want_len(unit.len(), spans.len(), "units")?;
        want_len(parent.len(), spans.len(), "parents")?;
        done(&r, "levels")?;
        Ok(Levels { spans, unit, parent })
    }
}

impl LayerCodec for Containment {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_strs(&mut w, &self.ancestor_id)?;
        put_strs(&mut w, &self.ancestor_name)?;
        put_strs(&mut w, &self.ancestor_kind)?;
        put_u32s(&mut w, &self.chain_end)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let ancestor_id = get_strs(&mut r)?;
        let ancestor_name = get_strs(&mut r)?;
        let ancestor_kind = get_strs(&mut r)?;
        let chain_end = get_u32s(&mut r)?;
        want_len(ancestor_name.len(), ancestor_id.len(), "ancestor names")?;
        want_len(ancestor_kind.len(), ancestor_id.len(), "ancestor kinds")?;
        want_len(chain_end.len(), spans.len(), "chain ends")?;
        if chain_end.windows(2).any(|w| w[0] > w[1])
            || chain_end.last().copied().unwrap_or(0) as usize != ancestor_id.len()
        {
            return Err(malformed("chain ends do not partition the ancestors"));
        }
        done(&r, "containment")?;
        Ok(Containment { spans, ancestor_id, ancestor_name, ancestor_kind, chain_end })
    }
}

impl LayerCodec for Phrases {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ranges(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.tokens)?;
        put_tagset(&mut w, &self.kinds)?;
        put_tags(&mut w, &self.kind)?;
        w.usize(self.head.len())?;
        for h in &self.head {
            w.u32(h.get() as u32);
        }
        put_parents(&mut w, &self.parent)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let tokens = get_ranges(&mut r, cx)?;
        let kinds = get_tagset(&mut r)?;
        let kind = get_tags(&mut r, &kinds)?;
        let mut head = Vec::new();
        for (i, _) in (0..r.usize()?).enumerate() {
            let h = r.u32()?;
            let range = tokens.get(i).ok_or_else(|| malformed("a head without a token range"))?;
            if (h as usize) < range.start.get() || (h as usize) >= range.end.get() {
                return Err(malformed(format!("phrase {i} head {h} outside its token range")));
            }
            head.push(Ix::new(h));
        }
        let parent = get_parents(&mut r)?;
        want_len(tokens.len(), spans.len(), "token ranges")?;
        want_len(kind.len(), spans.len(), "kinds")?;
        want_len(head.len(), spans.len(), "heads")?;
        want_len(parent.len(), spans.len(), "parents")?;
        done(&r, "phrases")?;
        Ok(Phrases { spans, tokens, kind, kinds, head, parent })
    }
}
