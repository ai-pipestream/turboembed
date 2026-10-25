use super::*;

const BLOCKS: [Block; 16] = [
    Block::Section,
    Block::Heading,
    Block::Paragraph,
    Block::List,
    Block::ListItem,
    Block::Table,
    Block::TableRow,
    Block::TableCell,
    Block::Code,
    Block::Quote,
    Block::Caption,
    Block::Footnote,
    Block::PageHeader,
    Block::PageFooter,
    Block::LooseList,
    Block::Html,
];

const CUTS: [(&str, Cut); 7] = [
    ("document", Cut::DocumentStart),
    ("sentence", Cut::Sentence),
    ("section", Cut::Section),
    ("heading", Cut::Heading),
    ("paragraph", Cut::Paragraph),
    ("topic", Cut::TopicShift),
    ("clause", Cut::Clause),
];

pub(super) fn put_cut(w: &mut Writer, cut: Cut) -> Result<(), ExportError> {
    match cut {
        Cut::Inside(block) => {
            w.u8(8);
            w.u8(BLOCKS.iter().position(|b| *b == block).expect("a block in the table") as u8);
        }
        Cut::Overlap(block) => {
            w.u8(9);
            w.u8(BLOCKS.iter().position(|b| *b == block).expect("a block in the table") as u8);
        }
        Cut::Item => w.u8(7),
        other => {
            let at = CUTS.iter().position(|(_, c)| *c == other).expect("a cut in the table");
            w.u8(at as u8);
        }
    }
    Ok(())
}

pub(super) fn get_cut(r: &mut Reader<'_>) -> Result<Cut, ExportError> {
    let block = |r: &mut Reader<'_>| -> Result<Block, ExportError> {
        BLOCKS.get(r.u8()? as usize).copied().ok_or_else(|| malformed("a block outside the block table"))
    };
    Ok(match r.u8()? {
        7 => Cut::Item,
        8 => Cut::Inside(block(r)?),
        9 => Cut::Overlap(block(r)?),
        tag => CUTS.get(tag as usize).map(|(_, c)| *c).ok_or_else(|| malformed(format!("a cut tag of {tag}")))?,
    })
}

impl LayerCodec for Boundaries {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.cost.len())?;
        for c in &self.cost {
            w.u32(*c as u32);
        }
        for reason in &self.reason {
            put_cut(&mut w, *reason)?;
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let n = r.usize()?;
        let mut cost = Vec::new();
        for _ in 0..n {
            cost.push(r.u32()? as i32);
        }
        let mut reason = Vec::new();
        for _ in 0..n {
            reason.push(get_cut(&mut r)?);
        }
        done(&r, "boundaries")?;
        Ok(Boundaries { cost, reason })
    }
}

impl LayerCodec for Structure {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.kind.len())?;
        for k in &self.kind {
            w.u8(BLOCKS.iter().position(|b| b == k).expect("a block in the table") as u8);
        }
        put_parents(&mut w, &self.parent)?;
        w.usize(self.depth.len())?;
        for d in &self.depth {
            w.u8(*d);
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut kind = Vec::new();
        for _ in 0..r.usize()? {
            kind.push(*BLOCKS.get(r.u8()? as usize).ok_or_else(|| malformed("a block outside the block table"))?);
        }
        let parent = get_parents(&mut r)?;
        let mut depth = Vec::new();
        for _ in 0..r.usize()? {
            depth.push(r.u8()?);
        }
        want_len(kind.len(), spans.len(), "kinds")?;
        want_len(parent.len(), spans.len(), "parents")?;
        want_len(depth.len(), spans.len(), "depths")?;
        done(&r, "structure")?;
        Ok(Structure { spans, kind, parent, depth })
    }
}

impl LayerCodec for Chunks {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.level.len())?;
        for l in &self.level {
            w.u8(*l);
        }
        put_parents(&mut w, &self.parent)?;
        put_ranges(&mut w, cx, &Key::<Sentences>::DEFAULT, &self.sentences)?;
        for cut in &self.cut {
            put_cut(&mut w, *cut)?;
        }
        w.usize(self.cost.len())?;
        for c in &self.cost {
            w.u64(*c as u64);
        }
        w.usize(self.model_tokens.len())?;
        for t in &self.model_tokens {
            w.u32(*t);
        }
        w.usize(self.over_budget.len())?;
        for b in &self.over_budget {
            w.u8(u8::from(*b));
        }
        w.u32(self.prompt_overhead);
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = spans.len();
        let mut level = Vec::new();
        for _ in 0..r.usize()? {
            level.push(r.u8()?);
        }
        let parent = get_parents(&mut r)?;
        let sentences = get_ranges(&mut r, cx)?;
        let mut cut = Vec::new();
        for _ in 0..n {
            cut.push(get_cut(&mut r)?);
        }
        let mut cost = Vec::new();
        for _ in 0..r.usize()? {
            cost.push(r.u64()? as i64);
        }
        let mut model_tokens = Vec::new();
        for _ in 0..r.usize()? {
            model_tokens.push(r.u32()?);
        }
        let mut over_budget = Vec::new();
        for _ in 0..r.usize()? {
            over_budget.push(match r.u8()? {
                0 => false,
                1 => true,
                tag => return Err(malformed(format!("a boolean tag of {tag}"))),
            });
        }
        want_len(level.len(), n, "levels")?;
        want_len(parent.len(), n, "parents")?;
        want_len(sentences.len(), n, "sentence ranges")?;
        want_len(cost.len(), n, "costs")?;
        want_len(model_tokens.len(), n, "model token counts")?;
        let prompt_overhead = r.u32()?;
        want_len(over_budget.len(), n, "over-budget flags")?;
        done(&r, "chunks")?;
        Ok(Chunks { spans, level, parent, sentences, cut, cost, model_tokens, over_budget, prompt_overhead })
    }
}
