use super::*;

impl<From: Layer, To: Layer> LayerCodec for Edges<From, To> {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_layer_ref(&mut w, &cx.resolve(&self.from_key))?;
        put_layer_ref(&mut w, &cx.resolve(&self.to_key))?;
        put_tagset(&mut w, &self.kinds)?;
        w.usize(self.len())?;
        for (_, edge) in self.iter() {
            w.u32(edge.from.get() as u32);
            w.u32(edge.to.get() as u32);
            w.u32(u32::from(edge.kind.0));
            w.f64(edge.score);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let from = get_layer_ref(&mut r)?;
        let to = get_layer_ref(&mut r)?;
        cx.want_keyed_target::<From>(&from)?;
        cx.want_keyed_target::<To>(&to)?;
        cx.layer_len(&from)?;
        cx.layer_len(&to)?;
        let kinds = get_tagset(&mut r)?;
        let mut rows = Vec::new();
        for _ in 0..r.usize()? {
            let (a, b, kind, score) = (r.u32()?, r.u32()?, r.u32()?, r.f64()?);
            cx.require_node(&NodeRef { layer: from.clone(), index: a })?;
            cx.require_node(&NodeRef { layer: to.clone(), index: b })?;
            let kind = Tag(u16::try_from(kind).map_err(|_| malformed("an edge kind outside a u16"))?);
            // Export writes rows in (from, to, kind) order, each once; a
            // payload in any other order is not one export wrote, and
            // sorting it here would silently rewrite it.
            if rows
                .last()
                .is_some_and(|e: &Edge<From, To>| (e.from.get(), e.to.get(), e.kind) >= (a as usize, b as usize, kind))
            {
                return Err(malformed("edges out of (from, to, kind) order or repeated"));
            }
            rows.push(Edge::new(Ix::new(a), Ix::new(b), kind, score));
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in an edges payload"));
        }
        let (from_key, to_key) = (cx.key::<From>(&from.instance.clone()), cx.key::<To>(&to.instance.clone()));
        Ok(Edges::between(from_key, to_key, kinds, rows))
    }

    /// Keeps the edges whose endpoints both survived slicing, with
    /// endpoints re-pointed through the earlier layers' remaps;
    /// kinds and scores are copied whole. Row order is preserved,
    /// so the sliced payload is still in (from, to, kind) order
    /// once endpoints are remapped (remaps are order-preserving).
    fn slice_body(data: &[u8], cx: &SliceCx) -> Result<SlicedBody, ExportError> {
        let mut r = Reader::new(data);
        let from = get_layer_ref(&mut r)?;
        let to = get_layer_ref(&mut r)?;
        let kinds = get_tagset(&mut r)?;
        let mut rows: Vec<(u32, u32, u32, f64)> = Vec::new();
        let mut remap: Vec<Option<u32>> = Vec::new();
        for _ in 0..r.usize()? {
            let (a, b, kind, score) = (r.u32()?, r.u32()?, r.u32()?, r.f64()?);
            match (cx.remapped(&from, a)?, cx.remapped(&to, b)?) {
                (Some(a2), Some(b2)) => {
                    remap.push(Some(rows.len() as u32));
                    rows.push((a2, b2, kind, score));
                }
                _ => remap.push(None),
            }
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in an edges payload being sliced"));
        }
        let mut w = Writer::new();
        put_layer_ref(&mut w, &from)?;
        put_layer_ref(&mut w, &to)?;
        put_tagset(&mut w, &kinds)?;
        w.usize(rows.len())?;
        for (a, b, kind, score) in rows {
            w.u32(a);
            w.u32(b);
            w.u32(kind);
            w.f64(score);
        }
        Ok(Some((w.finish(), remap)))
    }
}

impl<L: Layer> LayerCodec for Groups<L> {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_layer_ref(&mut w, &cx.resolve(&self.of_key))?;
        w.usize(self.len())?;
        for (_, members) in self.iter() {
            w.usize(members.len())?;
            for member in members {
                w.u32(member.get() as u32);
            }
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let of = get_layer_ref(&mut r)?;
        cx.want_keyed_target::<L>(&of)?;
        cx.layer_len(&of)?;
        let mut groups: Vec<Vec<Ix<L>>> = Vec::new();
        for _ in 0..r.usize()? {
            let mut members = Vec::new();
            for _ in 0..r.usize()? {
                let index = r.u32()?;
                cx.require_node(&NodeRef { layer: of.clone(), index })?;
                // Export writes each group's members in strictly
                // increasing order; deduplicating or sorting a payload
                // that is not would silently change the group.
                if members.last().is_some_and(|m: &Ix<L>| m.get() >= index as usize) {
                    return Err(malformed("group members out of order or repeated"));
                }
                members.push(Ix::new(index));
            }
            groups.push(members);
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a groups payload"));
        }
        let of_key = cx.key::<L>(&of.instance.clone());
        Ok(Groups::of(of_key, groups))
    }

    /// Keeps the members that survived slicing, re-pointed through
    /// their layer's remap, and drops groups left empty. Members
    /// stay in increasing order (remaps are order-preserving).
    fn slice_body(data: &[u8], cx: &SliceCx) -> Result<SlicedBody, ExportError> {
        let mut r = Reader::new(data);
        let of = get_layer_ref(&mut r)?;
        let mut groups: Vec<Vec<u32>> = Vec::new();
        let mut remap: Vec<Option<u32>> = Vec::new();
        for _ in 0..r.usize()? {
            let mut members: Vec<u32> = Vec::new();
            for _ in 0..r.usize()? {
                let index = r.u32()?;
                if let Some(new) = cx.remapped(&of, index)? {
                    members.push(new);
                }
            }
            if members.is_empty() {
                remap.push(None);
            } else {
                remap.push(Some(groups.len() as u32));
                groups.push(members);
            }
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a groups payload being sliced"));
        }
        let mut w = Writer::new();
        put_layer_ref(&mut w, &of)?;
        w.usize(groups.len())?;
        for members in &groups {
            w.usize(members.len())?;
            for member in members {
                w.u32(*member);
            }
        }
        Ok(Some((w.finish(), remap)))
    }
}
