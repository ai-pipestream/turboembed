use super::PrebuiltError;
use crate::lattice::{DictionaryLimits, MecabDictionary};
use turbo_analysis_layers::Tag;

fn require(ok: bool, detail: &'static str) -> Result<(), PrebuiltError> {
    if ok { Ok(()) } else { Err(PrebuiltError::Invalid(detail)) }
}

pub(super) fn dictionary(d: &MecabDictionary, limits: DictionaryLimits) -> Result<(), PrebuiltError> {
    let (rows, cols) = (d.matrix.rows() as usize, d.matrix.cols() as usize);
    let cells = rows.checked_mul(cols).ok_or(PrebuiltError::Invalid("matrix dimensions"))?;
    require(
        rows > 0
            && cols > 0
            && rows <= limits.max_entries
            && cols <= limits.max_entries
            && cells <= limits.max_matrix_cells
            && cells <= i32::MAX as usize
            && cells == d.matrix.costs.len(),
        "matrix size",
    )?;
    let lex = d.lexicon_len as usize;
    require(
        lex > 0 && lex <= limits.max_entries && lex <= d.entries.len() && d.entries.len() - lex <= limits.max_entries,
        "entry limit",
    )?;
    require(
        d.groups.len() > 1
            && d.groups[0] == 0
            && d.groups.last() == Some(&d.lexicon_len)
            && d.groups.windows(2).all(|p| p[0] < p[1]),
        "surface groups",
    )?;
    let f = &d.features;
    require(
        !f.first.is_empty()
            && f.first[0] == 0
            && f.first.last().copied().map(u64::from) == Some(f.ends.len() as u64)
            && f.first.windows(2).all(|p| p[0] <= p[1]),
        "feature records",
    )?;
    let mut last = 0;
    for &end in f.ends.iter() {
        require(
            end >= last && (end as usize) <= f.text.len() && f.text.is_char_boundary(end as usize),
            "feature UTF-8 boundary",
        )?;
        last = end;
    }
    require(last as usize == f.text.len(), "feature text length")?;
    require(!d.pos.is_empty() && d.pos.len() <= 65536 && d.tags.len() == d.entries.len(), "tag table")?;
    for (i, e) in d.entries.iter().enumerate() {
        require(
            (e.left as usize) < cols && (e.right as usize) < rows && (e.features as usize) < f.first.len() - 1,
            "entry context or feature index",
        )?;
        require((d.tags[i] as usize) < d.pos.len(), "tag index")?;
        require(
            d.pos.label(Tag(d.tags[i])) == f.columns(e.features).next().unwrap_or(""),
            "entry tag disagrees with features",
        )?;
    }
    for i in 0..d.pos.len() {
        require(d.pos.tag(d.pos.label(Tag(i as u16))) == Some(Tag(i as u16)), "duplicate tag labels")?;
    }
    let c = &d.chars;
    require(
        !c.categories.is_empty() && c.categories.len() <= 17 && (c.default.0 as usize) < c.categories.len(),
        "categories",
    )?;
    require(c.categories[c.default.0 as usize].name == "DEFAULT", "DEFAULT category")?;
    for (i, cat) in c.categories.iter().enumerate() {
        require(!cat.name.is_empty() && !c.categories[..i].iter().any(|p| p.name == cat.name), "category names")?;
    }
    let mask = (1 << c.categories.len()) - 1;
    require(!c.assignments.is_empty() && (c.fallback as usize) < c.assignments.len(), "assignment count")?;
    for a in &c.assignments {
        require(
            (a.primary.0 as usize) < c.categories.len() && a.mask & !mask == 0 && a.mask & (1 << a.primary.0) != 0,
            "assignment category mask",
        )?;
    }
    let fallback = c.assignments[c.fallback as usize];
    require(fallback.primary == c.default && fallback.mask == 1 << c.default.0, "fallback assignment")?;
    require(c.bmp.len() == 65536 && c.bmp.iter().all(|&a| (a as usize) < c.assignments.len()), "BMP assignments")?;
    let mut end = 65535;
    for &(a, b, id) in &c.ranges {
        require(a > end && a <= b && b <= 0x10ffff && (id as usize) < c.assignments.len(), "supplementary ranges")?;
        end = b;
    }
    require(d.unknown.len() == c.categories.len(), "unknown categories")?;
    let mut next = lex as u32;
    for &(a, b) in d.unknown.iter() {
        if a == 0 && b == 0 {
            continue;
        }
        require(a == next && a < b && b as usize <= d.entries.len(), "unknown entry range")?;
        next = b;
    }
    require(next as usize == d.entries.len(), "unassigned unknown entries")?;
    trie(d)
}

/// Linear-time validation of parent links, terminal groups and UTF-8 paths.
/// A four-byte pending prefix uses std's UTF-8 checker, not a local decoder.
fn trie(d: &MecabDictionary) -> Result<(), PrebuiltError> {
    const VALUE: u32 = 1 << 31;
    const OFFSET: u32 = !VALUE;
    const FREE: u32 = u32::MAX;
    let units = &d.trie.units;
    require(
        !units.is_empty() && units.len() < VALUE as usize && units[0].check == 0 && units[0].base & VALUE == 0,
        "trie root",
    )?;
    let mut terminal = vec![false; units.len()];
    let mut groups = vec![false; d.groups.len() - 1];
    for (i, u) in units.iter().enumerate().skip(1) {
        if u.check == FREE {
            continue;
        }
        let p = units.get(u.check as usize).ok_or(PrebuiltError::Invalid("trie parent"))?;
        require(p.check != FREE, "trie free parent")?;
        let label = (i as u64).checked_sub(u64::from(p.base & OFFSET)).ok_or(PrebuiltError::Invalid("trie label"))?;
        require(label <= 256, "trie byte label")?;
        if label == 0 {
            require(p.base & VALUE != 0 && (u.base as usize) < groups.len(), "trie terminal")?;
            require(!groups[u.base as usize], "duplicate surface group")?;
            groups[u.base as usize] = true;
            terminal[i] = true;
        }
    }
    require(groups.iter().all(|&g| g), "missing surface group")?;
    // 254: unseen, 255: on the current parent chain, 0..3: pending UTF-8 bytes.
    let mut lengths = vec![254u8; units.len()];
    let mut pending = vec![[0; 4]; units.len()];
    lengths[0] = 0;
    let mut chain = Vec::new();
    for (i, u) in units.iter().enumerate().skip(1) {
        if u.check == FREE || terminal[i] || lengths[i] != 254 {
            continue;
        }
        let mut at = i;
        while lengths[at] == 254 {
            require(!terminal[at], "terminal used as parent")?;
            lengths[at] = 255;
            chain.push(at);
            at = units[at].check as usize;
        }
        require(lengths[at] != 255 && !terminal[at], "trie cycle or terminal parent")?;
        while let Some(node) = chain.pop() {
            let u = units[node];
            let parent = u.check as usize;
            let label = node as u32 - (units[parent].base & OFFSET);
            require(label > 0, "empty trie edge")?;
            let mut buf = pending[parent];
            let n = lengths[parent] as usize;
            buf[n] = (label - 1) as u8;
            lengths[node] = match std::str::from_utf8(&buf[..n + 1]) {
                Ok(_) => 0,
                Err(e) if e.error_len().is_none() && n < 3 => (n + 1) as u8,
                _ => return Err(PrebuiltError::Invalid("trie UTF-8 path")),
            };
            pending[node] = buf;
            if u.base & VALUE != 0 {
                let t = (u.base & OFFSET) as usize;
                require(
                    lengths[node] == 0 && terminal.get(t) == Some(&true) && units[t].check as usize == node,
                    "trie word boundary",
                )?;
            }
        }
    }
    Ok(())
}
