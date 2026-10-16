//! FSA5 and CFSA2 finite-state automaton readers, ported from OpenNLP 3.x's
//! `FsaSequenceReader`, `FSA5Reader`, `CFSA2Reader`, and `FsaTraversal`.

use crate::error::MorfologikError;

/// ASCII `\fsa`, the shared magic header of both Morfologik automaton formats.
pub const MAGIC: [u8; 4] = [0x5c, 0x66, 0x73, 0x61];

/// Version byte of the FSA5 format.
pub const VERSION_FSA5: u8 = 0x05;

/// Version byte of the CFSA2 format.
pub const VERSION_CFSA2: u8 = 0xc6;

const HEADER_SIZE: usize = 8;
const TERMINAL_NODE: usize = 0;
const NO_ARC: usize = 0;

/// Checks the ASCII `\fsa` signature and the presence of a version byte.
pub fn require_fsa_header(bytes: &[u8]) -> Result<(), MorfologikError> {
    if bytes.len() <= MAGIC.len() || bytes[..MAGIC.len()] != MAGIC {
        return Err(MorfologikError::format("not an FSA automaton: bad magic header"));
    }
    Ok(())
}

/// Format-agnostic reader over an FSA5 or CFSA2 finite-state automaton.
#[derive(Debug, Clone)]
pub enum FsaReader {
    /// An FSA5 automaton (version `0x05`).
    Fsa5(Fsa5Reader),
    /// A CFSA2 automaton (version `0xc6`).
    Cfsa2(Cfsa2Reader),
}

impl FsaReader {
    /// Reads an FSA5 or CFSA2 automaton from bytes, dispatching on the version byte.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MorfologikError> {
        require_fsa_header(bytes)?;
        let version = bytes[4];
        match version {
            VERSION_CFSA2 => Cfsa2Reader::from_bytes(bytes).map(Self::Cfsa2),
            VERSION_FSA5 => Fsa5Reader::from_bytes(bytes).map(Self::Fsa5),
            _ => Err(MorfologikError::format(format!(
                "unsupported FSA version 0x{version:x}; only FSA5 (0x05) and CFSA2 (0xc6) are read"
            ))),
        }
    }

    /// Passes every accepted byte sequence to `action` in the automaton's stored order.
    pub fn for_each_sequence<F>(&self, action: F)
    where
        F: FnMut(&[u8]),
    {
        match self {
            Self::Fsa5(reader) => reader.for_each_sequence(action),
            Self::Cfsa2(reader) => reader.for_each_sequence(action),
        }
    }

    /// Passes every accepted byte sequence to `action`, stopping early on the first error.
    pub fn try_for_each_sequence<F, E>(&self, action: F) -> Result<(), E>
    where
        F: FnMut(&[u8]) -> Result<(), E>,
    {
        match self {
            Self::Fsa5(reader) => reader.try_for_each_sequence(action),
            Self::Cfsa2(reader) => reader.try_for_each_sequence(action),
        }
    }

    /// Collects all accepted byte sequences in stored order.
    pub fn sequences(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        self.for_each_sequence(|seq| out.push(seq.to_vec()));
        out
    }

    /// Traverses the automaton from the root along `word` followed by `separator`,
    /// calling `action` with each accepted suffix after `separator` (in stored order).
    ///
    /// This mirrors Morfologik's `DictionaryLookup`: it walks the prefix `word + separator`
    /// in $O(|word|)$ arc steps without expanding unrelated entries in the automaton.
    pub fn lookup_with_separator<F>(&self, word: &[u8], separator: u8, action: F)
    where
        F: FnMut(&[u8]),
    {
        match self {
            Self::Fsa5(reader) => traverse_suffix_after_prefix(reader, word, separator, action),
            Self::Cfsa2(reader) => traverse_suffix_after_prefix(reader, word, separator, action),
        }
    }
}

/// Internal low-level trait implemented by `Fsa5Reader` and `Cfsa2Reader`.
trait AutomatonOps {
    fn root_node(&self) -> usize;
    fn data_len(&self) -> usize;
    fn first_arc(&self, node: usize) -> Result<usize, MorfologikError>;
    fn next_arc(&self, arc: usize) -> Result<usize, MorfologikError>;
    fn destination_node(&self, arc: usize) -> Result<usize, MorfologikError>;
    fn arc_label(&self, arc: usize) -> Result<u8, MorfologikError>;
    fn is_final(&self, arc: usize) -> bool;
}

fn require_range(offset: usize, length: usize, limit: usize) -> Result<(), MorfologikError> {
    if length > limit || offset > limit - length {
        return Err(MorfologikError::invalid_argument(format!(
            "FSA byte range outside data: {offset}, length {length}"
        )));
    }
    Ok(())
}

/// Validates reachable nodes and arcs without recursion, rejecting out-of-range offsets
/// and cycles.
fn validate_automaton<A: AutomatonOps>(automaton: &A) -> Result<(), MorfologikError> {
    let root = automaton.root_node();
    if root == TERMINAL_NODE {
        return Ok(());
    }
    let data_len = automaton.data_len();
    let word_len = data_len.div_ceil(64);
    let mut active = vec![0u64; word_len];
    let mut complete = vec![0u64; word_len];
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(32);

    push_validated_node(automaton, &mut stack, root, data_len, &mut active)?;
    while let Some(&(_, arc)) = stack.last() {
        if arc == NO_ARC {
            let (node, _) = stack.pop().unwrap();
            clear_bit(&mut active, node);
            set_bit(&mut complete, node);
            continue;
        }
        let _ = automaton.arc_label(arc)?;
        let next = automaton.next_arc(arc)?;
        stack.last_mut().unwrap().1 = next;
        let target = automaton.destination_node(arc)?;
        if target != TERMINAL_NODE {
            require_range(target, 1, data_len)?;
            if get_bit(&active, target) {
                return Err(MorfologikError::invalid_argument(format!("FSA contains a cycle at byte {target}")));
            }
            if !get_bit(&complete, target) {
                push_validated_node(automaton, &mut stack, target, data_len, &mut active)?;
            }
        }
    }
    Ok(())
}

fn push_validated_node<A: AutomatonOps>(
    automaton: &A,
    stack: &mut Vec<(usize, usize)>,
    node: usize,
    data_len: usize,
    active: &mut [u64],
) -> Result<(), MorfologikError> {
    require_range(node, 1, data_len)?;
    let first = automaton.first_arc(node)?;
    set_bit(active, node);
    stack.push((node, first));
    Ok(())
}

#[inline]
fn get_bit(bits: &[u64], idx: usize) -> bool {
    (bits[idx >> 6] & (1u64 << (idx & 63))) != 0
}

#[inline]
fn set_bit(bits: &mut [u64], idx: usize) {
    bits[idx >> 6] |= 1u64 << (idx & 63);
}

#[inline]
fn clear_bit(bits: &mut [u64], idx: usize) {
    bits[idx >> 6] &= !(1u64 << (idx & 63));
}

fn try_traverse_all<A: AutomatonOps, F, E>(automaton: &A, mut action: F) -> Result<(), E>
where
    F: FnMut(&[u8]) -> Result<(), E>,
{
    let root = automaton.root_node();
    if root == TERMINAL_NODE {
        return Ok(());
    }
    let mut path: Vec<u8> = Vec::with_capacity(64);
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(32);
    let first = automaton.first_arc(root).expect("validated on load");
    stack.push((root, first));
    while let Some(&(_, arc)) = stack.last() {
        if arc == NO_ARC {
            stack.pop();
            if !stack.is_empty() {
                path.pop();
            }
            continue;
        }
        let next = automaton.next_arc(arc).expect("validated on load");
        stack.last_mut().unwrap().1 = next;
        let label = automaton.arc_label(arc).expect("validated on load");
        path.push(label);
        if automaton.is_final(arc) {
            action(&path)?;
        }
        let target = automaton.destination_node(arc).expect("validated on load");
        if target == TERMINAL_NODE {
            path.pop();
        } else {
            let target_first = automaton.first_arc(target).expect("validated on load");
            stack.push((target, target_first));
        }
    }
    Ok(())
}

fn traverse_suffix_after_prefix<A: AutomatonOps, F>(automaton: &A, word: &[u8], separator: u8, mut action: F)
where
    F: FnMut(&[u8]),
{
    let mut node = automaton.root_node();
    if node == TERMINAL_NODE {
        return;
    }
    for &b in word.iter().chain(std::iter::once(&separator)) {
        let mut arc = automaton.first_arc(node).expect("validated on load");
        let mut found = None;
        while arc != NO_ARC {
            if automaton.arc_label(arc).expect("validated on load") == b {
                found = Some(arc);
                break;
            }
            arc = automaton.next_arc(arc).expect("validated on load");
        }
        let Some(matched_arc) = found else {
            return;
        };
        let target = automaton.destination_node(matched_arc).expect("validated on load");
        if target == TERMINAL_NODE {
            return;
        }
        node = target;
    }

    let mut path: Vec<u8> = Vec::with_capacity(32);
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(16);
    let first = automaton.first_arc(node).expect("validated on load");
    stack.push((node, first));
    while let Some(&(_, arc)) = stack.last() {
        if arc == NO_ARC {
            stack.pop();
            if !stack.is_empty() {
                path.pop();
            }
            continue;
        }
        let next = automaton.next_arc(arc).expect("validated on load");
        stack.last_mut().unwrap().1 = next;
        let label = automaton.arc_label(arc).expect("validated on load");
        path.push(label);
        if automaton.is_final(arc) {
            action(&path);
        }
        let target = automaton.destination_node(arc).expect("validated on load");
        if target == TERMINAL_NODE {
            path.pop();
        } else {
            let target_first = automaton.first_arc(target).expect("validated on load");
            stack.push((target, target_first));
        }
    }
}

/// Reader over an FSA5 finite-state automaton (`opennlp.tools.formats.FSA5Reader`).
#[derive(Debug, Clone)]
pub struct Fsa5Reader {
    arcs: Vec<u8>,
    goto_length: usize,
    node_data_length: usize,
    root_node: usize,
}

impl Fsa5Reader {
    const BIT_FINAL_ARC: u8 = 0x01;
    const BIT_LAST_ARC: u8 = 0x02;
    const BIT_TARGET_NEXT: u8 = 0x04;
    const ADDRESS_OFFSET: usize = 1;
    const MAX_ENCODED_ADDRESS: u64 = ((i32::MAX as u64) << 3) | 7;

    /// Reads and validates an FSA5 automaton from a byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MorfologikError> {
        require_fsa_header(bytes)?;
        if bytes.len() < HEADER_SIZE {
            return Err(MorfologikError::format(format!("truncated FSA5 header: fewer than {HEADER_SIZE} bytes")));
        }
        if bytes[4] != VERSION_FSA5 {
            return Err(MorfologikError::format(format!(
                "unsupported FSA version 0x{:x}; only FSA5 (0x05) is read here",
                bytes[4]
            )));
        }
        let widths = bytes[7];
        let goto_length = usize::from(widths & 0x0f);
        let node_data_length = usize::from((widths >> 4) & 0x0f);
        if goto_length < 1 {
            return Err(MorfologikError::format(format!("invalid FSA5 goto length: {goto_length}")));
        }
        let arcs = bytes[HEADER_SIZE..].to_vec();
        Self::new(arcs, goto_length, node_data_length)
            .map_err(|e| MorfologikError::format(format!("malformed FSA5 automaton: {e}")))
    }

    fn new(arcs: Vec<u8>, goto_length: usize, node_data_length: usize) -> Result<Self, MorfologikError> {
        let mut reader = Self { arcs, goto_length, node_data_length, root_node: TERMINAL_NODE };
        let dummy_first = reader.first_arc(TERMINAL_NODE)?;
        let epsilon_node = reader.skip_arc(dummy_first)?;
        let epsilon_first = reader.first_arc(epsilon_node)?;
        reader.root_node = reader.destination_node(epsilon_first)?;
        validate_automaton(&reader)?;
        Ok(reader)
    }

    fn skip_arc(&self, arc: usize) -> Result<usize, MorfologikError> {
        require_range(arc, Self::ADDRESS_OFFSET + 1, self.arcs.len())?;
        let flags = self.arcs[arc + Self::ADDRESS_OFFSET];
        let length = Self::ADDRESS_OFFSET + if (flags & Self::BIT_TARGET_NEXT) != 0 { 1 } else { self.goto_length };
        require_range(arc, length, self.arcs.len())?;
        Ok(arc + length)
    }

    /// Passes every accepted byte sequence to `action` in stored order.
    pub fn for_each_sequence<F>(&self, mut action: F)
    where
        F: FnMut(&[u8]),
    {
        let _: Result<(), std::convert::Infallible> = self.try_for_each_sequence(|seq| {
            action(seq);
            Ok(())
        });
    }

    /// Passes every accepted byte sequence to `action`, stopping early on the first error.
    pub fn try_for_each_sequence<F, E>(&self, action: F) -> Result<(), E>
    where
        F: FnMut(&[u8]) -> Result<(), E>,
    {
        try_traverse_all(self, action)
    }

    /// Collects all accepted byte sequences in stored order.
    pub fn sequences(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        self.for_each_sequence(|seq| out.push(seq.to_vec()));
        out
    }
}

impl AutomatonOps for Fsa5Reader {
    fn root_node(&self) -> usize {
        self.root_node
    }

    fn data_len(&self) -> usize {
        self.arcs.len()
    }

    fn first_arc(&self, node: usize) -> Result<usize, MorfologikError> {
        require_range(node, self.node_data_length + 1, self.arcs.len())?;
        Ok(self.node_data_length + node)
    }

    fn next_arc(&self, arc: usize) -> Result<usize, MorfologikError> {
        let end = self.skip_arc(arc)?;
        if (self.arcs[arc + Self::ADDRESS_OFFSET] & Self::BIT_LAST_ARC) != 0 { Ok(NO_ARC) } else { Ok(end) }
    }

    fn destination_node(&self, arc: usize) -> Result<usize, MorfologikError> {
        let end = self.skip_arc(arc)?;
        if (self.arcs[arc + Self::ADDRESS_OFFSET] & Self::BIT_TARGET_NEXT) != 0 {
            return Ok(end);
        }
        let mut value: u64 = 0;
        for i in (0..self.goto_length).rev() {
            value = (value << 8) | u64::from(self.arcs[arc + Self::ADDRESS_OFFSET + i]);
            if value > Self::MAX_ENCODED_ADDRESS {
                return Err(MorfologikError::invalid_argument("FSA5 target exceeds the address range"));
            }
        }
        Ok((value >> 3) as usize)
    }

    fn arc_label(&self, arc: usize) -> Result<u8, MorfologikError> {
        require_range(arc, 1, self.arcs.len())?;
        Ok(self.arcs[arc])
    }

    fn is_final(&self, arc: usize) -> bool {
        (self.arcs[arc + Self::ADDRESS_OFFSET] & Self::BIT_FINAL_ARC) != 0
    }
}

/// Reader over a CFSA2 finite-state automaton (`opennlp.tools.formats.CFSA2Reader`).
#[derive(Debug, Clone)]
pub struct Cfsa2Reader {
    arcs: Vec<u8>,
    label_mapping: Vec<u8>,
    has_numbers: bool,
    root_node: usize,
}

impl Cfsa2Reader {
    const FLAG_NUMBERS: u16 = 0x0100;
    const KNOWN_FLAGS: u16 = 0x030f;

    const BIT_TARGET_NEXT: u8 = 0x80;
    const BIT_LAST_ARC: u8 = 0x40;
    const BIT_FINAL_ARC: u8 = 0x20;
    const LABEL_INDEX_MASK: u8 = 0x1f;
    const MAX_VINT_BYTES: usize = 5;

    /// Reads and validates a CFSA2 automaton from a byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MorfologikError> {
        require_fsa_header(bytes)?;
        if bytes.len() < HEADER_SIZE {
            return Err(MorfologikError::format(format!("truncated CFSA2 header: fewer than {HEADER_SIZE} bytes")));
        }
        if bytes[4] != VERSION_CFSA2 {
            return Err(MorfologikError::format(format!(
                "unsupported FSA version 0x{:x}; only CFSA2 (0xc6) is read",
                bytes[4]
            )));
        }
        let flags = (u16::from(bytes[5]) << 8) | u16::from(bytes[6]);
        if (flags & !Self::KNOWN_FLAGS) != 0 {
            return Err(MorfologikError::format(format!("unrecognized CFSA2 flags: 0x{flags:x}")));
        }
        let label_mapping_size = usize::from(bytes[7]);
        let arcs_start = HEADER_SIZE + label_mapping_size;
        if arcs_start > bytes.len() {
            return Err(MorfologikError::format("truncated CFSA2 header: label table runs past end of data"));
        }
        let label_mapping = bytes[HEADER_SIZE..arcs_start].to_vec();
        let arcs = bytes[arcs_start..].to_vec();
        Self::new(arcs, label_mapping, (flags & Self::FLAG_NUMBERS) != 0)
            .map_err(|e| MorfologikError::format(format!("malformed CFSA2 automaton: {e}")))
    }

    fn new(arcs: Vec<u8>, label_mapping: Vec<u8>, has_numbers: bool) -> Result<Self, MorfologikError> {
        let mut reader = Self { arcs, label_mapping, has_numbers, root_node: TERMINAL_NODE };
        let first = reader.first_arc(TERMINAL_NODE)?;
        reader.root_node = reader.destination_node(first)?;
        validate_automaton(&reader)?;
        Ok(reader)
    }

    fn flags(&self, arc: usize) -> Result<u8, MorfologikError> {
        require_range(arc, 1, self.arcs.len())?;
        Ok(self.arcs[arc])
    }

    fn skip_arc(&self, mut offset: usize) -> Result<usize, MorfologikError> {
        let flag = self.flags(offset)?;
        let prefix = if (flag & Self::LABEL_INDEX_MASK) == 0 { 2 } else { 1 };
        require_range(offset, prefix, self.arcs.len())?;
        offset += prefix;
        if (flag & Self::BIT_TARGET_NEXT) == 0 {
            offset = self.skip_vint(offset)?;
        }
        Ok(offset)
    }

    fn read_vint(&self, offset: usize) -> Result<usize, MorfologikError> {
        let end = self.skip_vint(offset)?;
        let mut value: i32 = 0;
        for i in offset..end {
            value |= i32::from(self.arcs[i] & 0x7f) << ((i - offset) * 7);
        }
        if value < 0 {
            return Err(MorfologikError::invalid_argument("CFSA2 target exceeds the address range"));
        }
        Ok(value as usize)
    }

    fn skip_vint(&self, offset: usize) -> Result<usize, MorfologikError> {
        for i in 0..Self::MAX_VINT_BYTES {
            require_range(offset, i + 1, self.arcs.len())?;
            let value = self.arcs[offset + i];
            if i == Self::MAX_VINT_BYTES - 1 && (value & 0xf0) != 0 {
                break;
            }
            if (value & 0x80) == 0 {
                return Ok(offset + i + 1);
            }
        }
        Err(MorfologikError::invalid_argument("CFSA2 variable-length integer exceeds 32 bits"))
    }

    /// Passes every accepted byte sequence to `action` in stored order.
    pub fn for_each_sequence<F>(&self, mut action: F)
    where
        F: FnMut(&[u8]),
    {
        let _: Result<(), std::convert::Infallible> = self.try_for_each_sequence(|seq| {
            action(seq);
            Ok(())
        });
    }

    /// Passes every accepted byte sequence to `action`, stopping early on the first error.
    pub fn try_for_each_sequence<F, E>(&self, action: F) -> Result<(), E>
    where
        F: FnMut(&[u8]) -> Result<(), E>,
    {
        try_traverse_all(self, action)
    }

    /// Collects all accepted byte sequences in stored order.
    pub fn sequences(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        self.for_each_sequence(|seq| out.push(seq.to_vec()));
        out
    }
}

impl AutomatonOps for Cfsa2Reader {
    fn root_node(&self) -> usize {
        self.root_node
    }

    fn data_len(&self) -> usize {
        self.arcs.len()
    }

    fn first_arc(&self, node: usize) -> Result<usize, MorfologikError> {
        let first = if self.has_numbers { self.skip_vint(node)? } else { node };
        require_range(first, 1, self.arcs.len())?;
        Ok(first)
    }

    fn next_arc(&self, arc: usize) -> Result<usize, MorfologikError> {
        let end = self.skip_arc(arc)?;
        let flag = self.flags(arc)?;
        if (flag & Self::BIT_LAST_ARC) != 0 { Ok(NO_ARC) } else { Ok(end) }
    }

    fn destination_node(&self, arc: usize) -> Result<usize, MorfologikError> {
        let flag = self.flags(arc)?;
        if (flag & Self::BIT_TARGET_NEXT) != 0 {
            let mut last = arc;
            while (self.flags(last)? & Self::BIT_LAST_ARC) == 0 {
                last = self.skip_arc(last)?;
            }
            return self.skip_arc(last);
        }
        let prefix = if (flag & Self::LABEL_INDEX_MASK) == 0 { 2 } else { 1 };
        self.read_vint(arc + prefix)
    }

    fn arc_label(&self, arc: usize) -> Result<u8, MorfologikError> {
        let index = usize::from(self.flags(arc)? & Self::LABEL_INDEX_MASK);
        if index > 0 {
            require_range(index, 1, self.label_mapping.len())?;
            return Ok(self.label_mapping[index]);
        }
        require_range(arc, 2, self.arcs.len())?;
        Ok(self.arcs[arc + 1])
    }

    fn is_final(&self, arc: usize) -> bool {
        (self.arcs[arc] & Self::BIT_FINAL_ARC) != 0
    }
}
