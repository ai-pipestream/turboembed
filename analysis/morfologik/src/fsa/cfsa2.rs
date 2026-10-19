//! CFSA2 finite-state automaton reader (`opennlp.tools.formats.CFSA2Reader`).

use super::{
    AutomatonOps, HEADER_SIZE, NO_ARC, TERMINAL_NODE, VERSION_CFSA2, require_fsa_header, require_range,
    try_traverse_all, validate_automaton,
};
use crate::error::MorfologikError;

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
