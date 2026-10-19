use super::SnapshotError;
use crate::table::HashSlots;

pub(super) struct Reader<'a> {
    pub(super) bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8], SnapshotError> {
        if self.bytes.len() < n {
            return Err(SnapshotError::Truncated);
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    pub(super) fn len(&mut self) -> Result<usize, SnapshotError> {
        let n = u64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes"));
        usize::try_from(n).map_err(|_| SnapshotError::Truncated)
    }

    pub(super) fn bytes(&mut self) -> Result<&'a [u8], SnapshotError> {
        let n = self.len()?;
        self.take(n)
    }

    pub(super) fn string(&mut self) -> Result<String, SnapshotError> {
        let b = self.bytes()?;
        String::from_utf8(b.to_vec()).map_err(|_| SnapshotError::Corrupt("text that is not UTF-8"))
    }

    pub(super) fn array<T, const N: usize>(&mut self, value: impl Fn([u8; N]) -> T) -> Result<Vec<T>, SnapshotError> {
        let n = self.len()?;
        let bytes = self.take(n.checked_mul(N).ok_or(SnapshotError::Truncated)?)?;
        let (chunks, _) = bytes.as_chunks::<N>();
        Ok(chunks.iter().map(|&c| value(c)).collect())
    }

    pub(super) fn u32s(&mut self) -> Result<Vec<u32>, SnapshotError> {
        self.array(u32::from_le_bytes)
    }

    pub(super) fn u16s(&mut self) -> Result<Vec<u16>, SnapshotError> {
        self.array(u16::from_le_bytes)
    }

    pub(super) fn u64s(&mut self) -> Result<Vec<u64>, SnapshotError> {
        self.array(u64::from_le_bytes)
    }

    pub(super) fn f64s(&mut self) -> Result<Vec<f64>, SnapshotError> {
        self.array(|b| f64::from_bits(u64::from_le_bytes(b)))
    }

    pub(super) fn slots(&mut self) -> Result<HashSlots, SnapshotError> {
        Ok(HashSlots { values: self.u32s()?, tags: self.bytes()?.to_vec(), len: self.len()? })
    }
}
