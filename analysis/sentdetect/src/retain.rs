//! How much a per-thread buffer keeps between calls.

/// The most elements a per-thread buffer keeps between calls. A call that
/// grows one past this hands it back to the allocator afterwards, so a
/// thread that once handled a very large document does not hold that
/// document's peak memory for the rest of its life.
pub(crate) const RETAINED: usize = 1 << 16;

/// Hands `v` back to the allocator when it grew past [`RETAINED`].
pub(crate) fn release<T>(v: &mut Vec<T>) {
    if v.capacity() > RETAINED {
        drop(std::mem::take(v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_buffer_past_the_limit_is_released() {
        let mut small: Vec<u8> = Vec::with_capacity(RETAINED);
        release(&mut small);
        assert!(small.capacity() >= RETAINED);
        let mut large: Vec<u8> = Vec::with_capacity(RETAINED + 1);
        release(&mut large);
        assert_eq!(large.capacity(), 0);
    }
}
