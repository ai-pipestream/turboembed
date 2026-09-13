//! gRPC / OIP **output scratch** — a size-class freelist for LE FP32 blobs.
//!
//! SOLIDIFY (6): Embed `PACKED_BYTES`, OIP `raw_output_contents`, and
//! Rerank score rows rent a pre-sized slab. After warmup of a given
//! byte/float shape, [`allocs`] stays flat — the payload is not a fresh
//! heap `Vec` per request. Prost still copies the slab into the HTTP/2
//! frame; that wire copy is not this pool. Claiming "protobuf has to
//! copy" without a freelist is not done.
//!
//! `Bytes::from_owner` holds the slab until the encoded response is
//! dropped, then [`PooledBytes`] returns it. Concurrent RPCs check out
//! distinct slabs; sequential unary/batch reuse the same one.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use bytes::Bytes;

use crate::tensor::pack_fp32_into;

/// Soft cap on free slabs so a burst of large batches cannot pin RAM.
const MAX_FREE_BYTE_SLABS: usize = 32;
const MAX_FREE_F32_SLABS: usize = 32;

struct BytePool {
    free: Vec<Vec<u8>>,
}

struct F32Pool {
    free: Vec<Vec<f32>>,
}

struct Pools {
    bytes: BytePool,
    f32s: F32Pool,
}

static POOLS: Mutex<Option<Pools>> = Mutex::new(None);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static RENTS: AtomicU64 = AtomicU64::new(0);
static RETURNS: AtomicU64 = AtomicU64::new(0);

fn pools() -> std::sync::MutexGuard<'static, Option<Pools>> {
    POOLS.lock().expect("output scratch mutex poisoned")
}

fn with_pools<T>(f: impl FnOnce(&mut Pools) -> T) -> T {
    let mut guard = pools();
    let pools = guard.get_or_insert_with(|| Pools {
        bytes: BytePool { free: Vec::new() },
        f32s: F32Pool { free: Vec::new() },
    });
    f(pools)
}

/// New backing allocations (byte + f32 slabs) since process start or
/// the last [`reset_counters`].
pub fn allocs() -> u64 {
    ALLOCS.load(Ordering::Relaxed)
}

pub fn rents() -> u64 {
    RENTS.load(Ordering::Relaxed)
}

pub fn returns() -> u64 {
    RETURNS.load(Ordering::Relaxed)
}

/// Reset counters only. Live / free slabs stay checked out so warmup
/// reuse is still visible after a test snapshot.
pub fn reset_counters() {
    ALLOCS.store(0, Ordering::Relaxed);
    RENTS.store(0, Ordering::Relaxed);
    RETURNS.store(0, Ordering::Relaxed);
}

fn take_bytes(min_cap: usize) -> Vec<u8> {
    RENTS.fetch_add(1, Ordering::Relaxed);
    with_pools(|p| {
        if let Some(idx) = p
            .bytes
            .free
            .iter()
            .enumerate()
            .filter(|(_, s)| s.capacity() >= min_cap)
            .min_by_key(|(_, s)| s.capacity())
            .map(|(i, _)| i)
        {
            let mut slab = p.bytes.free.swap_remove(idx);
            slab.clear();
            slab
        } else {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            Vec::with_capacity(min_cap)
        }
    })
}

fn give_bytes(mut slab: Vec<u8>) {
    if slab.capacity() == 0 {
        return;
    }
    slab.clear();
    RETURNS.fetch_add(1, Ordering::Relaxed);
    with_pools(|p| {
        if p.bytes.free.len() >= MAX_FREE_BYTE_SLABS {
            return;
        }
        p.bytes.free.push(slab);
    });
}

fn take_f32(min_cap: usize) -> Vec<f32> {
    RENTS.fetch_add(1, Ordering::Relaxed);
    with_pools(|p| {
        if let Some(idx) = p
            .f32s
            .free
            .iter()
            .enumerate()
            .filter(|(_, s)| s.capacity() >= min_cap)
            .min_by_key(|(_, s)| s.capacity())
            .map(|(i, _)| i)
        {
            let mut slab = p.f32s.free.swap_remove(idx);
            slab.clear();
            slab
        } else {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            Vec::with_capacity(min_cap)
        }
    })
}

fn give_f32(mut slab: Vec<f32>) {
    if slab.capacity() == 0 {
        return;
    }
    slab.clear();
    RETURNS.fetch_add(1, Ordering::Relaxed);
    with_pools(|p| {
        if p.f32s.free.len() >= MAX_FREE_F32_SLABS {
            return;
        }
        p.f32s.free.push(slab);
    });
}

/// Rent an empty byte slab with `capacity >= min_cap`. Caller fills
/// `len`; return it with [`recycle_bytes`] or [`adopt_bytes`].
pub fn rent_bytes(min_cap: usize) -> Vec<u8> {
    take_bytes(min_cap)
}

/// Return a byte slab to the freelist. No-op for a zero-capacity vec.
pub fn recycle_bytes(buf: Vec<u8>) {
    give_bytes(buf);
}

/// Rent an empty f32 slab with `capacity >= min_cap`.
pub fn rent_f32(min_cap: usize) -> Vec<f32> {
    take_f32(min_cap)
}

pub fn recycle_f32(buf: Vec<f32>) {
    give_f32(buf);
}

/// Ensure `dest` can hold `need` bytes without treating a pool hit as
/// a new alloc. A grow of an undersized slab counts.
pub fn ensure_bytes(dest: &mut Vec<u8>, need: usize) {
    if dest.capacity() < need {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        dest.reserve(need);
    }
}

pub fn ensure_f32(dest: &mut Vec<f32>, need: usize) {
    if dest.capacity() < need {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        dest.reserve(need);
    }
}

/// Write `values` as little-endian FP32 into `dest`, reusing capacity.
pub fn pack_le_f32_into(values: &[f32], dest: &mut Vec<u8>) {
    let need = values.len() * 4;
    ensure_bytes(dest, need);
    dest.clear();
    pack_fp32_into(values, dest);
}

/// Rent + pack + wrap as [`Bytes`] that returns the slab on drop.
pub fn pack_le_f32(values: &[f32]) -> Bytes {
    let mut dest = rent_bytes(values.len() * 4);
    pack_le_f32_into(values, &mut dest);
    adopt_bytes(dest)
}

/// Copy `src` into a rented slab and wrap as [`Bytes`].
pub fn copy_bytes(src: &[u8]) -> Bytes {
    let mut dest = rent_bytes(src.len());
    dest.extend_from_slice(src);
    adopt_bytes(dest)
}

/// Wrap a (usually rented) `Vec<u8>` so Drop returns it to the pool.
pub fn adopt_bytes(buf: Vec<u8>) -> Bytes {
    if buf.is_empty() && buf.capacity() == 0 {
        return Bytes::new();
    }
    Bytes::from_owner(PooledBytes { buf })
}

/// Owner stored inside [`Bytes`]. Last clone drop returns the slab.
struct PooledBytes {
    buf: Vec<u8>,
}

impl AsRef<[u8]> for PooledBytes {
    fn as_ref(&self) -> &[u8] {
        &self.buf
    }
}

impl Drop for PooledBytes {
    fn drop(&mut self) {
        give_bytes(std::mem::take(&mut self.buf));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_blob_reuses_after_warmup() {
        reset_counters();
        let row = [1.0f32, 2.0, 3.0, 4.0];
        let first = pack_le_f32(&row);
        assert_eq!(first.len(), 16);
        assert_eq!(&first[..4], &1.0f32.to_le_bytes());
        drop(first);
        // A prior test may have left a ≥16-byte slab; warmup then does
        // not allocate. Steady-state must not grow either way.
        let after_warm = allocs();

        for _ in 0..8 {
            let blob = pack_le_f32(&row);
            assert_eq!(blob.len(), 16);
            drop(blob);
        }
        assert_eq!(
            allocs(),
            after_warm,
            "steady-state pack_le_f32 must not grow the freelist"
        );
        assert!(returns() >= 8);
    }

    #[test]
    fn adopt_returns_capacity_to_next_rent() {
        reset_counters();
        let mut dest = rent_bytes(64);
        dest.extend_from_slice(&[1, 2, 3, 4]);
        let bytes = adopt_bytes(dest);
        assert_eq!(&bytes[..], &[1, 2, 3, 4]);
        drop(bytes);
        let again = rent_bytes(64);
        assert!(
            again.capacity() >= 64,
            "returned slab must keep capacity, got {}",
            again.capacity()
        );
        recycle_bytes(again);
    }

    #[test]
    fn f32_scores_reuse_after_warmup() {
        reset_counters();
        let mut scores = rent_f32(3);
        scores.extend_from_slice(&[0.1, 0.2, 0.3]);
        recycle_f32(scores);
        let after_warm = allocs();
        for _ in 0..8 {
            let mut dest = rent_f32(3);
            dest.extend_from_slice(&[0.4, 0.5, 0.6]);
            assert_eq!(dest.len(), 3);
            recycle_f32(dest);
        }
        assert_eq!(allocs(), after_warm);
    }

    #[test]
    fn bytes_clone_returns_once() {
        reset_counters();
        let blob = pack_le_f32(&[9.0f32; 8]);
        let clone = blob.clone();
        drop(blob);
        // Slab still held by clone — next rent of this size may allocate
        // a sibling. Dropping the clone must return exactly one slab.
        let before = returns();
        drop(clone);
        assert_eq!(returns(), before + 1);
    }
}
