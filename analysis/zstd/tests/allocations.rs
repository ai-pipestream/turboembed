//! A size the frames cannot come to is refused before the output is
//! allocated, so asking for 32 GiB of a small input costs nothing. This
//! binary counts the bytes allocated on the test's thread; it holds one
//! test.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use common::build::{frame, raw, rle, windowed};
use common::fixture;
use turbo_zstd::{Error, decompress};

struct Counting;

thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

fn count(n: usize) {
    BYTES.with(|c| c.set(c.get() + n));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn refused_sizes_allocate_nothing() {
    let zeros = vec![0u8; 1 << 20];
    // The same 1 MiB of zeros as a frame with no content size and a 1 MiB
    // window: eight RLE blocks of 128 KiB.
    let mut blocks = vec![rle(0, 128 << 10, false); 7];
    blocks.push(rle(0, 128 << 10, true));
    let rle_zeros = frame(&windowed(10 << 3), &blocks);
    // A frame with no content size whose blocks hold 8 bytes.
    let small = frame(&windowed(0), &[raw(b"abc", false), rle(b'x', 5, true)]);
    // The `zstd` tool's frames: with a content size of 150,000, and from
    // a pipe, with none and one compressed block of at most 128 KiB.
    let text = fixture("text-1.zst");
    let pipe = fixture("small-pipe.zst");
    let cases = [
        (&zeros[..], 1 << 35, Error::BadMagic(0)),
        (&rle_zeros[..], 1 << 35, Error::CannotExpand { max: 1 << 20, want: 1 << 35 }),
        (&small[..], 1 << 35, Error::CannotExpand { max: 8, want: 1 << 35 }),
        (&small[..], 9, Error::CannotExpand { max: 8, want: 9 }),
        (&pipe[..], (128 << 10) + 1, Error::CannotExpand { max: 128 << 10, want: (128 << 10) + 1 }),
        (&text[..], 1 << 35, Error::TooShort { got: 150_000, want: 1 << 35 }),
        (&text[..], 150_001, Error::TooShort { got: 150_000, want: 150_001 }),
        (&text[..], 149_999, Error::TooLong { want: 149_999 }),
    ];
    for (src, want, e) in cases {
        let before = BYTES.with(Cell::get);
        let r = decompress(src, want);
        let allocated = BYTES.with(Cell::get) - before;
        assert_eq!(r, Err(e));
        assert_eq!(allocated, 0, "{want} asked for");
    }
    // A size the frames can come to is allocated once.
    let before = BYTES.with(Cell::get);
    assert_eq!(decompress(&small, 8).unwrap(), b"abcxxxxx");
    assert_eq!(BYTES.with(Cell::get) - before, 8);
    let before = BYTES.with(Cell::get);
    assert_eq!(decompress(&rle_zeros, 1 << 20).unwrap(), zeros);
    assert_eq!(BYTES.with(Cell::get) - before, 1 << 20);
}
