//! Linux x86-64 native guard-page check of the actual production scanner.
#[path = "../../../analysis/structure/src/columns/scan.rs"]
mod scan;
use std::ffi::{c_int, c_void};
unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: isize) -> *mut c_void;
    fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn sysconf(name: c_int) -> isize;
}
fn main() {
    // SAFETY: this Linux-only probe owns the three-page mapping. Only its
    // writable middle page is borrowed, every slice is within that page,
    // and all borrows end before the owned mapping is released.
    unsafe {
        let page = usize::try_from(sysconf(30)).unwrap();
        assert!(page >= 4096);
        let area = mmap(std::ptr::null_mut(), page * 3, 0, 0x22, -1, 0);
        assert_ne!(area as isize, -1);
        let middle = (area as *mut u8).add(page);
        assert_eq!(mprotect(middle.cast(), page, 3), 0);
        let mut checks = 0;
        for len in 0..=96 {
            let ptr = middle.add(page - len);
            let data = std::slice::from_raw_parts_mut(ptr, len);
            data.fill(b'x');
            let check = |data: &[u8]| {
                let text = std::str::from_utf8(data).unwrap();
                assert_eq!(scan::candidate(text), text.contains("  ") || text.contains('\t'));
            };
            check(data);
            checks += 1;
            for i in 0..len {
                data[i] = b'\t';
                check(data);
                checks += 1;
                data[i] = b'x';
            }
            for i in 1..len {
                data[i - 1] = b' ';
                data[i] = b' ';
                check(data);
                checks += 1;
                data[i - 1] = b'x';
                data[i] = b'x';
            }
        }
        assert_eq!(munmap(area, page * 3), 0);
        println!(
            "{checks} actual scanner checks passed against protected page boundaries; all lengths0..96, every tab/pair position, all address alignments"
        );
    }
}
