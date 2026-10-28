mod common;
use common::*;
use std::ptr;
use turbo::quantize::{self, Quantization};
use turbo::*;

fn model(name: &str) -> (Fixture, Loaded) {
    let mut f = Fixture::model(name);
    f.manifest["embed"]["output_dims"] = serde_json::json!([1, 4, 7]);
    f.write();
    let l = f.load().unwrap();
    (f, l)
}
fn read(r: &Outcome) -> Vec<u8> {
    let mut bytes = vec![0; r.info().bytes as usize];
    let mut err = new_error();
    assert_eq!(
        unsafe { turbo_result_read(r.0, bytes.as_mut_ptr().cast(), bytes.len() as u64, ptr::null_mut(), &mut err) },
        0
    );
    bytes
}
fn options(q: u32, width: u32, ranges: &[f32]) -> turbo_embed_options {
    opts(|o| {
        o.quantization = q;
        o.output_dim = width;
        o.int8_ranges = if ranges.is_empty() { ptr::null() } else { ranges.as_ptr() };
    })
}
#[test]
fn ranges_are_owned_after_write_for_text_tokens_and_ranges() {
    let (f, l) = model("probe-calibration-ownership");
    let s = Session::create(l.m, None).unwrap();
    let text = ["a b c"];
    let tok = Tok::create(&f.dir).unwrap();
    let row = tok.row(text[0], None).unwrap();
    let tokens = Tokens::new(&[row.clone()], 0);
    let ranges = [range(0, 0, row.len() as u32)];
    for kind in 0..3 {
        let write = |o: &turbo_embed_options| match kind {
            0 => s.write_text(&text, Some(o)),
            1 => s.write_tokens(&tokens.batch(), Some(o)),
            _ => s.write_ranges(&tokens.batch(), &ranges, Some(o)),
        };
        let float = options(TURBO_QUANTIZE_NONE, 7, &[]);
        write(&float).unwrap();
        let values = s.run().unwrap().rows().concat();
        let original: Vec<f32> = [-1.0; 7].into_iter().chain([1.0; 7]).collect();
        let mut expected = [0; 7];
        quantize::rows(Quantization::Int8, &values, 7, &original, &mut expected);
        let mut caller_ranges = original.clone();
        let o = options(TURBO_QUANTIZE_INT8, 7, &caller_ranges);
        write(&o).unwrap();
        caller_ranges.fill(0.0);
        drop(caller_ranges);
        assert_eq!(read(&s.run().unwrap()), expected, "write kind {kind}");
    }
}
#[test]
fn result_buffer_holds_code_memory_and_blocks_session_reuse() {
    let (_f, l) = model("probe-code-lifetime");
    let s = Session::create(l.m, None).unwrap();
    for q in [TURBO_QUANTIZE_INT8, TURBO_QUANTIZE_UBINARY] {
        let ranges: Vec<_> = [-1.0; 7].into_iter().chain([1.0; 7]).collect();
        let o = options(q, 7, if q == TURBO_QUANTIZE_INT8 { &ranges } else { &[] });
        s.write_text(&["a b"], Some(&o)).unwrap();
        let r = s.run().unwrap();
        let expected = read(&r);
        let mut buffer = ptr::null_mut();
        let mut err = new_error();
        assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buffer, &mut err) }, 0);
        drop(r);
        assert_eq!(s.write_text(&["different text"], None).unwrap_err().code, turbo::status::BUSY);
        let mut host = ptr::null_mut();
        assert_eq!(unsafe { turbo_buffer_host_ptr(buffer, &mut host, &mut err) }, 0);
        assert_eq!(unsafe { std::slice::from_raw_parts(host as *const u8, expected.len()) }, expected);
        unsafe { turbo_buffer_release(buffer) };
        s.write_text(&["different text"], None).unwrap();
        assert_eq!(s.run().unwrap().info().dtype, TURBO_DTYPE_F32);
    }
}
#[cfg(unix)]
#[test]
fn legacy_options_end_at_guard_page_with_only_four_byte_alignment() {
    let (_f, l) = model("probe-legacy-guard");
    let s = Session::create(l.m, None).unwrap();
    unsafe {
        let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
        let memory = libc::mmap(
            ptr::null_mut(),
            page * 2,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        assert_ne!(memory, libc::MAP_FAILED);
        assert_eq!(libc::mprotect(memory.add(page), page, libc::PROT_NONE), 0);
        let old = memory.add(page - 28).cast::<u32>();
        assert_eq!(old as usize % 8, 4);
        for (i, v) in [28, 0, 0, 0, 0, 0, 4].iter().enumerate() {
            old.add(i).write(*v);
        }
        let text = text("a b c");
        let mut err = new_error();
        assert_eq!(turbo_embed_write_text(s.0, &text, 1, old.cast(), &mut err), 0, "{:?}", failure(1, &err));
        assert_eq!(s.run().unwrap().info().dtype, TURBO_DTYPE_F32);
        assert_eq!(libc::munmap(memory, page * 2), 0);
    }
}
#[test]
fn codes_cover_more_ranges_than_batch_at_unaligned_binary_widths() {
    let (f, l) = model("probe-range-capacity");
    let mut d = session_desc(1, 0, TURBO_PRECISION_EXACT);
    d.max_ranges = 8;
    let s = Session::create(l.m, Some(&d)).unwrap();
    let tok = Tok::create(&f.dir).unwrap();
    let row = tok.row("a b c d e f g h i", None).unwrap();
    let tokens = Tokens::new(&[row], 0);
    let ranges: Vec<_> = (0..8).map(|i| range(0, i, i + 2)).collect();
    for width in [1, 4, 7, 8] {
        s.write_ranges(&tokens.batch(), &ranges, Some(&options(TURBO_QUANTIZE_NONE, width, &[]))).unwrap();
        let values = s.run().unwrap().rows().concat();
        s.write_ranges(&tokens.batch(), &ranges, Some(&options(TURBO_QUANTIZE_UBINARY, width, &[]))).unwrap();
        let r = s.run().unwrap();
        let mut expected = vec![0; 8 * (width as usize).div_ceil(8)];
        quantize::rows(Quantization::UBinary, &values, width as usize, &[], &mut expected);
        assert_eq!(read(&r), expected);
        assert_eq!((r.info().batch, r.info().dim, r.info().bytes), (8, width, expected.len() as u64));
    }
}

#[test]
fn quantized_result_exports_the_same_native_host_address_it_maps() {
    let (_f, l) = model("probe-native-code-export");
    let s = Session::create(l.m, None).unwrap();
    let mut failures = Vec::new();
    for q in [TURBO_QUANTIZE_NONE, TURBO_QUANTIZE_INT8, TURBO_QUANTIZE_UBINARY] {
        let ranges: Vec<_> = [-1.0; 8].into_iter().chain([1.0; 8]).collect();
        let o = options(q, 8, if q == TURBO_QUANTIZE_INT8 { &ranges } else { &[] });
        s.write_text(&["a b"], Some(&o)).unwrap();
        let r = s.run().unwrap();
        let mut buffer = ptr::null_mut();
        let mut err = new_error();
        assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buffer, &mut err) }, 0);
        let mut host = ptr::null_mut();
        assert_eq!(unsafe { turbo_buffer_host_ptr(buffer, &mut host, &mut err) }, 0);
        let mut native: turbo_native_handle = unsafe { std::mem::zeroed() };
        native.struct_size = size_of::<turbo_native_handle>() as u32;
        let rc = unsafe { turbo_buffer_export(buffer, TURBO_HANDLE_HOST_PTR, &mut native, &mut err) };
        unsafe { turbo_buffer_release(buffer) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        let matches = native.handle == host as usize as u64;
        eprintln!(
            "quantization={q} export_status={rc} exported_is_null={} matches_mapped_host={matches}",
            native.handle == 0
        );
        if !matches {
            failures.push(q);
        }
    }
    assert!(failures.is_empty(), "native host addresses differ for quantization selectors {failures:?}");
}
