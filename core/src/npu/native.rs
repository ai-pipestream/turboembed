//! `ZE_GRAPH_FORMAT_NATIVE`: the blob the driver compiled, loaded back
//! as a graph. The bytes come from `pfnGetNativeBinary2` (extension
//! 1.7, the driver owns the view) or `pfnGetNativeBinary` (the caller
//! owns the buffer). This module copies them. It does not call
//! OpenVINO or ONNX Runtime.

use std::ffi::c_char;

use super::ze::{self, GraphExt, Handle};

#[derive(Debug)]
pub enum BlobError {
    Unsupported(String),
    Runtime(String),
}

/// The graph descriptor that loads `blob` as a native graph. Build
/// flags are the caller's C string, which is empty for a blob: the
/// compiler already ran.
pub fn descriptor(blob: &[u8], build_flags: *const c_char) -> ze::GraphDesc2 {
    ze::GraphDesc2 {
        stype: ze::STRUCTURE_TYPE_GRAPH_DESC_2,
        p_next: std::ptr::null(),
        format: ze::GRAPH_FORMAT_NATIVE,
        input_size: blob.len(),
        input: blob.as_ptr(),
        build_flags,
        flags: 0,
    }
}

/// A copy of the graph's native blob. The view from
/// `pfnGetNativeBinary2` is not kept: it dies with the graph.
pub fn copy(ext: &GraphExt, graph: Handle) -> Result<Vec<u8>, BlobError> {
    if let Some(get2) = ext.get_native_binary2() {
        let mut size = 0usize;
        let mut ptr: *const u8 = std::ptr::null();
        let rc = unsafe { get2(graph, &mut size, &mut ptr) };
        if rc != 0 {
            return Err(BlobError::Runtime(format!("npu: pfnGetNativeBinary2 failed with 0x{rc:08x}")));
        }
        return owned(size, ptr, "pfnGetNativeBinary2");
    }
    let Some(get) = ext.get_native_binary() else {
        return Err(BlobError::Unsupported(
            "npu: the device lists ZE_GRAPH_FORMAT_NATIVE and the graph extension has no pfnGetNativeBinary".into(),
        ));
    };
    let mut size = 0usize;
    let rc = unsafe { get(graph, &mut size, std::ptr::null_mut()) };
    if rc != 0 {
        return Err(BlobError::Runtime(format!("npu: pfnGetNativeBinary failed with 0x{rc:08x}")));
    }
    if size == 0 {
        return Err(BlobError::Runtime("npu: pfnGetNativeBinary reported an empty blob".into()));
    }
    let mut buf = Vec::new();
    if buf.try_reserve_exact(size).is_err() {
        return Err(BlobError::Runtime(format!("npu: the native blob is {size} bytes")));
    }
    buf.resize(size, 0);
    let rc = unsafe { get(graph, &mut size, buf.as_mut_ptr()) };
    if rc != 0 {
        return Err(BlobError::Runtime(format!("npu: pfnGetNativeBinary failed with 0x{rc:08x}")));
    }
    if size > buf.len() {
        return Err(BlobError::Runtime("npu: pfnGetNativeBinary grew the blob between calls".into()));
    }
    buf.truncate(size);
    if buf.is_empty() {
        return Err(BlobError::Runtime("npu: pfnGetNativeBinary returned an empty blob".into()));
    }
    Ok(buf)
}

fn owned(size: usize, ptr: *const u8, what: &str) -> Result<Vec<u8>, BlobError> {
    if size == 0 || ptr.is_null() {
        return Err(BlobError::Runtime(format!("npu: {what} returned an empty blob")));
    }
    let mut buf = Vec::new();
    if buf.try_reserve_exact(size).is_err() {
        return Err(BlobError::Runtime(format!("npu: the native blob is {size} bytes")));
    }
    buf.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, size) });
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    static BLOB: [u8; 4] = [0x7f, b'E', b'L', b'F'];

    unsafe extern "C" fn binary2(_graph: Handle, size: *mut usize, out: *mut *const u8) -> ze::Status {
        unsafe {
            *size = BLOB.len();
            *out = BLOB.as_ptr();
        }
        0
    }

    unsafe extern "C" fn binary2_empty(_graph: Handle, size: *mut usize, out: *mut *const u8) -> ze::Status {
        unsafe {
            *size = 0;
            *out = std::ptr::null();
        }
        0
    }

    unsafe extern "C" fn binary2_fails(_graph: Handle, _size: *mut usize, _out: *mut *const u8) -> ze::Status {
        0x7800_0004
    }

    unsafe extern "C" fn binary1(_graph: Handle, size: *mut usize, out: *mut u8) -> ze::Status {
        unsafe {
            if out.is_null() {
                *size = BLOB.len();
                return 0;
            }
            if *size < BLOB.len() {
                return 0x7800_0005;
            }
            std::ptr::copy_nonoverlapping(BLOB.as_ptr(), out, BLOB.len());
            *size = BLOB.len();
        }
        0
    }

    fn ext(version: u32, table: &ze::GraphDdi) -> GraphExt {
        unsafe { GraphExt::new(table, version) }
    }

    #[test]
    fn a_native_descriptor_names_the_blob_and_the_format() {
        let blob = [1u8, 2, 3, 4];
        let flags = c"";
        let d = descriptor(&blob, flags.as_ptr());
        assert_eq!(d.format, ze::GRAPH_FORMAT_NATIVE);
        assert_eq!(d.input_size, 4);
        assert_eq!(d.input, blob.as_ptr());
        assert!(!d.build_flags.is_null());
        assert_eq!(unsafe { std::ffi::CStr::from_ptr(d.build_flags) }, flags);
        assert_eq!(d.flags, 0);
    }

    #[test]
    fn extension_1_7_copies_the_driver_view() {
        let mut table: ze::GraphDdi = unsafe { std::mem::zeroed() };
        table.pfn_get_native_binary2 = Some(binary2);
        let bytes = copy(&ext(ze::version(1, 7), &table), std::ptr::null_mut()).unwrap();
        assert_eq!(bytes, BLOB);
    }

    #[test]
    fn an_older_extension_uses_the_caller_buffer() {
        let mut table: ze::GraphDdi = unsafe { std::mem::zeroed() };
        table.pfn_get_native_binary2 = Some(binary2_fails);
        table.pfn_get_native_binary = Some(binary1);
        let bytes = copy(&ext(ze::version(1, 6), &table), std::ptr::null_mut()).unwrap();
        assert_eq!(bytes, BLOB, "1.6 must not read pfnGetNativeBinary2");
    }

    #[test]
    fn an_empty_blob_and_a_driver_error_are_refused() {
        let mut table: ze::GraphDdi = unsafe { std::mem::zeroed() };
        table.pfn_get_native_binary2 = Some(binary2_empty);
        let e = copy(&ext(ze::version(1, 7), &table), std::ptr::null_mut()).unwrap_err();
        assert!(matches!(e, BlobError::Runtime(ref m) if m.contains("empty blob")), "{e:?}");
        table.pfn_get_native_binary2 = Some(binary2_fails);
        let e = copy(&ext(ze::version(1, 7), &table), std::ptr::null_mut()).unwrap_err();
        assert!(matches!(e, BlobError::Runtime(ref m) if m.contains("0x78000004")), "{e:?}");
    }

    #[test]
    fn a_table_with_neither_export_is_unsupported() {
        let table: ze::GraphDdi = unsafe { std::mem::zeroed() };
        let e = copy(&ext(ze::version(1, 5), &table), std::ptr::null_mut()).unwrap_err();
        assert!(matches!(e, BlobError::Unsupported(ref m) if m.contains("pfnGetNativeBinary")), "{e:?}");
    }
}
