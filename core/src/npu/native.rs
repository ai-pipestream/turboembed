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

/// Whether the extension exports a compiled graph's blob, which is what
/// a NATIVE load needs. The device's graphFormatsSupported is not part
/// of it: Intel's driver clears bit 0x1 whenever its compiler loads.
pub fn exportable(ext: &GraphExt) -> bool {
    ext.get_native_binary2().is_some() || ext.get_native_binary().is_some()
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
            "npu: the graph extension has no pfnGetNativeBinary2 or pfnGetNativeBinary".into(),
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

    /// Either export makes a NATIVE load possible; neither leaves the
    /// compiled IR. The slots are never called here, only read.
    #[test]
    fn a_native_load_needs_an_export() {
        let table = |slot: Option<usize>| {
            let mut words = vec![0u64; size_of::<ze::GraphDdi>().div_ceil(8)];
            if let Some(at) = slot {
                words[at / 8] = 0xABAB_ABAB_ABAB_ABAB;
            }
            words
        };
        let one = std::mem::offset_of!(ze::GraphDdi, pfn_get_native_binary);
        let two = std::mem::offset_of!(ze::GraphDdi, pfn_get_native_binary2);
        let at = |words: &Vec<u64>, v| unsafe { GraphExt::new(words.as_ptr() as *const ze::GraphDdi, v) };
        let none = table(None);
        assert!(!exportable(&at(&none, ze::version(1, 20))));
        let first = table(Some(one));
        assert!(exportable(&at(&first, ze::version(1, 0))));
        let second = table(Some(two));
        assert!(exportable(&at(&second, ze::version(1, 7))));
        assert!(!exportable(&at(&second, ze::version(1, 6))), "1.7's slot is past a 1.6 table");
    }
}
