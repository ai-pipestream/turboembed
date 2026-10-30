//! Apache Arrow C Data Interface (`ArrowSchema`, `ArrowArray`, and
//! `ArrowArrayStream`) export for `turbo_analysis_result`.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::{null, null_mut};

use super::{
    AnalysisResultInner, TURBO_ANALYSIS_LAYER_CHUNKS, TURBO_SPAN_LABEL, TURBO_SPAN_VALUE, analysis_result_handle,
    turbo_analysis_result, turbo_span,
};
use crate::ffi::{call, out_ptr};
use crate::status::{CAPACITY, Error, INVALID_ARGUMENT, Result};
use crate::turbo_error;

pub const ARROW_FLAG_DICTIONARY_ORDERED: i64 = 1;
pub const ARROW_FLAG_NULLABLE: i64 = 2;
pub const ARROW_FLAG_MAP_KEYS_SORTED: i64 = 4;

const EINVAL: c_int = 22;

/// Apache Arrow C Data Interface schema descriptor.
#[repr(C)]
pub struct ArrowSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

/// Apache Arrow C Data Interface array descriptor.
#[repr(C)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const c_void,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    pub private_data: *mut c_void,
}

/// Apache Arrow C Stream Interface descriptor.
#[repr(C)]
pub struct ArrowArrayStream {
    pub get_schema: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowSchema) -> c_int>,
    pub get_next: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowArray) -> c_int>,
    pub get_last_error: Option<unsafe extern "C" fn(*mut ArrowArrayStream) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArrayStream)>,
    pub private_data: *mut c_void,
}

const _: () = assert!(size_of::<ArrowSchema>() == 72);
const _: () = assert!(align_of::<ArrowSchema>() == 8);
const _: () = assert!(size_of::<ArrowArray>() == 80);
const _: () = assert!(align_of::<ArrowArray>() == 8);
const _: () = assert!(size_of::<ArrowArrayStream>() == 40);
const _: () = assert!(align_of::<ArrowArrayStream>() == 8);

const COLUMN_SPECS: &[(&CStr, &CStr)] = &[
    (c"layer_kind", c"I"),
    (c"item_index", c"I"),
    (c"fragment_index", c"I"),
    (c"view_id", c"I"),
    (c"classification", c"I"),
    (c"view_byte_start", c"I"),
    (c"view_byte_end", c"I"),
    (c"view_utf16_start", c"I"),
    (c"view_utf16_end", c"I"),
    (c"view_codepoint_start", c"I"),
    (c"view_codepoint_end", c"I"),
    (c"source_byte_start", c"I"),
    (c"source_byte_end", c"I"),
    (c"source_utf16_start", c"I"),
    (c"source_utf16_end", c"I"),
    (c"source_codepoint_start", c"I"),
    (c"source_codepoint_end", c"I"),
    (c"token_start", c"I"),
    (c"token_end", c"I"),
    (c"layer", c"u"),
    (c"instance", c"u"),
    (c"label", c"u"),
    (c"value", c"u"),
    (c"prob", c"g"),
];

struct StructSchemaPrivate {
    children: Box<[*mut ArrowSchema]>,
}

enum LeafStorage {
    U32 { _values: Vec<u32>, buffers: [*const c_void; 2] },
    F64 { _values: Vec<f64>, buffers: [*const c_void; 2] },
    Utf8 { _offsets: Vec<i32>, _data: Vec<u8>, buffers: [*const c_void; 3] },
}

struct StructArrayPrivate {
    buffers: [*const c_void; 1],
    children: Box<[*mut ArrowArray]>,
}

unsafe extern "C" fn release_leaf_schema(schema: *mut ArrowSchema) {
    if let Some(s) = unsafe { schema.as_mut() } {
        s.release = None;
        s.private_data = null_mut();
    }
}

unsafe extern "C" fn release_struct_schema(schema: *mut ArrowSchema) {
    let Some(s) = (unsafe { schema.as_mut() }) else {
        return;
    };
    if s.release.is_none() {
        return;
    }
    if !s.private_data.is_null() {
        let priv_box = unsafe { Box::from_raw(s.private_data.cast::<StructSchemaPrivate>()) };
        for &child_ptr in &priv_box.children {
            if !child_ptr.is_null() {
                unsafe {
                    if let Some(rel) = (*child_ptr).release {
                        rel(child_ptr);
                    }
                    drop(Box::from_raw(child_ptr));
                }
            }
        }
    }
    s.children = null_mut();
    s.n_children = 0;
    s.private_data = null_mut();
    s.release = None;
}

unsafe extern "C" fn release_leaf_array(array: *mut ArrowArray) {
    let Some(a) = (unsafe { array.as_mut() }) else {
        return;
    };
    if a.release.is_none() {
        return;
    }
    if !a.private_data.is_null() {
        drop(unsafe { Box::from_raw(a.private_data.cast::<LeafStorage>()) });
    }
    a.buffers = null_mut();
    a.n_buffers = 0;
    a.private_data = null_mut();
    a.release = None;
}

unsafe extern "C" fn release_struct_array(array: *mut ArrowArray) {
    let Some(a) = (unsafe { array.as_mut() }) else {
        return;
    };
    if a.release.is_none() {
        return;
    }
    if !a.private_data.is_null() {
        let priv_box = unsafe { Box::from_raw(a.private_data.cast::<StructArrayPrivate>()) };
        for &child_ptr in &priv_box.children {
            if !child_ptr.is_null() {
                unsafe {
                    if let Some(rel) = (*child_ptr).release {
                        rel(child_ptr);
                    }
                    drop(Box::from_raw(child_ptr));
                }
            }
        }
    }
    a.buffers = null_mut();
    a.n_buffers = 0;
    a.children = null_mut();
    a.n_children = 0;
    a.private_data = null_mut();
    a.release = None;
}

fn build_schema() -> ArrowSchema {
    let mut children = Vec::with_capacity(COLUMN_SPECS.len());
    for &(name, format) in COLUMN_SPECS {
        let child = Box::new(ArrowSchema {
            format: format.as_ptr(),
            name: name.as_ptr(),
            metadata: null(),
            flags: 0,
            n_children: 0,
            children: null_mut(),
            dictionary: null_mut(),
            release: Some(release_leaf_schema),
            private_data: null_mut(),
        });
        children.push(Box::into_raw(child));
    }
    let mut priv_box = Box::new(StructSchemaPrivate { children: children.into_boxed_slice() });
    let children_ptr = priv_box.children.as_mut_ptr();
    let n_children = priv_box.children.len() as i64;
    ArrowSchema {
        format: c"+s".as_ptr(),
        name: c"".as_ptr(),
        metadata: null(),
        flags: 0,
        n_children,
        children: children_ptr,
        dictionary: null_mut(),
        release: Some(release_struct_schema),
        private_data: Box::into_raw(priv_box).cast::<c_void>(),
    }
}

fn make_u32_leaf(values: Vec<u32>) -> *mut ArrowArray {
    let len = values.len() as i64;
    let ptr = values.as_ptr().cast::<c_void>();
    let mut storage = Box::new(LeafStorage::U32 { _values: values, buffers: [null(), ptr] });
    let buffers_ptr = match &mut *storage {
        LeafStorage::U32 { buffers, .. } => buffers.as_mut_ptr(),
        _ => unreachable!(),
    };
    Box::into_raw(Box::new(ArrowArray {
        length: len,
        null_count: 0,
        offset: 0,
        n_buffers: 2,
        n_children: 0,
        buffers: buffers_ptr,
        children: null_mut(),
        dictionary: null_mut(),
        release: Some(release_leaf_array),
        private_data: Box::into_raw(storage).cast::<c_void>(),
    }))
}

fn make_f64_leaf(values: Vec<f64>) -> *mut ArrowArray {
    let len = values.len() as i64;
    let ptr = values.as_ptr().cast::<c_void>();
    let mut storage = Box::new(LeafStorage::F64 { _values: values, buffers: [null(), ptr] });
    let buffers_ptr = match &mut *storage {
        LeafStorage::F64 { buffers, .. } => buffers.as_mut_ptr(),
        _ => unreachable!(),
    };
    Box::into_raw(Box::new(ArrowArray {
        length: len,
        null_count: 0,
        offset: 0,
        n_buffers: 2,
        n_children: 0,
        buffers: buffers_ptr,
        children: null_mut(),
        dictionary: null_mut(),
        release: Some(release_leaf_array),
        private_data: Box::into_raw(storage).cast::<c_void>(),
    }))
}

fn make_utf8_leaf(
    rows: &[&turbo_span],
    pool: &[u8],
    select: impl Fn(&turbo_span) -> (u32, u32),
) -> Result<*mut ArrowArray> {
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let mut data = Vec::new();
    offsets.push(0i32);
    for &sp in rows {
        let (off, len) = select(sp);
        let start = off as usize;
        let end = start.saturating_add(len as usize);
        if len > 0 && end <= pool.len() {
            data.extend_from_slice(&pool[start..end]);
        }
        let next_off = i32::try_from(data.len()).map_err(|_| Error::new(CAPACITY, "Arrow UTF-8 column exceeds i32"))?;
        offsets.push(next_off);
    }
    let len = rows.len() as i64;
    let off_ptr = offsets.as_ptr().cast::<c_void>();
    let data_ptr = data.as_ptr().cast::<c_void>();
    let mut storage =
        Box::new(LeafStorage::Utf8 { _offsets: offsets, _data: data, buffers: [null(), off_ptr, data_ptr] });
    let buffers_ptr = match &mut *storage {
        LeafStorage::Utf8 { buffers, .. } => buffers.as_mut_ptr(),
        _ => unreachable!(),
    };
    Ok(Box::into_raw(Box::new(ArrowArray {
        length: len,
        null_count: 0,
        offset: 0,
        n_buffers: 3,
        n_children: 0,
        buffers: buffers_ptr,
        children: null_mut(),
        dictionary: null_mut(),
        release: Some(release_leaf_array),
        private_data: Box::into_raw(storage).cast::<c_void>(),
    })))
}

fn build_array(res: &AnalysisResultInner, layer_kind: u32) -> Result<ArrowArray> {
    if layer_kind > TURBO_ANALYSIS_LAYER_CHUNKS {
        return Err(Error::new(INVALID_ARGUMENT, format!("unknown analysis layer_kind {layer_kind}")));
    }
    let rows: Vec<&turbo_span> = res.spans.iter().filter(|s| layer_kind == 0 || s.layer_kind == layer_kind).collect();
    let n = rows.len();

    let u32_col = |f: fn(&turbo_span) -> u32| -> *mut ArrowArray { make_u32_leaf(rows.iter().map(|s| f(s)).collect()) };

    let mut children = Vec::with_capacity(COLUMN_SPECS.len());
    children.push(u32_col(|s| s.layer_kind));
    children.push(u32_col(|s| s.item_index));
    children.push(u32_col(|s| s.fragment_index));
    children.push(u32_col(|s| s.view_id));
    children.push(u32_col(|s| s.classification));
    children.push(u32_col(|s| s.view_byte_start));
    children.push(u32_col(|s| s.view_byte_end));
    children.push(u32_col(|s| s.view_utf16_start));
    children.push(u32_col(|s| s.view_utf16_end));
    children.push(u32_col(|s| s.view_codepoint_start));
    children.push(u32_col(|s| s.view_codepoint_end));
    children.push(u32_col(|s| s.source_byte_start));
    children.push(u32_col(|s| s.source_byte_end));
    children.push(u32_col(|s| s.source_utf16_start));
    children.push(u32_col(|s| s.source_utf16_end));
    children.push(u32_col(|s| s.source_codepoint_start));
    children.push(u32_col(|s| s.source_codepoint_end));
    children.push(u32_col(|s| s.token_start));
    children.push(u32_col(|s| s.token_end));
    children.push(make_utf8_leaf(&rows, &res.pool, |s| (s.layer_offset, s.layer_len))?);
    children.push(make_utf8_leaf(&rows, &res.pool, |s| (s.instance_offset, s.instance_len))?);
    children.push(make_utf8_leaf(&rows, &res.pool, |s| {
        if s.flags & TURBO_SPAN_LABEL != 0 { (s.label_offset, s.label_len) } else { (0, 0) }
    })?);
    children.push(make_utf8_leaf(&rows, &res.pool, |s| {
        if s.flags & TURBO_SPAN_VALUE != 0 { (s.value_offset, s.value_len) } else { (0, 0) }
    })?);
    children.push(make_f64_leaf(rows.iter().map(|s| s.prob).collect()));

    let mut priv_box = Box::new(StructArrayPrivate { buffers: [null()], children: children.into_boxed_slice() });
    let buffers_ptr = priv_box.buffers.as_mut_ptr();
    let children_ptr = priv_box.children.as_mut_ptr();
    let n_children = priv_box.children.len() as i64;

    Ok(ArrowArray {
        length: n as i64,
        null_count: 0,
        offset: 0,
        n_buffers: 1,
        n_children,
        buffers: buffers_ptr,
        children: children_ptr,
        dictionary: null_mut(),
        release: Some(release_struct_array),
        private_data: Box::into_raw(priv_box).cast::<c_void>(),
    })
}

struct StreamPrivate {
    batch: Option<ArrowArray>,
}

unsafe extern "C" fn stream_get_schema(stream: *mut ArrowArrayStream, out: *mut ArrowSchema) -> c_int {
    let (Some(st), Some(out_ref)) = (unsafe { stream.as_ref() }, unsafe { out.as_mut() }) else {
        return EINVAL;
    };
    if st.release.is_none() || st.private_data.is_null() {
        return EINVAL;
    }
    *out_ref = build_schema();
    0
}

unsafe extern "C" fn stream_get_next(stream: *mut ArrowArrayStream, out: *mut ArrowArray) -> c_int {
    let (Some(st), Some(out_ref)) = (unsafe { stream.as_mut() }, unsafe { out.as_mut() }) else {
        return EINVAL;
    };
    if st.release.is_none() || st.private_data.is_null() {
        return EINVAL;
    }
    let priv_ref = unsafe { &mut *st.private_data.cast::<StreamPrivate>() };
    if let Some(batch) = priv_ref.batch.take() {
        *out_ref = batch;
    } else {
        *out_ref = unsafe { std::mem::zeroed() };
    }
    0
}

unsafe extern "C" fn stream_get_last_error(_stream: *mut ArrowArrayStream) -> *const c_char {
    null()
}

unsafe extern "C" fn release_stream(stream: *mut ArrowArrayStream) {
    let Some(st) = (unsafe { stream.as_mut() }) else {
        return;
    };
    if st.release.is_none() {
        return;
    }
    if !st.private_data.is_null() {
        let mut priv_box = unsafe { Box::from_raw(st.private_data.cast::<StreamPrivate>()) };
        if let Some(mut batch) = priv_box.batch.take()
            && let Some(rel) = batch.release
        {
            unsafe { rel(&mut batch) };
        }
    }
    st.get_schema = None;
    st.get_next = None;
    st.get_last_error = None;
    st.private_data = null_mut();
    st.release = None;
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_to_arrow(
    r: *const turbo_analysis_result,
    layer_kind: u32,
    out_schema: *mut ArrowSchema,
    out_array: *mut ArrowArray,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let schema_out = out_ptr(out_schema, "out_schema")?;
            *schema_out = std::mem::zeroed();
            let array_out = out_ptr(out_array, "out_array")?;
            *array_out = std::mem::zeroed();
            let r = analysis_result_handle(r)?;
            let array = build_array(&r.inner, layer_kind)?;
            *schema_out = build_schema();
            *array_out = array;
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analysis_result_to_arrow_stream(
    r: *const turbo_analysis_result,
    layer_kind: u32,
    out_stream: *mut ArrowArrayStream,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let stream_out = out_ptr(out_stream, "out_stream")?;
            *stream_out = std::mem::zeroed();
            let r = analysis_result_handle(r)?;
            let batch = build_array(&r.inner, layer_kind)?;
            let priv_box = Box::new(StreamPrivate { batch: Some(batch) });
            *stream_out = ArrowArrayStream {
                get_schema: Some(stream_get_schema),
                get_next: Some(stream_get_next),
                get_last_error: Some(stream_get_last_error),
                release: Some(release_stream),
                private_data: Box::into_raw(priv_box).cast::<c_void>(),
            };
            Ok(())
        })
    }
}
