//! Shared FFI error, text, and array-boundary checks.

use super::*;

// ---- Calling convention ----------------------------------------------------

/// Run `f` behind the header's rules: the error struct's size is checked
/// before anything else, a panic stops at this boundary, and the status is
/// both returned and written to `err`.
pub(super) unsafe fn call(err: *mut turbo_error, f: impl FnOnce() -> Result<()>) -> i32 {
    if let Some(e) = unsafe { err.as_ref() }
        && e.struct_size as usize != size_of::<turbo_error>()
    {
        return INVALID_STRUCT_SIZE;
    }
    let outcome = catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let why = p
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| p.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        Err(Error::new(PANIC, format!("panic in libturbo: {why}")))
    });
    let e = match outcome {
        Ok(()) => Error::new(status::OK, ""),
        Err(e) => e,
    };
    if let Some(out) = unsafe { err.as_mut() } {
        out.code = e.code;
        out.field = e.field;
        write_str(&mut out.message, &e.message);
    }
    e.code
}

/// Copy `s` into a NUL-terminated buffer, cut at a character boundary.
pub(crate) fn write_str(dst: &mut [c_char], s: &str) {
    let mut n = s.len().min(dst.len() - 1);
    while !s.is_char_boundary(n) {
        n -= 1;
    }
    for (d, b) in dst.iter_mut().zip(&s.as_bytes()[..n]) {
        *d = *b as c_char;
    }
    dst[n] = 0;
}

pub(super) unsafe fn text<'a, W: std::fmt::Display + ?Sized>(t: turbo_text, what: &W) -> Result<&'a str> {
    if t.len == 0 {
        return Ok("");
    }
    if t.ptr.is_null() {
        return Err(Error::new(INVALID_ARGUMENT, format!("{what}: NULL with length {}", t.len)));
    }
    let len = text_len(t, what)?;
    let bytes = unsafe { std::slice::from_raw_parts(t.ptr.cast::<u8>(), len) };
    std::str::from_utf8(bytes).map_err(|e| Error::new(INVALID_UTF8, format!("{what}: {e}")))
}

pub(super) unsafe fn out_ptr<'a, T>(p: *mut T, what: &str) -> Result<&'a mut T> {
    unsafe { p.as_mut() }.ok_or_else(|| Error::new(INVALID_ARGUMENT, format!("{what} is NULL")))
}

pub(super) fn sized(struct_size: u32, want: usize, what: &str) -> Result<()> {
    if struct_size as usize != want {
        return Err(Error::new(
            INVALID_STRUCT_SIZE,
            format!("{what}.struct_size is {struct_size}, this library knows {want}"),
        ));
    }
    Ok(())
}

// ---- Library -----------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn turbo_version() -> *const c_char {
    VERSION.get_or_init(version).as_ptr()
}

/// The version and the backends linked into this build, in device order.
pub(super) static VERSION: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();

pub(super) fn version() -> std::ffi::CString {
    let mut v = String::from(record::library_version());
    for b in backend::linked() {
        v.push(' ');
        v.push_str(b.name());
    }
    std::ffi::CString::new(v).expect("backend names hold no NUL")
}

pub(crate) fn new_error() -> turbo_error {
    turbo_error {
        struct_size: size_of::<turbo_error>() as u32,
        code: 0,
        field: 0,
        message: [0; TURBO_ERROR_MESSAGE_LEN],
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn turbo_status_name(code: i32) -> *const c_char {
    status::name(code).as_ptr()
}

// ---- Runtime -------------------------------------------------------------------

/// Validate a foreign array's extent before converting counts or making slices.
pub(super) fn array_len<T>(count: u64, what: &str) -> Result<usize> {
    let size = size_of::<T>().max(1) as u64;
    if count > isize::MAX as u64 / size {
        return Err(Error::new(INVALID_SHAPE, format!("{what}: {count} elements exceed the addressable slice size")));
    }
    Ok(count as usize)
}

/// Check text metadata before reserving token buffers or dispatching work.
pub(super) fn text_len(t: turbo_text, what: &(impl std::fmt::Display + ?Sized)) -> Result<usize> {
    if t.len > isize::MAX as u64 || (t.len != 0 && t.ptr.is_null()) {
        return Err(Error::new(INVALID_ARGUMENT, format!("{what}: invalid text pointer or length {}", t.len)));
    }
    Ok(t.len as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_extents_check_bytes_before_conversion() {
        let limit = isize::MAX as u64 / 4;
        assert_eq!(array_len::<i32>(limit, "ids").unwrap() as u64, limit);
        assert_eq!(array_len::<i32>(limit + 1, "ids").unwrap_err().code, INVALID_SHAPE);
        assert!(array_len::<[u32; 2]>(isize::MAX as u64 / 8 + 1, "spans").is_err());
        assert!(array_len::<i32>(u32::MAX as u64 * u32::MAX as u64, "rows").is_err());
        assert_eq!(array_len::<i32>(0, "empty").unwrap(), 0);
    }

    #[test]
    fn text_metadata_checks_do_not_dereference_foreign_pointers() {
        let empty = turbo_text { ptr: std::ptr::null(), len: 0 };
        assert_eq!(text_len(empty, "text").unwrap(), 0);
        assert!(text_len(turbo_text { len: 1, ..empty }, "text").is_err());
        let view = turbo_text { ptr: std::ptr::dangling(), len: u64::MAX };
        assert!(text_len(view, "text").is_err());
    }
}
