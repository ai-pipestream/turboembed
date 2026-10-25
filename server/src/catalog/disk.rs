//! Free space on the file system that holds a directory.

use std::io;
use std::path::Path;

/// The bytes an unprivileged process may still write on the file system
/// holding `dir`.
#[cfg(unix)]
pub(super) fn free_bytes(dir: &Path) -> io::Result<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c = CString::new(dir.as_os_str().as_bytes()).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    // SAFETY: `c` is a NUL-terminated path and `s` a statvfs the call fills.
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        return Err(io::Error::last_os_error());
    }
    #[allow(clippy::unnecessary_cast)]
    Ok((s.f_bavail as u64).saturating_mul(s.f_frsize as u64))
}

#[cfg(windows)]
pub(super) fn free_bytes(dir: &Path) -> io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        #[allow(non_snake_case)]
        fn GetDiskFreeSpaceExW(dir: *const u16, caller: *mut u64, total: *mut u64, free: *mut u64) -> i32;
    }
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut caller = 0u64;
    // SAFETY: `wide` is a NUL-terminated path; the call writes the first
    // out-parameter and accepts null for the other two.
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut caller, std::ptr::null_mut(), std::ptr::null_mut()) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(caller)
}

#[cfg(not(any(unix, windows)))]
pub(super) fn free_bytes(_dir: &Path) -> io::Result<u64> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "free space is not known on this platform"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_temp_directory_has_some_free_space() {
        assert!(super::free_bytes(&std::env::temp_dir()).unwrap() > 0);
    }
}
