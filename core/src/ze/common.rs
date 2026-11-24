//! What the Intel GPU backend and the NPU backend do the same way around
//! their Level Zero drivers: open them once, name them, and log.

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::{turbo_log_fn, turbo_text};

/// The driver `cell` keeps, opened by `open` the first time: `Ok(None)`
/// when there is none, `Err` when it is there and answers wrongly. The
/// answer, an `Err` too, is kept for the life of the process.
pub(crate) fn opened<D>(
    cell: &'static OnceLock<Result<Option<D>, String>>,
    open: fn() -> Result<Option<D>, String>,
) -> Result<Option<&'static D>, &'static str> {
    match cell.get_or_init(open) {
        Ok(d) => Ok(d.as_ref()),
        Err(e) => Err(e),
    }
}

/// Level Zero only promises that driverVersion increases. Intel's driver
/// packs major, minor and build into it; another vendor's is shown as is.
pub(crate) fn driver_version(vendor_id: u32, v: u32) -> String {
    match vendor_id {
        0x8086 => format!("{}.{}.{}", v >> 24, (v >> 16) & 0xff, v & 0xffff),
        _ => v.to_string(),
    }
}

/// The name of the PCI vendor `vendor_id`: Intel's, or the id in hex.
pub(crate) fn vendor(vendor_id: u32) -> String {
    match vendor_id {
        0x8086 => "Intel".to_owned(),
        v => format!("0x{v:04x}"),
    }
}

/// Passes `message` at `level` to the caller's log, when it gave one.
pub(crate) fn say(log: turbo_log_fn, user_data: *mut c_void, level: u32, message: &str) {
    if let Some(f) = log {
        let t = turbo_text { ptr: message.as_ptr() as *const _, len: message.len() as u64 };
        unsafe { f(user_data, level, t) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intel_versions_unpack_and_other_vendors_show_as_is() {
        assert_eq!(driver_version(0x8086, (1 << 24) | (6 << 16) | 31294), "1.6.31294");
        assert_eq!(driver_version(0x8086, 0x0103_909c), "1.3.37020");
        assert_eq!(driver_version(0x10de, 0x0103_909c), "17010844");
        assert_eq!(vendor(0x8086), "Intel");
        assert_eq!(vendor(0x10de), "0x10de");
    }

    #[test]
    fn a_driver_is_opened_once_and_its_failure_is_kept() {
        static NONE: OnceLock<Result<Option<u8>, String>> = OnceLock::new();
        static FAILED: OnceLock<Result<Option<u8>, String>> = OnceLock::new();
        static FOUND: OnceLock<Result<Option<u8>, String>> = OnceLock::new();
        assert_eq!(opened(&NONE, || Ok(None)), Ok(None));
        assert_eq!(opened(&FAILED, || Err("no devices".into())), Err("no devices"));
        assert_eq!(opened(&FAILED, || Ok(Some(1))), Err("no devices"));
        assert_eq!(opened(&FOUND, || Ok(Some(7))), Ok(Some(&7)));
    }

    #[test]
    fn say_reaches_the_log_with_its_level_and_text() {
        unsafe extern "C" fn log(user_data: *mut c_void, level: u32, message: turbo_text) {
            let seen = unsafe { &mut *(user_data as *mut Vec<(u32, String)>) };
            let bytes = unsafe { std::slice::from_raw_parts(message.ptr as *const u8, message.len as usize) };
            seen.push((level, String::from_utf8(bytes.to_vec()).unwrap()));
        }
        let mut seen: Vec<(u32, String)> = Vec::new();
        say(Some(log), &mut seen as *mut _ as *mut c_void, 2, "device 0 ready");
        say(None, std::ptr::null_mut(), 2, "nobody hears this");
        assert_eq!(seen, [(2, "device 0 ready".to_string())]);
    }
}
