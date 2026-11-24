//! Which runtime device a model loads on.
//!
//! A runtime index is a position: every device of each linked backend,
//! in the order `turbo_version()` lists the backends, the CPU last. When
//! a linked backend lists no device (no driver, no device node, a driver
//! older than the runtime), every later device moves up, so `device=0`
//! that meant a GPU names the next backend's device or the CPU. A device
//! named by its backend and its ordinal there (`cuda:0`, `cpu`) names
//! one device or none, and none is a refusal, never another device. A
//! typed index is refused when a backend linked ahead of its device
//! lists none, as the number then names a different device than on a
//! host where every backend lists its devices. A backend that lists fewer
//! devices than another host's is not seen here: name the device.

use crate::api::{Failure, Runtime, TURBO_E_DEVICE_NOT_FOUND};
use crate::config::Device;

/// The backends a device can be named by, as `turbo_device_info.backend`
/// and `turbo_version()` spell them.
pub const BACKENDS: [&str; 6] = ["cpu", "cuda", "levelzero", "metal", "hailo", "npu"];

/// A device named by its backend and its ordinal within that backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Named {
    /// One of [`BACKENDS`].
    pub backend: &'static str,
    pub ordinal: u32,
}

impl std::fmt::Display for Named {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.backend == "cpu" {
            return f.write_str("cpu");
        }
        write!(f, "{}:{}", self.backend, self.ordinal)
    }
}

impl Named {
    /// `cpu`, a backend's name (its device 0) or `backend:N`; None for
    /// anything else, `cpu:N` included, as there is one CPU device.
    pub fn parse(s: &str) -> Option<Named> {
        let (name, ordinal) = match s.split_once(':') {
            None => (s, 0),
            Some(("cpu", _)) => return None,
            Some((name, n)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => (name, n.parse().ok()?),
            Some(_) => return None,
        };
        let backend = BACKENDS.into_iter().find(|b| *b == name)?;
        Some(Named { backend, ordinal })
    }
}

/// The text a refused device setting names as what is accepted.
pub const ACCEPTED: &str = "select, cpu, cuda[:N], levelzero[:N], metal[:N], hailo[:N], npu[:N] or a runtime index";

/// One listed device: its runtime index, backend and ordinal there.
struct Listed {
    index: u32,
    backend: String,
    ordinal: u32,
}

/// Every device the runtime lists.
fn listed(rt: &Runtime) -> Result<Vec<Listed>, Failure> {
    let mut out = Vec::new();
    for index in 0..rt.device_count()? {
        let info = rt.device_info(index)?;
        out.push(Listed { index, backend: crate::api::field(&info.backend), ordinal: info.ordinal });
    }
    Ok(out)
}

/// The backends linked into this build, in device order.
fn linked() -> Vec<String> {
    crate::api::version().split_whitespace().skip(1).map(str::to_owned).collect()
}

/// What the runtime lists, as `cuda:0, cpu`, for a refusal's message.
fn listing(devices: &[Listed]) -> String {
    let names: Vec<String> = devices
        .iter()
        .map(|d| if d.backend == "cpu" { "cpu".into() } else { format!("{}:{}", d.backend, d.ordinal) })
        .collect();
    if names.is_empty() { "no device".into() } else { names.join(", ") }
}

fn not_found(message: String) -> Failure {
    Failure::new(TURBO_E_DEVICE_NOT_FOUND, 0, message)
}

/// The runtime index `device` names; `task` is what `select` picks for.
/// A named device the runtime does not list is DEVICE_NOT_FOUND, and so
/// is a typed position when a backend linked ahead of its device lists
/// none, as its number has moved.
pub fn resolve(rt: &Runtime, device: Device, task: u32) -> Result<u32, Failure> {
    match device {
        Device::Index(i) => Ok(i),
        Device::Select => rt.select(task),
        Device::Named(n) => {
            let devices = listed(rt)?;
            if let Some(d) = devices.iter().find(|d| d.backend == n.backend && d.ordinal == n.ordinal) {
                return Ok(d.index);
            }
            let linked = linked();
            if !linked.iter().any(|b| b == n.backend) {
                let msg = format!(
                    "device {n}: the {} backend is not in this build (linked: {})",
                    n.backend,
                    linked.join(" ")
                );
                return Err(not_found(msg));
            }
            let msg = format!(
                "device {n}: the {} backend lists no such device; the runtime lists {}",
                n.backend,
                listing(&devices)
            );
            Err(not_found(msg))
        }
        Device::Position(i) => {
            let devices = listed(rt)?;
            // Only an empty backend linked before index i's own backend
            // moves i; past the end of the list, any empty one may have.
            let linked = linked();
            let own = devices.iter().find(|d| d.index == i).map(|d| d.backend.as_str());
            let before = own.and_then(|b| linked.iter().position(|l| l == b)).unwrap_or(linked.len());
            let empty: Vec<&str> = linked[..before]
                .iter()
                .filter(|b| !devices.iter().any(|d| d.backend == **b))
                .map(String::as_str)
                .collect();
            if !empty.is_empty() {
                let msg = format!(
                    "device {i}: the {} backend lists no device, so the runtime's device numbers have moved \
                     (it lists {}); name the device instead, as cpu or backend:N",
                    empty.join(" and "),
                    listing(&devices)
                );
                return Err(not_found(msg));
            }
            Ok(i)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Named;

    #[test]
    fn names_parse() {
        let n = |backend, ordinal| Some(Named { backend, ordinal });
        assert_eq!(Named::parse("cpu"), n("cpu", 0));
        assert_eq!(Named::parse("cuda"), n("cuda", 0));
        assert_eq!(Named::parse("cuda:3"), n("cuda", 3));
        assert_eq!(Named::parse("levelzero:1"), n("levelzero", 1));
        assert_eq!(Named::parse("metal"), n("metal", 0));
        assert_eq!(Named::parse("hailo:0"), n("hailo", 0));
        assert_eq!(Named::parse("npu:2"), n("npu", 2));
        for bad in ["", "0", "gpu", "cuda:", "cuda:x", "cuda:-1", "cuda:+1", "cpu:0", "CUDA:0", "cuda:0:1"] {
            assert_eq!(Named::parse(bad), None, "{bad:?} parsed");
        }
        assert_eq!(Named::parse("cuda:3").unwrap().to_string(), "cuda:3");
        assert_eq!(Named::parse("cpu").unwrap().to_string(), "cpu");
    }
}
