//! The runtime: provider registry, device table, selection, and capability lookup.

use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};

use turbo_abi as abi;

use crate::bundle::Bundle;
use crate::error::{Error, Result};
use crate::provider::{Capability, DeviceInfo, Provider};
use crate::types::{DeviceKind, Modality, SelectPolicy, Task};

/// Log level passed to the sink.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum LogLevel {
    /// Error.
    Error = 0,
    /// Warning.
    Warn = 1,
    /// Informational.
    Info = 2,
    /// Debug.
    Debug = 3,
}

/// Log sink.
pub type LogSink = Arc<dyn Fn(LogLevel, &str) + Send + Sync>;

/// One entry in the runtime's device table.
#[derive(Clone, Debug)]
pub struct DeviceEntry {
    /// Index into the runtime's providers.
    pub provider_index: usize,
    /// Static info.
    pub info: DeviceInfo,
}

/// Runtime creation parameters.
#[derive(Clone, Default)]
pub struct RuntimeDesc {
    /// Skip the default provider search path.
    pub no_default_providers: bool,
    /// Provider libraries to load explicitly, in order.
    pub provider_paths: Vec<String>,
    /// Log sink.
    pub log: Option<LogSink>,
}

/// Device selector.
#[derive(Clone, Debug, Default)]
pub struct DeviceSelector {
    /// Policy.
    pub policy: SelectPolicy,
    /// Accepted kinds; empty = any non-CPU for AUTO, any for EXPLICIT.
    pub kinds: Vec<DeviceKind>,
    /// Ordinal within provider (EXPLICIT).
    pub ordinal: u32,
    /// Provider id, empty = any.
    pub provider_id: String,
    /// Vendor substring, empty = any.
    pub vendor: String,
}

/// Provider load failure recorded by the runtime.
#[derive(Clone, Debug)]
pub struct ProviderFailure {
    /// Provider id or library path.
    pub what: String,
    /// The error.
    pub error: Error,
}

/// Library instance.
pub struct Runtime {
    providers: Vec<Arc<dyn Provider>>,
    devices: Vec<DeviceEntry>,
    failures: Mutex<Vec<ProviderFailure>>,
    log: Option<LogSink>,
    // Keeps dynamically loaded provider libraries alive for the runtime's lifetime.
    // Populated by plugin loading (PLAN.md P1); unused until then.
    #[allow(dead_code)]
    libraries: Mutex<Vec<libloading::Library>>,
}

impl fmt::Debug for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Runtime")
            .field("providers", &self.providers.iter().map(|p| p.id()).collect::<Vec<_>>())
            .field("devices", &self.devices.len())
            .finish()
    }
}

impl Runtime {
    /// Create a runtime with the given statically registered providers.
    /// Dynamic provider loading is added in P1; explicit `provider_paths`
    /// currently fail with `TURBO_E_NOT_IMPLEMENTED` rather than being ignored.
    pub fn new(desc: RuntimeDesc, static_providers: Vec<Arc<dyn Provider>>) -> Result<Arc<Self>> {
        let mut runtime = Self {
            providers: Vec::new(),
            devices: Vec::new(),
            failures: Mutex::new(Vec::new()),
            log: desc.log.clone(),
            libraries: Mutex::new(Vec::new()),
        };
        for provider in static_providers {
            runtime.register(provider);
        }
        if !desc.provider_paths.is_empty() {
            return Err(Error::not_implemented("turbo_runtime_desc.provider_paths (dynamic provider loading)"));
        }
        let _ = desc.no_default_providers;
        Ok(Arc::new(runtime))
    }

    /// Register a provider and probe its devices. A probe failure is recorded
    /// and logged; the provider stays registered with zero devices so the
    /// failure is visible through [`Runtime::failures`].
    pub fn register(&mut self, provider: Arc<dyn Provider>) {
        let index = self.providers.len();
        match provider.devices() {
            Ok(list) => {
                for info in list {
                    if info.provider_id != provider.id() {
                        self.record_failure(
                            provider.id(),
                            Error::internal(format!(
                                "provider `{}` reported a device with provider_id `{}`",
                                provider.id(),
                                info.provider_id
                            )),
                        );
                        continue;
                    }
                    self.devices.push(DeviceEntry { provider_index: index, info });
                }
            }
            Err(e) => self.record_failure(provider.id(), e),
        }
        self.providers.push(provider);
    }

    fn record_failure(&self, what: &str, error: Error) {
        self.log(LogLevel::Error, &format!("provider `{what}`: {error}"));
        self.failures.lock().unwrap_or_else(|p| p.into_inner()).push(ProviderFailure { what: what.to_string(), error });
    }

    /// Emit to the log sink, if any.
    pub fn log(&self, level: LogLevel, message: &str) {
        if let Some(sink) = &self.log {
            sink(level, message);
        }
    }

    /// Provider failures recorded during registration.
    pub fn failures(&self) -> Vec<ProviderFailure> {
        self.failures.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Registered providers.
    pub fn providers(&self) -> &[Arc<dyn Provider>] {
        &self.providers
    }

    /// Device table.
    pub fn devices(&self) -> &[DeviceEntry] {
        &self.devices
    }

    /// Device by index.
    pub fn device(&self, index: u32) -> Result<&DeviceEntry> {
        self.devices.get(index as usize).ok_or_else(|| {
            Error::device_not_found(format!(
                "device index {index} is out of range; {} device(s) enumerated",
                self.devices.len()
            ))
        })
    }

    /// Provider owning a device.
    pub fn provider_for(&self, index: u32) -> Result<&Arc<dyn Provider>> {
        let entry = self.device(index)?;
        Ok(&self.providers[entry.provider_index])
    }

    /// Select a device. AUTO returns the first non-CPU device matching the
    /// filters (never a CPU); EXPLICIT requires provider id + ordinal (and
    /// kind, if given) to match exactly.
    pub fn select(&self, sel: &DeviceSelector) -> Result<u32> {
        let matches_filters = |info: &DeviceInfo| {
            (sel.provider_id.is_empty() || info.provider_id == sel.provider_id)
                && (sel.vendor.is_empty() || info.vendor.contains(&sel.vendor))
                && (sel.kinds.is_empty() || sel.kinds.contains(&info.kind))
        };
        match sel.policy {
            SelectPolicy::Auto => {
                let found = self
                    .devices
                    .iter()
                    .enumerate()
                    .find(|(_, d)| d.info.kind != DeviceKind::Cpu && matches_filters(&d.info));
                match found {
                    Some((i, _)) => Ok(i as u32),
                    None => Err(Error::device_not_found(format!(
                        "AUTO found no accelerator{}; CPU is never selected automatically. Devices: {}",
                        self.describe_filters(sel),
                        self.describe_devices()
                    ))),
                }
            }
            SelectPolicy::Explicit => {
                if sel.provider_id.is_empty() {
                    return Err(Error::invalid_argument("EXPLICIT selection requires provider_id").with_field(5));
                }
                let found = self
                    .devices
                    .iter()
                    .enumerate()
                    .find(|(_, d)| d.info.ordinal == sel.ordinal && matches_filters(&d.info));
                match found {
                    Some((i, _)) => Ok(i as u32),
                    None => Err(Error::device_not_found(format!(
                        "no device with provider `{}` ordinal {}{}. Devices: {}",
                        sel.provider_id,
                        sel.ordinal,
                        self.describe_filters(sel),
                        self.describe_devices()
                    ))),
                }
            }
        }
    }

    fn describe_filters(&self, sel: &DeviceSelector) -> String {
        let mut parts = Vec::new();
        if !sel.kinds.is_empty() {
            parts.push(format!("kinds {:?}", sel.kinds));
        }
        if !sel.provider_id.is_empty() {
            parts.push(format!("provider `{}`", sel.provider_id));
        }
        if !sel.vendor.is_empty() {
            parts.push(format!("vendor containing `{}`", sel.vendor));
        }
        if parts.is_empty() {
            String::new()
        } else {
            format!(" matching {}", parts.join(", "))
        }
    }

    fn describe_devices(&self) -> String {
        if self.devices.is_empty() {
            return "(none)".to_string();
        }
        self.devices
            .iter()
            .enumerate()
            .map(|(i, d)| {
                format!("[{i}] {}:{} {:?} `{}`", d.info.provider_id, d.info.ordinal, d.info.kind, d.info.name)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Capability cell.
    pub fn capability(&self, index: u32, task: Task, modality: Modality) -> Result<Capability> {
        let entry = self.device(index)?;
        let provider = &self.providers[entry.provider_index];
        Ok(provider.capability(entry.info.ordinal, task, modality))
    }

    /// Feasibility check for a bundle on a device.
    pub fn can_run(&self, index: u32, bundle_dir: &Path, task: Task, modality: Modality) -> Result<()> {
        let entry = self.device(index)?;
        let provider = &self.providers[entry.provider_index];
        let cap = provider.capability(entry.info.ordinal, task, modality);
        if !cap.is_offered() {
            return Err(Error::unsupported_task(format!(
                "device [{index}] {}:{} does not offer {task:?} for {modality:?}",
                entry.info.provider_id, entry.info.ordinal
            )));
        }
        let bundle = Bundle::open(bundle_dir)?;
        provider.can_run(entry.info.ordinal, &bundle, task, modality)
    }

    /// Hold a dynamically loaded library for the runtime's lifetime.
    #[allow(dead_code)]
    pub(crate) fn retain_library(&self, lib: libloading::Library) {
        self.libraries.lock().unwrap_or_else(|p| p.into_inner()).push(lib);
    }

    /// ABI version this runtime implements.
    pub fn abi_version(&self) -> u32 {
        abi::TURBO_ABI_VERSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockProvider;

    fn runtime() -> Arc<Runtime> {
        Runtime::new(RuntimeDesc::default(), vec![Arc::new(MockProvider::new())]).unwrap()
    }

    #[test]
    fn auto_never_selects_cpu() {
        let rt = runtime();
        assert_eq!(rt.devices().len(), 2);
        let auto = rt.select(&DeviceSelector::default()).unwrap();
        assert_eq!(rt.device(auto).unwrap().info.kind, DeviceKind::Accel);
        let cpu_only = DeviceSelector { kinds: vec![DeviceKind::Cpu], ..Default::default() };
        let err = rt.select(&cpu_only).unwrap_err();
        assert_eq!(err.code(), abi::TURBO_E_DEVICE_NOT_FOUND);
        assert!(err.message().contains("never selected automatically"));
    }

    #[test]
    fn explicit_needs_provider_and_matches_ordinal() {
        let rt = runtime();
        let err = rt.select(&DeviceSelector { policy: SelectPolicy::Explicit, ..Default::default() }).unwrap_err();
        assert_eq!(err.code(), abi::TURBO_E_INVALID_ARGUMENT);
        let idx = rt
            .select(&DeviceSelector {
                policy: SelectPolicy::Explicit,
                provider_id: "mock".into(),
                ordinal: 0,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rt.device(idx).unwrap().info.kind, DeviceKind::Cpu);
        let missing = rt
            .select(&DeviceSelector {
                policy: SelectPolicy::Explicit,
                provider_id: "mock".into(),
                ordinal: 7,
                ..Default::default()
            })
            .unwrap_err();
        assert_eq!(missing.code(), abi::TURBO_E_DEVICE_NOT_FOUND);
    }

    #[test]
    fn provider_paths_are_not_silently_ignored() {
        let err = Runtime::new(RuntimeDesc { provider_paths: vec!["/nope.so".into()], ..Default::default() }, vec![])
            .unwrap_err();
        assert_eq!(err.code(), abi::TURBO_E_NOT_IMPLEMENTED);
    }
}
