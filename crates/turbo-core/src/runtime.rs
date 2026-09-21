//! The runtime: provider registry, device table, selection, capability
//! lookup, and provider library loading.
//!
//! Providers can be registered statically (built in) or loaded from shared
//! libraries at creation time or later through [`Runtime::load_provider`].
//! The device table grows append-only, so indices handed to callers stay
//! valid for the runtime's lifetime.

use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use turbo_abi as abi;

use crate::bundle::Bundle;
use crate::error::{Error, Result};
use crate::plugin;
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
    /// Do not register the built-in providers.
    pub no_default_providers: bool,
    /// Provider libraries to load, in order. A failure fails creation.
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

/// Provider load or probe failure recorded by the runtime.
#[derive(Clone, Debug)]
pub struct ProviderFailure {
    /// Provider id or library path.
    pub what: String,
    /// The error.
    pub error: Error,
}

#[derive(Default)]
struct Tables {
    providers: Vec<Arc<dyn Provider>>,
    devices: Vec<DeviceEntry>,
}

/// Library instance.
pub struct Runtime {
    tables: RwLock<Tables>,
    failures: Mutex<Vec<ProviderFailure>>,
    log: Option<LogSink>,
    // Loaded provider libraries, kept alive for the runtime's lifetime.
    libraries: Mutex<Vec<libloading::Library>>,
}

impl fmt::Debug for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.tables.read().unwrap_or_else(|p| p.into_inner());
        f.debug_struct("Runtime")
            .field("providers", &t.providers.iter().map(|p| p.id().to_string()).collect::<Vec<_>>())
            .field("devices", &t.devices.len())
            .finish()
    }
}

impl Runtime {
    /// Create a runtime with statically registered providers, then load any
    /// explicit provider libraries. A library that fails to load fails
    /// creation; a provider whose device probe fails is registered with no
    /// devices and the failure is recorded (see [`Runtime::failures`]).
    pub fn new(desc: RuntimeDesc, static_providers: Vec<Arc<dyn Provider>>) -> Result<Arc<Self>> {
        let runtime = Arc::new(Self {
            tables: RwLock::new(Tables::default()),
            failures: Mutex::new(Vec::new()),
            log: desc.log.clone(),
            libraries: Mutex::new(Vec::new()),
        });
        for provider in static_providers {
            runtime.register(provider)?;
        }
        for path in &desc.provider_paths {
            runtime.load_provider(Path::new(path))?;
        }
        Ok(runtime)
    }

    /// Register a provider and probe its devices. Fails if a provider with
    /// the same id is already registered.
    pub fn register(&self, provider: Arc<dyn Provider>) -> Result<()> {
        let id = provider.id().to_string();
        if id.is_empty() {
            return Err(Error::provider_load("provider id is empty"));
        }
        let mut t = self.tables.write().unwrap_or_else(|p| p.into_inner());
        if t.providers.iter().any(|p| p.id() == id) {
            return Err(Error::provider_load(format!("a provider with id `{id}` is already registered")));
        }
        let index = t.providers.len();
        match provider.devices() {
            Ok(list) => {
                for info in list {
                    if info.provider_id != id {
                        self.record_failure(
                            &id,
                            Error::internal(format!(
                                "provider `{id}` reported a device with provider_id `{}`",
                                info.provider_id
                            )),
                        );
                        continue;
                    }
                    t.devices.push(DeviceEntry { provider_index: index, info });
                }
            }
            Err(e) => self.record_failure(&id, e),
        }
        t.providers.push(provider);
        self.log(LogLevel::Info, &format!("registered provider `{id}`"));
        Ok(())
    }

    /// Load a provider library and register it. The library stays loaded
    /// for the runtime's lifetime.
    pub fn load_provider(&self, path: &Path) -> Result<()> {
        let loaded = plugin::load(path)?;
        let id = loaded.provider.id().to_string();
        // Keep the library alive before any vtable call can happen.
        self.libraries.lock().unwrap_or_else(|p| p.into_inner()).push(loaded.library);
        self.register(loaded.provider)?;
        self.log(LogLevel::Info, &format!("loaded provider `{id}` from `{}`", path.display()));
        Ok(())
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

    /// Provider failures recorded so far.
    pub fn failures(&self) -> Vec<ProviderFailure> {
        self.failures.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Registered providers (snapshot).
    pub fn providers(&self) -> Vec<Arc<dyn Provider>> {
        self.tables.read().unwrap_or_else(|p| p.into_inner()).providers.clone()
    }

    /// Device table (snapshot).
    pub fn devices(&self) -> Vec<DeviceEntry> {
        self.tables.read().unwrap_or_else(|p| p.into_inner()).devices.clone()
    }

    /// Number of devices.
    pub fn device_count(&self) -> u32 {
        self.tables.read().unwrap_or_else(|p| p.into_inner()).devices.len() as u32
    }

    /// Device by index.
    pub fn device(&self, index: u32) -> Result<DeviceEntry> {
        let t = self.tables.read().unwrap_or_else(|p| p.into_inner());
        t.devices.get(index as usize).cloned().ok_or_else(|| {
            Error::device_not_found(format!(
                "device index {index} is out of range; {} device(s) enumerated",
                t.devices.len()
            ))
        })
    }

    /// Provider owning a device.
    pub fn provider_for(&self, index: u32) -> Result<Arc<dyn Provider>> {
        let t = self.tables.read().unwrap_or_else(|p| p.into_inner());
        let entry = t.devices.get(index as usize).ok_or_else(|| {
            Error::device_not_found(format!(
                "device index {index} is out of range; {} device(s) enumerated",
                t.devices.len()
            ))
        })?;
        Ok(t.providers[entry.provider_index].clone())
    }

    /// Select a device. AUTO returns the first non-CPU device matching the
    /// filters (never a CPU); EXPLICIT requires provider id + ordinal (and
    /// kind, if given) to match exactly.
    pub fn select(&self, sel: &DeviceSelector) -> Result<u32> {
        let t = self.tables.read().unwrap_or_else(|p| p.into_inner());
        let matches_filters = |info: &DeviceInfo| {
            (sel.provider_id.is_empty() || info.provider_id == sel.provider_id)
                && (sel.vendor.is_empty() || info.vendor.contains(&sel.vendor))
                && (sel.kinds.is_empty() || sel.kinds.contains(&info.kind))
        };
        match sel.policy {
            SelectPolicy::Auto => {
                let found = t
                    .devices
                    .iter()
                    .enumerate()
                    .find(|(_, d)| d.info.kind != DeviceKind::Cpu && matches_filters(&d.info));
                match found {
                    Some((i, _)) => Ok(i as u32),
                    None => Err(Error::device_not_found(format!(
                        "AUTO found no accelerator{}; CPU is never selected automatically. Devices: {}",
                        describe_filters(sel),
                        describe_devices(&t.devices)
                    ))),
                }
            }
            SelectPolicy::Explicit => {
                if sel.provider_id.is_empty() {
                    return Err(Error::invalid_argument("EXPLICIT selection requires provider_id").with_field(5));
                }
                let found = t
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
                        describe_filters(sel),
                        describe_devices(&t.devices)
                    ))),
                }
            }
        }
    }

    /// Capability cell.
    pub fn capability(&self, index: u32, task: Task, modality: Modality) -> Result<Capability> {
        let entry = self.device(index)?;
        let provider = self.provider_for(index)?;
        Ok(provider.capability(entry.info.ordinal, task, modality))
    }

    /// Feasibility check for a bundle on a device.
    pub fn can_run(&self, index: u32, bundle_dir: &Path, task: Task, modality: Modality) -> Result<()> {
        let entry = self.device(index)?;
        let provider = self.provider_for(index)?;
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

    /// ABI version this runtime implements.
    pub fn abi_version(&self) -> u32 {
        abi::TURBO_ABI_VERSION
    }
}

fn describe_filters(sel: &DeviceSelector) -> String {
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

fn describe_devices(devices: &[DeviceEntry]) -> String {
    if devices.is_empty() {
        return "(none)".to_string();
    }
    devices
        .iter()
        .enumerate()
        .map(|(i, d)| format!("[{i}] {}:{} {:?} `{}`", d.info.provider_id, d.info.ordinal, d.info.kind, d.info.name))
        .collect::<Vec<_>>()
        .join(", ")
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
        assert_eq!(rt.device_count(), 2);
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
    fn duplicate_provider_id_is_rejected() {
        let rt = runtime();
        let err = rt.register(Arc::new(MockProvider::new())).unwrap_err();
        assert_eq!(err.code(), abi::TURBO_E_PROVIDER_LOAD);
        assert!(err.message().contains("already registered"));
    }

    #[test]
    fn missing_provider_library_fails_creation() {
        let err = Runtime::new(
            RuntimeDesc { provider_paths: vec!["/nonexistent/libturbo_provider_x.so".into()], ..Default::default() },
            vec![],
        )
        .unwrap_err();
        assert_eq!(err.code(), abi::TURBO_E_PROVIDER_LOAD);
    }
}
