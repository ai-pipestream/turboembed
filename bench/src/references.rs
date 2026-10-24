use turbo::record::ReferenceRun;
use turbo_bench::measure::{Measurement, Plan};
use turbo_bench::openvino::{self, OpenVino};
use turbo_bench::tei::{self, Tei};
use turbo_bench::tensorrt::{self, TensorRt};
use turbo_bench::{Result, hailo};

/// A reference program disabled on the command line.
fn disabled(name: &str, role: &str, flag: &str) -> ReferenceRun {
    ReferenceRun {
        name: name.into(),
        role: role.into(),
        pinned: String::new(),
        version: String::new(),
        commands: Vec::new(),
        procedure: String::new(),
        measured: None,
        not_run: Some(format!("disabled on the command line ({flag})")),
    }
}

/// Which reference programs the command line named, and which it
/// disabled.
pub(crate) struct Given {
    pub(crate) tei: (bool, bool),
    pub(crate) trt: (bool, bool),
    pub(crate) ov: (bool, bool),
    pub(crate) hailo: (bool, bool),
}

impl Given {
    fn get(&self, name: &str) -> (bool, bool) {
        match name {
            tei::NAME => self.tei,
            tensorrt::NAME => self.trt,
            openvino::NAME => self.ov,
            hailo::NAME => self.hailo,
            _ => (false, false),
        }
    }

    /// Each program for the backend is named or disabled, and no other.
    pub(crate) fn check(&self, backend: &str) -> Result<()> {
        turbo_bench::wanted(backend, |name| {
            let (named, disabled) = self.get(name);
            named || disabled
        })
    }
}

/// The devices the runtime lists for `backend`.
fn devices_of(backend: &str) -> Result<u32> {
    let rt = turbo_bench::api::Runtime::create()?;
    let mut n = 0;
    for i in 0..rt.device_count()? {
        n += u32::from(turbo_bench::api::field(&rt.device_info(i)?.backend) == backend);
    }
    Ok(n)
}

/// Every reference program for the device's backend, run or recorded as
/// disabled; `wanted` has checked the command line against the backend.
pub(crate) fn run(
    m: &Measurement,
    plan: &Plan,
    tei: Option<Tei>,
    trt: Option<TensorRt>,
    ov: Option<OpenVino>,
    given: &Given,
    hailort: Option<hailo::HailoRt>,
) -> Result<Vec<ReferenceRun>> {
    let backend = m.backend();
    given.check(&backend)?;
    let gpu = (backend == "cuda").then_some(m.device.ordinal);
    let library = if m.is_cpu() {
        Some(turbo_bench::cpus::library_threads(std::env::var(turbo_bench::cpus::LIBRARY_VAR).ok().as_deref())?)
    } else {
        None
    };
    // docker numbers GPUs in the driver's (PCI bus) order, CUDA fastest
    // first unless told otherwise: with more than one, they must agree.
    let named = tei.is_some() || trt.is_some();
    if gpu.is_some() && named && std::env::var("CUDA_DEVICE_ORDER").as_deref() != Ok("PCI_BUS_ID") {
        let cuda = devices_of("cuda")?;
        if cuda > 1 {
            return Err(format!(
                "{cuda} CUDA devices are listed: set CUDA_DEVICE_ORDER=PCI_BUS_ID so the device measured is \
                 the one docker gives the reference program"
            ));
        }
    }
    // benchmark_app's GPU is OpenVINO's first, which with several Intel
    // GPUs need not be the one measured. The same for NPU.
    if backend == "levelzero" && ov.is_some() {
        let n = devices_of("levelzero")?;
        if n > 1 {
            return Err(format!(
                "{n} Level Zero devices are listed: OpenVINO's GPU need not be the device measured, so \
                 benchmark_app is run only where there is one"
            ));
        }
    }
    if backend == "npu" && ov.is_some() {
        let n = devices_of("npu")?;
        if n > 1 {
            return Err(format!(
                "{n} NPU devices are listed: OpenVINO's NPU need not be the device measured, so benchmark_app is \
                 run only where there is one"
            ));
        }
    }
    let mut out = Vec::new();
    for &name in turbo_bench::applies(&backend) {
        out.push(match name {
            tei::NAME => match &tei {
                Some(t) => tei::run(t, m, gpu, library, plan.warmup, plan.iterations)?,
                None => disabled(tei::NAME, "end_to_end", "--no-tei"),
            },
            tensorrt::NAME => match &trt {
                Some(t) => tensorrt::run(t, m, m.device.ordinal, plan.iterations)?,
                None => disabled(tensorrt::NAME, "kernel", "--no-tensorrt"),
            },
            openvino::NAME => match ov.clone() {
                Some(mut o) => {
                    o.device = if backend == "npu" { openvino::DEVICE_NPU } else { openvino::DEVICE_GPU }.into();
                    openvino::run(&o, m, plan.iterations)?
                }
                None => disabled(openvino::NAME, "kernel", "--no-openvino"),
            },
            hailo::NAME => match &hailort {
                Some(h) => hailo::run(h, m)?,
                None => disabled(hailo::NAME, "kernel", "--no-hailort"),
            },
            other => return Err(format!("{other}: no runner")),
        });
    }
    Ok(out)
}
