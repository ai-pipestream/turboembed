//! Without a device, the devices listed, and capability and selection.

use super::*;

// ---- Without a device ------------------------------------------------------------------

#[test]
fn the_backend_is_linked_before_the_cpu_and_names_itself() {
    let linked = turbo::backend::linked();
    let b = linked.iter().find(|b| b.name() == "levelzero").expect("linked");
    assert_eq!(b.struct_size as usize, size_of::<turbo::backend::turbo_backend>());
    turbo::backend::check_table(b).unwrap();
    assert!(b.buffer_read.is_some() && b.session_run.is_some());
    let at = |name: &str| linked.iter().position(|b| b.name() == name).unwrap();
    assert!(at("levelzero") < at("cpu"), "its devices come before the cpu's");
    let v = unsafe { std::ffi::CStr::from_ptr(b.runtime_version) }.to_str().unwrap();
    assert_eq!(v, "", "nothing is linked at build time; each device names the loader it opened");
}

// ---- Devices -------------------------------------------------------------------------------

/// The PCI device ids of the Intel GPUs the kernel has bound a compute
/// driver to, one per render node.
fn intel_gpus() -> Vec<u32> {
    let mut ids = Vec::new();
    let Ok(dir) = std::fs::read_dir("/sys/class/drm") else { return ids };
    for entry in dir {
        let path = entry.unwrap().path();
        if !path.file_name().unwrap().to_str().unwrap().starts_with("renderD") {
            continue;
        }
        let dev = path.join("device");
        let read = |f: &str| std::fs::read_to_string(dev.join(f)).unwrap_or_default().trim().to_owned();
        let driver = std::fs::read_link(dev.join("driver")).unwrap_or_default();
        let driver = driver.file_name().and_then(|d| d.to_str()).unwrap_or("");
        if read("vendor") == "0x8086" && (driver == "xe" || driver == "i915") {
            ids.push(u32::from_str_radix(read("device").trim_start_matches("0x"), 16).unwrap());
        }
    }
    ids.sort();
    ids
}

/// The version in an installed library's file name: libze_loader.so.1
/// links to libze_loader.so.1.28.2, which is "1.28.2". None where the
/// library is not in one of the usual directories.
fn installed_version(soname: &str) -> Option<String> {
    let lib = ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"]
        .iter()
        .find_map(|d| std::fs::read_link(format!("{d}/{soname}")).ok())?;
    let name = lib.file_name()?.to_str()?;
    Some(name.strip_prefix(soname.trim_end_matches(".1"))?.trim_start_matches('.').to_owned())
}

/// Every Intel GPU the kernel drives is listed, and each reads as the
/// hardware and software it is.
#[test]
fn the_devices_listed_are_the_kernels() {
    let _t = turn();
    let rt = Rt::new();
    let want = intel_gpus();
    let listed = rt.listed();
    if listed.is_empty() {
        // No GPU, or one without the compute runtime: the listing says
        // nothing about the kernel's unless a device is required.
        assert!(!required(), "TURBO_TEST_REQUIRE_LEVELZERO=1 and the levelzero backend lists no device");
        println!("the_devices_listed_are_the_kernels: skipped: the levelzero backend lists no device");
        return;
    }
    assert_eq!(listed.len(), want.len(), "the kernel drives {want:x?}");
    let mut archs: Vec<String> = listed.iter().map(|&i| field(&rt.info(i).arch)).collect();
    archs.sort();
    let mut expect: Vec<String> =
        want.iter().map(|&id| if id == 0xe223 { "b70".to_owned() } else { format!("intel-{id:04x}") }).collect();
    expect.sort();
    assert_eq!(archs, expect);
    let (loader, driver) = (installed_version("libze_loader.so.1"), installed_version("libze_intel_gpu.so.1"));
    for (ordinal, &i) in listed.iter().enumerate() {
        let d = rt.info(i);
        assert_eq!(d.ordinal, ordinal as u32, "numbered within the backend");
        assert!(d.kind == TURBO_DEVICE_GPU || d.kind == TURBO_DEVICE_IGPU);
        assert_eq!(d.unified_memory, u32::from(d.kind == TURBO_DEVICE_IGPU));
        assert_eq!(field(&d.vendor), "Intel");
        assert!(!field(&d.name).is_empty(), "the driver names it");
        if let Some(loader) = &loader {
            assert_eq!(&field(&d.runtime_version), loader, "the loader's version");
        }
        if let Some(build) = driver.as_ref().and_then(|v| v.rsplit('.').next()) {
            assert!(field(&d.driver_version).ends_with(&format!(".{build}")), "the installed driver's build {build}");
        }
        assert!(d.memory_total > 0);
        assert!(d.memory_free > 0 && d.memory_free <= d.memory_total, "sysman reads free memory");
        println!(
            "levelzero device {ordinal}: {}, arch {}, loader {}, driver {}, {} of {} bytes free",
            field(&d.name),
            field(&d.arch),
            field(&d.runtime_version),
            field(&d.driver_version),
            d.memory_free,
            d.memory_total
        );
    }
    let first = listed[0];
    assert_eq!(listed, (first..first + listed.len() as u32).collect::<Vec<_>>(), "a backend's devices are contiguous");
    assert!((first + listed.len() as u32..rt.count()).all(|i| field(&rt.info(i).backend) == "cpu"), "then the cpu");
}

#[test]
fn runtimes_on_many_threads_list_the_same_devices() {
    let infos: Vec<Vec<String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..16)
            .map(|_| {
                scope.spawn(|| {
                    let rt = Rt::new();
                    rt.listed()
                        .into_iter()
                        .map(|i| {
                            let mut info = rt.info(i);
                            info.memory_free = 0;
                            format!("{info:?}")
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(infos.iter().all(|i| *i == infos[0]), "{infos:#?}");
}

// ---- Capability and selection --------------------------------------------------------------

#[test]
fn embed_is_offered_in_f32_and_f16_as_sessions_run_it() {
    let _t = turn();
    let Some(_) = gpu_device("embed_is_offered_in_f32_and_f16_as_sessions_run_it") else { return };
    let l = on_gpu(&tiny_bundle());
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        assert_eq!(unsafe { turbo_runtime_capability(l.rt, gpu(l.rt), TURBO_TASK_EMBED, p, &mut cap, null_err()) }, 0);
        // A benchmark record compiled into the build for this machine's
        // cell makes it SUPPORTED, naming the record; else it runs
        // unmeasured.
        if cap.status == backend::TURBO_CAP_SUPPORTED {
            assert!(field(&cap.benchmark).starts_with(&format!("{}.levelzero.embed.", field(&arch_of(&l)))));
            assert!(cap.cosine_floor > 0.999 && cap.speed_ratio > 0.0, "a record's numbers");
        } else {
            assert_eq!(cap.status, backend::TURBO_CAP_EXPERIMENTAL, "{}", field(&cap.reason));
        }
        // FASTEST runs the linear layers on the matrix engines in F16.
        let want = if p == TURBO_PRECISION_FASTEST { TURBO_DTYPE_F16 } else { TURBO_DTYPE_F32 };
        assert_eq!(cap.dtype, want, "precision {p}");
        assert_eq!(cap.options_honored, 0b111111, "every field of turbo_embed_options");
        let info = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap().info();
        assert_eq!(info.compute_dtype, cap.dtype, "precision {p}");
    }
}

/// The arch label of the levelzero device a bundle is loaded on.
fn arch_of(l: &Loaded) -> [c_char; 32] {
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    assert_eq!(unsafe { turbo_runtime_device_info(l.rt, gpu(l.rt), &mut info, null_err()) }, 0);
    info.arch
}

#[test]
fn selection_picks_the_gpu() {
    let _t = turn();
    let Some(dev) = gpu_device("selection_picks_the_gpu") else { return };
    let rt = Rt::new();
    let mut pick = 99u32;
    let mut reason = [0 as c_char; 256];
    let rc = unsafe { turbo_runtime_select(rt.0, TURBO_TASK_EMBED, &mut pick, reason.as_mut_ptr(), 256, null_err()) };
    assert_eq!(rc, 0, "{}", field(&reason));
    assert_eq!(pick, dev, "{}", field(&reason));
    assert!(field(&reason).contains("(levelzero "), "{}", field(&reason));
}
