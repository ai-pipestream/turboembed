//! Group `device`, Rust layer: selection policy (PLAN.md principle 4).
//! AUTO means the host's best accelerator and never a CPU; CPU runs only when
//! it is named; an absent device is an error, never a fallback.

use turbo::abi::*;
use turbo::provider::{EmbedOptions, RunOptions};
use turbo::runtime::DeviceSelector;
use turbo::types::{DeviceKind, SelectPolicy};
use turbo_conformance::{assert_err, needs, read_f32, BundleKind, Target};

#[test]
fn device_auto_never_selects_a_cpu() {
    let t = Target::from_env();
    let (runtime, _) = t.detached();
    let index = runtime.select(&DeviceSelector::default()).expect("AUTO must find the accelerator");
    let entry = runtime.device(index).expect("device");
    assert_ne!(entry.info.kind, DeviceKind::Cpu, "AUTO selected a CPU device: {}", entry.info.name);
    // Repeating the selection is stable.
    let again = runtime.select(&DeviceSelector::default()).expect("AUTO again");
    assert_eq!(index, again, "AUTO must be deterministic");
}

#[test]
fn device_auto_with_a_cpu_only_mask_is_device_not_found() {
    let t = Target::from_env();
    let (runtime, _) = t.detached();
    let cpu_only = DeviceSelector { kinds: vec![DeviceKind::Cpu], ..Default::default() };
    let e = assert_err!(runtime.select(&cpu_only), TURBO_E_DEVICE_NOT_FOUND);
    assert!(
        e.message().contains("never selected automatically"),
        "the message must say why CPU is excluded: {}",
        e.message()
    );
}

#[test]
fn device_auto_honors_kind_and_provider_filters() {
    let t = Target::from_env();
    let (runtime, index) = t.detached();
    let kind = runtime.device(index).expect("device").info.kind;
    let filtered = DeviceSelector { kinds: vec![kind], ..Default::default() };
    if kind == DeviceKind::Cpu {
        // AUTO never selects a CPU, even when the filter asks for that kind
        // alone (a CPU target such as openvino ordinal 1 or ggml's CPU).
        let e = assert_err!(runtime.select(&filtered), TURBO_E_DEVICE_NOT_FOUND);
        assert!(e.message().contains("CPU"), "the refusal must say why: {}", e.message());
    } else {
        let picked = runtime.select(&filtered).expect("AUTO with the device's own kind");
        assert_eq!(runtime.device(picked).expect("device").info.kind, kind);
    }

    let wrong_provider = DeviceSelector { provider_id: "no-such-provider".into(), ..Default::default() };
    assert_err!(runtime.select(&wrong_provider), TURBO_E_DEVICE_NOT_FOUND);
    let wrong_vendor = DeviceSelector { vendor: "no-such-vendor".into(), ..Default::default() };
    assert_err!(runtime.select(&wrong_vendor), TURBO_E_DEVICE_NOT_FOUND);
}

#[test]
fn device_explicit_without_a_provider_is_invalid_argument() {
    let t = Target::from_env();
    let (runtime, _) = t.detached();
    let no_provider = DeviceSelector { policy: SelectPolicy::Explicit, ..Default::default() };
    assert_err!(runtime.select(&no_provider), TURBO_E_INVALID_ARGUMENT, field = 5);
}

#[test]
fn device_explicit_with_a_wrong_ordinal_is_device_not_found() {
    let t = Target::from_env();
    let (runtime, _) = t.detached();
    let missing = DeviceSelector {
        policy: SelectPolicy::Explicit,
        provider_id: t.provider_id().to_string(),
        ordinal: 9999,
        ..Default::default()
    };
    let e = assert_err!(runtime.select(&missing), TURBO_E_DEVICE_NOT_FOUND);
    assert!(e.message().contains("9999"), "the message must name the ordinal: {}", e.message());
}

#[test]
fn device_explicit_selects_the_named_device() {
    let t = Target::from_env();
    let (runtime, _) = t.detached();
    let exact = DeviceSelector {
        policy: SelectPolicy::Explicit,
        provider_id: t.provider_id().to_string(),
        ordinal: t.ordinal(),
        ..Default::default()
    };
    let index = runtime.select(&exact).expect("explicit selection");
    let info = runtime.device(index).expect("device").info.clone();
    assert_eq!(info.provider_id, t.provider_id());
    assert_eq!(info.ordinal, t.ordinal());
}

#[test]
fn device_explicit_cpu_loads_and_runs() {
    let t = Target::from_env();
    needs!(t, Embedding);
    let (runtime, _) = t.detached();
    // The provider under test's own CPU device; the built-in mock's CPU is
    // always present and would load nothing but mock bundles.
    let cpus: Vec<_> = runtime
        .devices()
        .into_iter()
        .filter(|d| d.info.kind == DeviceKind::Cpu && d.info.provider_id == t.provider_id())
        .collect();
    let Some(cpu) = cpus.first() else {
        // The AUTO-never-picks-a-CPU half of the rule is asserted in
        // `device_auto_never_selects_a_cpu` and
        // `device_auto_with_a_cpu_only_mask_is_device_not_found`, which run
        // on every provider.
        println!("not applicable: provider `{}` enumerates no CPU device", t.provider_id());
        return;
    };
    let selector = DeviceSelector {
        policy: SelectPolicy::Explicit,
        provider_id: cpu.info.provider_id.clone(),
        ordinal: cpu.info.ordinal,
        ..Default::default()
    };
    let index = runtime.select(&selector).expect("explicit CPU selection");
    assert_eq!(runtime.device(index).expect("device").info.kind, DeviceKind::Cpu);

    let ctx = turbo::handles::Context::create(runtime.clone(), index, &turbo::provider::ContextDesc::default())
        .expect("context on the CPU device");
    let model = ctx
        .load_model(&t.bundle(BundleKind::Embedding), &turbo::provider::ModelDesc::default())
        .expect("load on the CPU device");
    let session = model.create_session(&turbo::provider::SessionDesc::default()).expect("session");
    session.write_text(&["hello world"], &EmbedOptions::default()).expect("write");
    let result = session.run(&RunOptions::default()).expect("run on the CPU device");
    let v = read_f32(&result, 0);
    assert_eq!(v.len(), model.info().dim as usize);
    assert!(v.iter().all(|x| x.is_finite()), "the CPU device produced a non-finite vector: {v:?}");
    assert!(v.iter().any(|x| *x != 0.0), "the CPU device produced an all-zero vector");
}

#[test]
fn device_out_of_range_index_is_device_not_found() {
    let t = Target::from_env();
    let count = t.runtime.device_count();
    assert_err!(t.runtime.device(count), TURBO_E_DEVICE_NOT_FOUND);
    assert_err!(t.runtime.device(u32::MAX), TURBO_E_DEVICE_NOT_FOUND);
    assert_err!(
        turbo::handles::Context::create(t.runtime.clone(), count, &turbo::provider::ContextDesc::default()),
        TURBO_E_DEVICE_NOT_FOUND
    );
}

#[test]
fn device_info_is_self_consistent() {
    let t = Target::from_env();
    let count = t.runtime.device_count();
    assert!(count > 0, "a runtime with no devices cannot be conformance-tested");
    for index in 0..count {
        let entry = t.runtime.device(index).expect("device");
        let d = &entry.info;
        assert!(!d.provider_id.is_empty(), "every device names its provider");
        assert!(!d.name.is_empty(), "every device has a name");
        assert!(DeviceKind::ALL.contains(&d.kind), "device kind {:?} is not a known constant", d.kind);
        assert!(d.memory_free <= d.memory_total || d.memory_total == 0, "{}: free > total", d.name);
        // The device's provider is the one it names, not just some provider.
        let provider = t.runtime.provider_for(index).expect("provider for the device");
        assert_eq!(provider.id(), d.provider_id, "device {index} ({}) names another provider", d.name);
        // The ordinal is the device's index within its own provider, so
        // re-selecting it explicitly must land on this same device.
        let selector = turbo::runtime::DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: d.provider_id.clone(),
            ordinal: d.ordinal,
            ..Default::default()
        };
        let again = t.runtime.select(&selector).expect("re-selecting an enumerated device");
        assert_eq!(again, index, "{}/{} does not select back to device {index}", d.provider_id, d.ordinal);
    }
}
