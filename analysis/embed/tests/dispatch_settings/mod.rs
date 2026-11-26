pub(super) fn inputs<'a>(
    bundle: Option<&'a str>,
    device: Option<&'a str>,
    required: Option<&str>,
) -> Result<Option<(&'a str, &'a str)>, String> {
    let device = device.filter(|value| !value.trim().is_empty());
    let bundle = bundle.filter(|value| !value.trim().is_empty());
    turbo_pins::check_test_device(device, required)?;
    if let Some(name) = device
        && let Some((backend, _)) = name.split_once(':')
    {
        return Err(format!(
            "TURBO_TEST_DEVICE={name}: the dispatch test picks a backend's first device ({backend}) or a device index"
        ));
    }
    if let Some(backend) = required
        && bundle.is_none()
    {
        return Err(format!("strict {backend} testing requires TURBO_TEST_VECTORS_BUNDLE to name a bundle"));
    }
    Ok(bundle.zip(device))
}

pub(super) fn check_execution(execution: &str, required: &str) -> Result<(), String> {
    if execution.split_whitespace().next() == Some(required) {
        Ok(())
    } else {
        Err(format!("strict device test requires {required} on the fixed device lane, got {execution}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{check_execution, inputs};

    #[test]
    fn strict_hailo_rejects_missing_device_or_bundle() {
        let required = Some("hailo");
        for missing in [None, Some(""), Some(" ")] {
            assert!(inputs(Some("bundle"), missing, required).unwrap_err().contains("TURBO_TEST_DEVICE=hailo"));
            let no_bundle = inputs(missing, Some("hailo"), required).unwrap_err();
            assert_eq!(no_bundle, "strict hailo testing requires TURBO_TEST_VECTORS_BUNDLE to name a bundle");
        }
    }

    #[test]
    fn strict_hailo_rejects_other_backends() {
        for device in ["cpu", "cuda", "levelzero", "0"] {
            assert!(
                inputs(Some("bundle"), Some(device), Some("hailo")).unwrap_err().contains("requires a hailo device")
            );
        }
        assert_eq!(inputs(Some("bundle"), Some("hailo"), Some("hailo")).unwrap(), Some(("bundle", "hailo")));
    }

    #[test]
    fn a_backend_with_a_number_is_refused_before_any_timing() {
        for (device, required) in [("levelzero:0", None), ("hailo:0", Some("hailo")), ("hailo:1", None)] {
            let backend = device.split(':').next().unwrap();
            assert_eq!(
                inputs(Some("bundle"), Some(device), required).unwrap_err(),
                format!(
                    "TURBO_TEST_DEVICE={device}: the dispatch test picks a backend's first device ({backend}) or a device index"
                )
            );
        }
        assert!(
            inputs(Some("bundle"), Some("levelzero:0"), Some("hailo")).unwrap_err().contains("requires a hailo device")
        );
    }

    #[test]
    fn optional_run_skips_if_inputs_are_missing() {
        assert_eq!(inputs(Some("bundle"), None, None).unwrap(), None);
        assert_eq!(inputs(None, Some("hailo"), None).unwrap(), None);
        assert_eq!(inputs(Some("bundle"), Some("cpu"), None).unwrap(), Some(("bundle", "cpu")));
    }

    #[test]
    fn strict_hailo_requires_fixed_lane_provenance() {
        assert!(check_execution("hailo i8 (model)", "hailo").is_ok());
        for actual in ["cpu f32 (model)", "cuda f16 (model)", "hailo-other i8 (model)", ""] {
            assert!(check_execution(actual, "hailo").unwrap_err().contains("fixed device lane"));
        }
    }
}
