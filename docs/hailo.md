# The Hailo backend

The `hailo` backend reaches Hailo accelerators through HailoRT's C API.
It is C++ in `core/hailo/`, compiled by `core/build.rs` into a static
library that libturbo links against HailoRT's shared library, and the
core reaches it only through its `turbo_backend` table
(`include/turbo/turbo_backend.h`). It is off by default: the `hailo`
feature of the `turbo` crate links it, and `turbo_version()` then names
`hailo`.

Today it lists devices and runs no task. Its table stops at
`capability`, so contexts, buffers, models and sessions on a Hailo
device are refused with `TURBO_E_UNSUPPORTED`, naming the function.

## Requirements

- Linux, with a Hailo PCIe driver loaded: `hailo1x_pci` for a Hailo-10H,
  `hailo_pci` for a Hailo-8 or Hailo-8L.
- HailoRT's headers (`include/hailo/hailort.h`) and `libhailort.so`, of
  the same version as the driver. On Raspberry Pi OS these come with
  `hailo-h10-all` (HailoRT 5.1.1) or `hailo-all` (HailoRT 4.23.0).
- A host C++17 compiler.

## Building

```
cargo build -p turbo --release --features hailo
```

| Variable | Meaning |
|---|---|
| `TURBO_HAILORT_ROOT` | The prefix HailoRT is installed under, with `include/hailo/hailort.h` and `libhailort.so` in `lib/`, `lib/<arch>-linux-gnu/` or `lib64/`. Unset: `/usr`. |
| `CXX` | The C++ compiler. Unset: `c++`. |

The library links `libhailort.so` with its directory as a run path. A
machine that runs it needs HailoRT; a build without the feature does
not. Without a driver HailoRT scans nothing, and the backend lists no
device.

## What it does

- **Devices.** One per device `hailo_scan_devices` returns, in its
  order, each opened once per process and identified with
  `hailo_identify`. `arch`, the label benchmark records are filed under,
  is the architecture identify reports: `hailo10h`, `hailo8`, `hailo8l`.
  `name` is the product name identify gives, else the board name, else
  the architecture's (`Hailo-10H`, which names no board). `kind` is
  `TURBO_DEVICE_NPU`. `memory_total` and `memory_free` are 0: HailoRT
  reports neither. `runtime_version` is HailoRT's library version, and
  `driver_version` the kernel module's and the firmware's
  (`hailo1x_pci 5.1.1, firmware 5.1.1`). A device HailoRT scans but that
  does not answer identify is left out, and the runtime's log says why.
- **Capability.** Every cell is UNSUPPORTED, with the reason that the
  backend runs no task yet, so `turbo_runtime_select` never picks a Hailo
  device.

## Testing on a Hailo machine

From the workspace root. `TURBO_TEST_REQUIRE_HAILO=1` makes every test in
`core/tests/hailo.rs` that needs a device fail when the backend lists
none, where without it the test passes with a line saying it was
skipped; set it on a Hailo machine, so a run that found no device cannot
pass.

```
export TURBO_TEST_REQUIRE_HAILO=1

# Everything, with the Hailo-only tests in core/tests/hailo.rs:
cargo test -p turbo --features hailo

# The Hailo-only tests, with the devices they list:
cargo test -p turbo --features hailo --test hailo -- --nocapture
```

The listing test compares the devices listed with what
`hailo_scan_devices` returns, and each one's `arch` with its PCI device
id in sysfs (`1e60:45c4` is a Hailo-10H; `1e60:2864` a Hailo-8 or
Hailo-8L). HailoRT writes `hailort.log` into the directory a test runs
in.
