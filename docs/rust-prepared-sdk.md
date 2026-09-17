# Rust prepared SDK

The optional `turboembed::prepared` module wraps the
[Intel native SDK](native-sdk.md). Build and install that SDK first, then enable
`prepared` on the existing `turboembed` crate and set
`TURBOEMBED_PREPARED_SDK=/path/to/sdk` when compiling. This feature links the
installed library; it does not build or download OpenVINO.

The current target is Linux x86_64. At execution, the loader must find
`libturboembed_prepared.so.1`, either through your application's RPATH or a
loader search path such as `LD_LIBRARY_PATH=/path/to/sdk/lib`. OpenVINO and TBB
resolve from the packaged SDK. GPU drivers remain host prerequisites.

```rust,no_run
use turboembed::prepared::{Context, Device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new(Device::Gpu { ordinal: 0 })?;
    let model = context.load_model("/path/to/minilm-bundle")?;
    let mut slot = model.slot(1, 32)?;
    let mut output = vec![0.0_f32; model.info()?.dimension as usize];
    slot.write_text(&["hello world"])?;
    let result = slot.execute()?;
    result.read_into(&mut output)?;
    result.close()?;
    Ok(())
}
```

`prepared::devices()` lists the selectable devices (GPUs by ascending ordinal,
then CPU) with the same resolved identity and capability bits a created context
reports; `DeviceInfo::selector()` yields the matching explicit `Device`.
Discovery informs selection only: an unlisted device still fails to create a
context, and CPU is never an automatic fallback.

Contexts and models can be shared between threads. Each slot owns its request
and buffers; it can move between threads but requires exclusive access for
writes and execution. A result borrows its slot, preventing reuse or destruction
until the result is released. Models and slots retain their native parents, so
closing a Rust parent does not invalidate its children.

`write_tokens` accepts complete row-major i32 token, mask and optional type-ID
slices for the fixed shape. Inputs are copied before return and may be reused
across executions. A failed write invalidates earlier inputs. `write_text`
accepts UTF-8, including empty strings and embedded NUL, and uses the model's
native tokenizer. Text descriptors reuse slot-owned capacity.

Use `read_into` with a caller-reused f32 slice for explicit host copies;
`to_vec` is an allocating convenience. `opencl` returns a borrowed view that
keeps the result borrowed. Extracting raw GPU handles is unsafe: retain the
lease, treat its buffer as read-only, use the supplied in-order queue, and never
release borrowed OpenCL references. Explicit result `close` reports completion
errors; `Drop` releases resources but cannot report them.

## Validation

From the repository root, on the Intel GPU host with the SDK installed:

```bash
export TURBOEMBED_PREPARED_SDK=/path/to/sdk
export TURBOEMBED_PREPARED_BUNDLE=/path/to/minilm-bundle
export LD_LIBRARY_PATH="$TURBOEMBED_PREPARED_SDK/lib"
cargo test --locked -p turboembed --features prepared --doc
cargo test --locked -p turboembed --features prepared --lib prepared_v1_layout_matches_c_header
cargo test --locked -p turboembed --features prepared --test prepared_sdk -- --ignored
```

The hardware tests are explicitly ignored in ordinary test runs. The last
command requires both the Intel GPU and CPU reference. On a host without a GPU,
the SDK-only subset still runs and must pass:

```bash
cargo test --locked -p turboembed --features prepared --test prepared_sdk -- \
  --ignored device_discovery_and_explicit_selection prepared_cpu_reference_execution
```

See the [binding validation receipt](intel-bindings-2026-09-14.md) for the
exact tested SDK, model, device and scope, and the
[discovery validation receipt](prepared-discovery-2026-09-17.md) for the
CPU-only discovery run. `machine_b_gpu_discovery_receipt` remains
hardware-unverified until re-run on the Machine B Intel GPU host.
