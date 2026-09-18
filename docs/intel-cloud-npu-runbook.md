# Intel Cloud NPU runbook — proving `Device::OpenVinoNpu` live

`Device::OpenVinoNpu` has never produced a passing receipt. The committed
[`testdata/receipts/turboembed/intel-npu.json`](../testdata/receipts/turboembed/intel-npu.json)
is an honest **fail** from Machine B (AMD Ryzen CPU + Intel Battlemage dGPU:
no NPU silicon, no `intel-npu` accel node, no
`libopenvino_intel_npu_plugin.so`). This runbook is the path to a live pass
on an **Intel Core Ultra** host provisioned from Intel's AI PC cloud —
referred to in this repo as **Machine D / "Intel Cloud AI PC"**. Do not put
real hostnames in docs or receipts.

Scope notes:

- **Gaudi is out of scope for this path.** Gaudi (Habana) is Intel
  datacenter silicon on the Synapse stack, not the OpenVINO NPU plugin.
  See [future-accelerators.md](future-accelerators.md).
- Device policy is unchanged: an NPU request either runs on NPU or fails
  loud. Never CPU, never GPU, never the 8-d FNV mock.
- Cosine floors are unchanged: 0.99 vs the committed Intel and NVIDIA
  MiniLM goldens, the same gates as the Machine B GPU/CPU receipts.
- No Python on the hot path.

Progress tracking: [intel-npu-checklist.md](intel-npu-checklist.md).

## 1. Get an Intel Cloud AI PC instance

Intel's AI PC cloud is part of Intel Cloud Services. Follow the
[AI PC Cloud Quick User Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/ai-pc-cloud-quick-user-guide.html):

1. Sign up at [https://cloud.intel.com](https://cloud.intel.com) with a
   **company or university email** (personal addresses are rejected),
   verify the email, and accept the Terms of Service.
2. In **Hardware Catalog**, request a Core Ultra client platform. Platforms
   observed in the catalog include `BM-HPARL` (Arrow Lake, Core Ultra 7
   265T), `BM-LNL` (Lunar Lake), `BM-PTL` (Panther Lake), `BM-NVL`
   (Nova Lake), and `BM-WCL` (Wildcat Lake). All of these carry an Intel
   NPU; any of them can serve as Machine D.
3. Availability is gated. The request form requires an **Intended Use**
   statement, and the Use case field may need to be set to **"AI PC USA"**.
   Processing can take up to ~48 hours — request the instance before
   scheduling the qualification work.

## 2. Install and verify the NPU driver

On the instance (Ubuntu assumed; the images usually ship the driver):

```bash
ls /dev/accel/            # expect accel0
dmesg | grep -i intel_vpu # or: modinfo intel_vpu
```

If `/dev/accel/accel0` is missing, install the
[Intel NPU driver (linux-npu-driver)](https://github.com/intel/linux-npu-driver/releases)
matching the kernel, then re-check. Make sure the login user can open the
accel node (typically membership in the `render` group).

## 3. Install OpenVINO GenAI and verify the NPU plugin

Use the same archive layout Machine B uses (build discovery honors
`OPENVINO_GENAI_DIR`; `/work/opt/openvino_genai` is the conventional
prefix — see [intel-genai-embed.md](intel-genai-embed.md)). Machine B runs
the 2026.3.1 archive; matching it keeps the two Intel hosts comparable:

```bash
curl -LO https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.3.1/linux/openvino_genai_ubuntu24_2026.3.1.0_x86_64.tar.gz
mkdir -p /work/opt && tar -xzf openvino_genai_ubuntu24_2026.3.1.0_x86_64.tar.gz -C /work/opt
mv /work/opt/openvino_genai_ubuntu24_2026.3.1.0_x86_64 /work/opt/openvino_genai
source /work/opt/openvino_genai/setupvars.sh
ls /work/opt/openvino_genai/runtime/lib/intel64/libopenvino_intel_npu_plugin.so
```

The last line must exist. Without that plugin `ov::Core` will never list
`NPU` and engine create fails loud (that is the Machine B state).

## 4. Prove `ov::Core` lists NPU

Quick standalone probe (this is the same check `require_ov_device` runs
inside `turboembed_engine_create`):

```bash
cat > /tmp/probe.cpp <<'EOF'
#include <openvino/openvino.hpp>
#include <iostream>
int main() {
    for (const auto& d : ov::Core().get_available_devices())
        std::cout << d << "\n";
}
EOF
g++ -std=c++17 /tmp/probe.cpp -I"$OPENVINO_GENAI_DIR/runtime/include" \
  -L"$OPENVINO_GENAI_DIR/runtime/lib/intel64" -lopenvino -o /tmp/probe
/tmp/probe          # must print a line starting with NPU
```

(`setupvars.sh` sets `OPENVINO_GENAI_DIR` and the library path.) If `NPU`
is not listed, stop here and fix the driver/plugin — do not proceed to the
embed test expecting a fallback; there is none.

## 5. Fetch the MiniLM OpenVINO IR

Same SHA-pinned IR as the Machine B GPU/CPU receipts
(see [fetching-models.md](fetching-models.md)):

```bash
make fetch-ov-genai ALIASES=minilm     # → models/ov/minilm
```

## 6. Run the live NPU test

The pass harness is
[`crates/turboembed/tests/intel_npu.rs`](../crates/turboembed/tests/intel_npu.rs)
(`#[ignore]`d — it must be requested explicitly, and it FAILS on a host
without an NPU rather than skipping):

```bash
source /work/opt/openvino_genai/setupvars.sh
cargo test -p turboembed --features genai --test intel_npu \
  -- --ignored --nocapture --test-threads=1
```

The test:

- creates the engine with `Device::OpenVinoNpu` (fail-loud; a missing NPU
  is a test failure, never a CPU/GPU/mock substitution),
- asserts the live `ov::Core` device list contains `NPU`,
- loads `minilm` and embeds with the catalog contract (mean pooling + L2),
- gates cosine ≥ 0.99 against both
  `testdata/e2e/goldens/intel/minilm.json` and
  `testdata/e2e/goldens/nvidia/minilm.json` (the existing floors),
- requires `libopenvino_intel_npu_plugin` mapped in the process and
  forbids `libopenvino_genai` / `libopenvino_tokenizers` / `libpython`
  (WordPiece write-through stays the tokenizer path),
- on pass, overwrites `testdata/receipts/turboembed/intel-npu.json` with
  `pass=true`, `wired=true`, the chip name, git SHA, and the live
  `available_devices` list.

The always-on policy tests still run with the rest of the Intel suite
(`make test-turboembed-intel`): `npu_request_never_silently_uses_cpu_or_mock`
in `intel_genai_gpu.rs` and the synthetic-list cases in `device_policy.rs`.

## 7. Refresh the receipt

Only a live pass may replace the committed fail receipt. After the test
passes:

1. Inspect the rewritten `testdata/receipts/turboembed/intel-npu.json`:
   `pass=true`, `wired=true`, `available_devices` includes `NPU`, cosine
   numbers at or above 0.99, git SHA matching the checkout.
2. Commit the receipt from that checkout. Keep the host anonymized —
   the harness writes `"Intel Cloud AI PC (Machine D)"`, not a hostname.
3. Update the checklist ([intel-npu-checklist.md](intel-npu-checklist.md))
   and the NPU status lines in the root `README.md` and
   `testdata/receipts/turboembed/README.md`.

Until then, the fail receipt stays committed and no doc may claim a
passing NPU run.
