# Intel Core Ultra NPU qualification — checklist

Goal: the first live pass for `Device::OpenVinoNpu`, on an Intel Cloud
AI PC instance (**Machine D**). Commands and details:
[intel-cloud-npu-runbook.md](intel-cloud-npu-runbook.md). Gaudi and the AMD
accelerators are separate, later tracks
([future-accelerators.md](future-accelerators.md)).

**Nothing below may be marked done without the named live evidence.**
As of the last update, every step is **open**: the only committed NPU
artifact is the Machine B fail receipt
([`testdata/receipts/turboembed/intel-npu.json`](../testdata/receipts/turboembed/intel-npu.json),
`pass=false`).

1. **Access** — *open.* Intel Cloud account approved and a Core Ultra
   instance (e.g. `BM-HPARL` / `BM-LNL` / `BM-PTL`) reachable over SSH.
   Evidence: instance reachable; record the platform name (anonymized as
   Machine D / "Intel Cloud AI PC") — no real hostnames in docs or
   receipts.
2. **Probe** — *open.* NPU driver + OpenVINO GenAI installed;
   `/dev/accel/accel0` present; `libopenvino_intel_npu_plugin.so` on the
   runtime lib path; the runbook §4 probe prints an `NPU` device from
   `ov::Core`. Evidence: probe output captured in the eventual receipt's
   `available_devices`.
3. **Live embed** — *open.* On Machine D:
   `make fetch-ov-genai ALIASES=minilm`, then
   `cargo test -p turboembed --features genai --test intel_npu -- --ignored --nocapture --test-threads=1`
   passes: engine on `Device::OpenVinoNpu`, MiniLM mean+L2, cosine ≥ 0.99
   vs the committed Intel and NVIDIA goldens (existing floors — do not
   raise or lower), `libopenvino_intel_npu_plugin` mapped, no
   CPU/GPU/mock substitution. Evidence: the test run itself.
4. **Receipt** — *open.* The passing run overwrites
   `testdata/receipts/turboembed/intel-npu.json` (`pass=true`,
   `wired=true`, chip, git SHA, `available_devices` including `NPU`);
   commit it from that checkout and update the NPU status lines in the
   root `README.md` and `testdata/receipts/turboembed/README.md`.
   Until then the honest fail receipt stays committed.
