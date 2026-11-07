# #1480 chunker v1 records: Intel NPU (Arrow Lake)

Recordings for the #1480 chunker v1 gate on an Intel NPU (Intel Core Ultra 7 265T, Arrow Lake).

- Commit: 89e6a6ae0150730ad38e5d43a5333819e3769a24 (see head.txt)
- Bundle: minilm-npu-token-ids-f16-s256-605aeb78e, fixed_seq 256
- Device: Intel(R) AI Boost (DEVICE_NPU), driver_version 0.15.21738, runtime_version 1.28.2 (from provenance.device in the records)
- All five hybrid pipelines were recorded, 4 JSON records each.
- The turbo-eval gate against the NPU baselines exited 1 (FAIL) on all four comparisons.
- CPU-vs-NPU cut comparison (CPU dumps regenerated on the same host with --device cpu and the same bundle, see cuts-vs-cpu.txt):
  - `reports/cuts-cpu-npu-turbo-chunker-254-200-hybrid.txt`: 0 documents differ on all four sets.
  - `reports/cuts-cpu-npu-turbo-chunker-254-200-cohesion-1000-hybrid.txt`: 4/10/6/7 documents differ on arguana/nfcorpus/qasper/scifact; each divergent document is listed with its cut positions and margins.

Metrics are not tabulated here. See summary.csv and the logs/ folder.

Host fields (host, hostname) were removed from provenance.machine in every record per receipt rule #1804.
