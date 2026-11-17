# v1-172 NPU cut records at 3e000040

Run on the Intel Core Ultra 7 265T (Arrow Lake) Windows NPU host at commit 3e000040ecf455b5237d4fd028de899b8f16ced3 on 2026-11-09. The NPU and CPU cut dumps were both produced on that host with the same bundle.

- reports/cuts-vs-cpu-*.txt: cut-by-cut comparison of the NPU dumps against the CPU dumps.
- cut-dumps.sha256: SHA-256 of the 16 cut dumps, eight under cpu/ and eight under npu/. The dumps themselves are not committed.

The earlier records in ../npu/ and ../reports/ come from the run at 89e6a6ae. Refs #1970.

Host paths were scrubbed from the committed reports (Refs #2395). The reports contain only turbo-eval's cut comparison output, written as UTF-8 with LF line endings. Captured cargo command-line fragments, compiler diagnostics, and the trailing cargo unsuccessful-exit footer were removed. Report data lines, cut positions, margins, and sha256 digests are unchanged.

The comparison was:

    turbo-eval cuts <npu-dumps>/turbo-chunker-254-200-hybrid <cpu-dumps>/turbo-chunker-254-200-hybrid
    turbo-eval cuts <npu-dumps>/turbo-chunker-254-200-cohesion-1000-hybrid <cpu-dumps>/turbo-chunker-254-200-cohesion-1000-hybrid

Placeholders:

- `<npu-dumps>` is the NPU cut-dump directory from the 2026-11-09 Intel host run (the first argv path).
- `<cpu-dumps>` is the CPU cut-dump directory regenerated on the same host (the second argv path).

`cut-dumps.sha256` already uses the repo-relative names `cpu/...` and `npu/...`. The dumps themselves are not committed.
