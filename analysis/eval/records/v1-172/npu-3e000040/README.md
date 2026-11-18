# v1-172 NPU cut records at 3e000040

Run on the Intel Core Ultra 7 265T (Arrow Lake) Windows NPU host at commit 3e000040ecf455b5237d4fd028de899b8f16ced3 on 2026-11-09. The NPU and CPU cut dumps were both produced on that host with the same bundle.

- reports/cuts-vs-cpu-*.txt: cut-by-cut comparison with the CPU dumps as A and the NPU dumps as B, the same order as ../reports/cuts-cpu-npu-*.txt. A positive margin for A means the CPU side scored its cut higher.
- cut-dumps.sha256: SHA-256, size in bytes and repo-relative path of the 16 cut dumps, eight under cpu/ and eight under npu/, in the same three-column format as ../reports/cut-dumps.sha256. The dumps themselves are not committed.

The earlier records in ../npu/ and ../reports/ come from the run at 89e6a6ae. Refs #1970.

The reports are turbo-eval's standard output only, written as UTF-8 with LF line endings. They were produced by running the release turbo-eval binary built at 3e000040 directly, so they carry no cargo output and no host paths. Standard error was empty for both runs.

The comparison was:

    turbo-eval cuts <cpu-dumps>/turbo-chunker-254-200-hybrid <npu-dumps>/turbo-chunker-254-200-hybrid
    turbo-eval cuts <cpu-dumps>/turbo-chunker-254-200-cohesion-1000-hybrid <npu-dumps>/turbo-chunker-254-200-cohesion-1000-hybrid

The hybrid comparison exited 0 with no document differing. The cohesion comparison exited 1, and its report lists the documents that differ.

Placeholders:

- `<cpu-dumps>` is the CPU cut-dump directory regenerated on the same host (the first argv path).
- `<npu-dumps>` is the NPU cut-dump directory from the 2026-11-09 Intel host run (the second argv path).
