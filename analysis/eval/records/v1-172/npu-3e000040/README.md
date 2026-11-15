# v1-172 NPU cut records at 3e000040

Run on the Intel Core Ultra 7 265T (Arrow Lake) Windows NPU host at commit 3e000040ecf455b5237d4fd028de899b8f16ced3 on 2026-11-09. The NPU and CPU cut dumps were both produced on that host with the same bundle.

- reports/cuts-vs-cpu-*.txt: cut-by-cut comparison of the NPU dumps against the CPU dumps.
- cut-dumps.sha256: SHA-256 of the 16 cut dumps, eight under cpu/ and eight under npu/. The dumps themselves are not committed.

The earlier records in ../npu/ and ../reports/ come from the run at 89e6a6ae. Refs #1970.
