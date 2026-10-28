# #172 chunker v1 records: Metal (Apple M2)

Commit 16e74c040898d961d0c50e4e9bf1412f1722f86c (claude/172-dump-cuts), release build, `--device metal`;
every record's provenance names backend `metal` on the Apple M2 and this commit.
Bundle all-MiniLM-L6-v2, manifest sha256 0f17adc4b795ed27bbe59ec88d35d37bf10f25af15f07b09f3249e5c1001b6d3.
Sets scifact, nfcorpus, arguana, qasper; retrieval hybrid. TURBO_GOLDENS_MODELS held
opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin. Records were written outside the tree with `--out` (the tool refuses
a dirty tree) and copied here unchanged; logs have the scratch directory replaced by `$SCRATCH`.

Command per pipeline:

```
cargo run --release -p turbo-eval --features turbo/metal -- run --bundle <bundle> \
  --sets scifact,nfcorpus,arguana,qasper --cache <cache> --device metal --retrieval hybrid \
  --out <out>/<name> [--dump-cuts <dumps>/<name>] <chunker flags>
```

| Pipeline | Chunker flags | Wall (s) | Load 1/5/15 at start | GPU busy at start |
|---|---|---|---|---|
| turbo-chunker-254-200-hybrid | --chunker turbo-chunker --max 254 --target 200 (cuts dumped) | 299.8 | 9.47 32.92 24.33 | 24% |
| turbo-chunker-254-200-cohesion-1000-hybrid | --chunker turbo-chunker --max 254 --target 200 --cohesion 1000 (cuts dumped) | 710.7 | 8.48 18.16 19.95 | 20% |
| sentence-windows-5-hybrid | --chunker sentence-windows --window 5 | 321.4 | 4.96 10.02 14.23 | 0% |
| fixed-tokens-200-hybrid | --chunker fixed-tokens --tokens 200 --overlap 0 | 337.8 | 6.85 7.53 11.58 | 12% |
| opennlp-sandbox-semantic-hybrid | --chunker opennlp-sandbox-semantic | 572.4 | 8.03 8.16 10.36 | 17% |

Host load: the M2 is a busy desktop (desktop apps on the GPU, outside builds). These are quality records, not
timing receipts; wall times above are context only and not a speed claim. Load and GPU utilisation before and
after each run are in `logs/`.

Cut dumps (not committed, about 107 MB per pipeline) are on the B70 host under
eval-cuts-172/metal/<name>/<set>.cuts. sha256:

```
9775d44d55477ec71b235d59b41078d941f057746601f708a3c4191393cc88ed  turbo-chunker-254-200-cohesion-1000-hybrid/arguana.cuts
8b2dc617bbdb11e797113e3eb33090db8f819e52732f17a70852faeb228d9af9  turbo-chunker-254-200-cohesion-1000-hybrid/nfcorpus.cuts
a19a96913fc885b44b80c429ffa3bd09e8f57cc4dd5fd4391bbac37385f39d80  turbo-chunker-254-200-cohesion-1000-hybrid/qasper.cuts
92f709d695b850abd692c0dddb1874bc31b2fb61597cefa1dd8d51f274a7acaa  turbo-chunker-254-200-cohesion-1000-hybrid/scifact.cuts
9a5a5ef46b6b63a8783bc8a9af037ff99302fe1c3b94378f85f5dc4ea90eee05  turbo-chunker-254-200-hybrid/arguana.cuts
21326dd638bed5386f243aa21060debaff7d6d5acb1e7cc9035a53ba3e7cc260  turbo-chunker-254-200-hybrid/nfcorpus.cuts
17b87654c86e950d9a3252af0055d280b69829b1e11efbf0fad2842bad8a137c  turbo-chunker-254-200-hybrid/qasper.cuts
97b97b6e7a3dac17c9a088dea9551fdbd6cf27c70d1c15aab0caa1577e18f4d2  turbo-chunker-254-200-hybrid/scifact.cuts
```
