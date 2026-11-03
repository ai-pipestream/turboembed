# #172 chunker v1 records: Metal (Apple M2)

Commit 0d58e8629e85345b5c585c4beffa39b66a63c248 (main, with the rule sentence splitter turbo-sentdetect:rules en v3),
release build, `--device metal`. Every record's provenance names backend `metal` on the Apple M2 and this commit.
Bundle all-MiniLM-L6-v2, manifest sha256 0f17adc4b795ed27bbe59ec88d35d37bf10f25af15f07b09f3249e5c1001b6d3.
Sets scifact, nfcorpus, arguana, qasper; retrieval hybrid. No TURBO_GOLDENS_MODELS (the rule splitter needs no model).
Records were written outside the tree with `--out` (the tool refuses a dirty tree) and copied here unchanged;
logs have the scratch directory replaced by `$SCRATCH`. These replace the earlier Metal records taken at 16e74c04.

Command per pipeline:

```
cargo run --release -p turbo-eval --features turbo/metal -- run --bundle <bundle> \
  --sets scifact,nfcorpus,arguana,qasper --cache <cache> --device metal --retrieval hybrid \
  --out <out>/<name> [--dump-cuts <dumps>/<name>] <chunker flags>
```

| Pipeline | Chunker flags | Load 1/5/15 at start | GPU busy at start |
|---|---|---|---|
| turbo-chunker-254-200-hybrid | --chunker turbo-chunker --max 254 --target 200 (cuts dumped) | 14.35 6.97 5.06 | 45% |
| turbo-chunker-254-200-cohesion-1000-hybrid | --chunker turbo-chunker --max 254 --target 200 --cohesion 1000 (cuts dumped) | 6.11 6.56 5.25 | 0% |
| sentence-windows-5-hybrid | --chunker sentence-windows --window 5 | 3.70 5.18 5.00 | 0% |
| fixed-tokens-200-hybrid | --chunker fixed-tokens --tokens 200 --overlap 0 | 3.36 4.15 4.56 | 0% |
| opennlp-sandbox-semantic-hybrid | --chunker opennlp-sandbox-semantic | 5.37 5.03 4.77 | 36% |

The M2 is a shared desktop. These are quality records only and make no timing claim. Load and GPU utilisation
before and after each run are in `logs/`.

Cut dumps (not committed, about 107 MB per pipeline) are on the B70 host under
eval-cuts-172-main/metal/<name>/<set>.cuts. sha256:

```
dc90c4a7f3d456cbff9215d35006b2fbf80248bb26b93438dbcd5f332787ba8e  turbo-chunker-254-200-cohesion-1000-hybrid/arguana.cuts
30720d47916c9f4048c64a60c3836968cf925b158c96c06e97bcd8b2be7ce8e9  turbo-chunker-254-200-cohesion-1000-hybrid/nfcorpus.cuts
b6e9e04314c27fdf4200cfda319823a8a2e674cb10c886c20a360dc2a756e2c2  turbo-chunker-254-200-cohesion-1000-hybrid/qasper.cuts
18549d73f992b22e24d8a338496a71da1a5913a0e05788b01a7b5b53ecdba08d  turbo-chunker-254-200-cohesion-1000-hybrid/scifact.cuts
441808cdc3a64195c1f6a3ccc01cf17796f5dfde5540af84aa667b6da0dee94f  turbo-chunker-254-200-hybrid/arguana.cuts
18f9fcf502f817bd0c0ae536b588f21f68caa08d9596db3c4164b1d04798fb1a  turbo-chunker-254-200-hybrid/nfcorpus.cuts
640cc529cf3655d31c4aaf98117f5acb7612763b5f82d67ef347b721452e5ba0  turbo-chunker-254-200-hybrid/qasper.cuts
5b9a9a1dcf8b568cf4f114720d6dafb7e17a5e84010e751aec6ba7a668eb00c2  turbo-chunker-254-200-hybrid/scifact.cuts
```
