# The BiLSTM on Metal (#860)

Device: Apple M2 through the metal backend, F32 (MODEL). Release builds,
`TURBO_TEST_DEVICE=metal TURBO_TEST_REQUIRE_METAL=1`. Each log's first line
names its commit, command, time and load.

- `device_lstm.log`: `core/tests/device_lstm.rs`, twenty consecutive runs,
  9 tests each, all passing; the worst difference from the CPU form
  5.1e-7 every run (bound 1e-5); FASTEST refused.
- `tagger.log`: `analysis/postag` `neural_goldens` and `bilstm_device`: the
  goldens' 1980 tags on Metal(0), packed emissions within 8.6e-7 and tag
  probabilities within 1.5e-7 of the JVM's, the device emissions within
  1.8e-7 of the CPU's; FASTEST refused.
- `ewt.log`: `bilstm_ewt` `device_tiers_on_ud_english_ewt` on UD English EWT
  r2.15's test split (`en_ewt-ud-test.conllu`, SHA-256
  `0612e45914359e4b36d187ef2bfeaf66fc8933210925be985c5fbf6a9e311036`, 25,094
  tokens as the test reads them; not in the repository), with an
  `ONLP-BLPT` model trained by `bilstm_ewt`'s own trainer at its defaults
  (SHA-256 `fcf0b057bf1e340d08292026cfaf581a6c30a798019cefb21e5c9ba6b3502eab`,
  not in the repository): MODEL on Metal gives the M2 CPU packed path's tag
  on all 25,094 tokens (94.8553% each); FASTEST refused. The claim is Metal
  against the CPU on this machine alone. The same model on CUDA (#1420) gave
  94.9032% against a test file of another SHA-256 (`fa024f43...`) of the
  same token count, so the two accuracies are not of one split and are not
  compared here; the repository's documented model (`819328e7`) is another
  model again.
- `timing.log`: `bilstm_device` `device_tagger_timing`, 30 calls a cell, the
  CPU packed path and Metal interleaved, with the load before and after
  (one-minute 5.6 to 5.7: a desktop in use). Metal is slower than the CPU
  at every size; docs/metal.md says why.
