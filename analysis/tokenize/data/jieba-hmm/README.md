# Jieba unknown-word model

`bmes.bin` contains the pretrained BMES start, transition and emission
log probabilities published in
[fxsjy/jieba](https://github.com/fxsjy/jieba/tree/67fa2e36e72f69d9134b8a1037b83fbb070b9775/jieba/finalseg),
revision `67fa2e36e72f69d9134b8a1037b83fbb070b9775`. The upstream
[MIT licence](https://github.com/fxsjy/jieba/blob/67fa2e36e72f69d9134b8a1037b83fbb070b9775/LICENSE)
is copied in `LICENSE`. No corpus text is included. The upstream files
supply the trained parameters, but do not identify a training corpus or
training run; no claim about that training provenance is made here.

`generate.py` checks the SHA-256 of each input and reads its `P` dictionary
with `ast.literal_eval`, without executing the input. It stores states in
B, E, M, S order. Impossible starts and transitions become negative infinity;
the upstream missing-emission value remains `-3.14e100`.

The binary is 599,488 bytes: four f64 starts, sixteen f64 transitions,
65,536 u16 row indices, and 14,633 rows of four f64 emissions. Row zero is
unobserved. All numbers are little endian. The decoder accesses the bytes
directly and keeps only the Viterbi back pointers in caller-owned scratch.
Characters absent from the model remain individual tokens; they do not
enter an all-missing-emission path. Only characters classified as Han by
`turbo-unicode` enter this model.

```text
b72dbb9f25a088473b0f7b4bb892514bbadbd8a510d6360449c29b0e64054bd5  bmes.bin
14c5706ced5cd3b42eb4873d4b88f7f52a7bdf80fbd767bc4423d361e20c5330  prob_start.py
54dfbc252ed71480d4f0cdfdf516ecfbe44efd0f6c3c64b158e7039f2906c91b  prob_trans.py
27d46b1c9efe4dd148fde8be042a21be40e3562d0c7f1273f9de7abae12ebb8d  prob_emit.py
```

Download those three Python files from the pinned `jieba/finalseg` directory,
then run `python3 generate.py INPUT_DIRECTORY` to reproduce the binary.

`test-dict.txt` contains 41 unchanged rows of that revision's `jieba/dict.txt`:
all words that are substrings of the concatenation below. It is a real
source subset, not a fitted test model. Removing other words changes the
dictionary total and therefore can change the best segmentation. The
integration assertions were checked against jieba-rs 0.11.0 with this same
subset; corpus measurements use the entire upstream dictionary.

```text
小明硕士毕业于中国科学院计算所他来到了网易杭研大厦abc123北京2026年𠀀🙂 ９９ＡＢ12.5%
```
