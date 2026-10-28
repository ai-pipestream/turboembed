# turbo-trainer-recipes

The corpora `turbo-trainer` heads train on, the scoring every run
reports, and the `turbo-train-head` command (#1518, stage R1b).

| Corpus id | Data | Task | Pinned by | Splits |
|---|---|---|---|---|
| `masc-pos` | MASC 3.0.0, Penn tags (CC BY 3.0 US) | tags | corpus digest of the extracted release (`turbo_train_recipes::masc::RELEASE`) | test: the recipes' held-out documents (every tenth, Twitter genre left out), so numbers compare with the MASC tagger's gate; dev: every tenth training-side document |
| `multiconer2-en` | MultiCoNER v2 English (CC BY 4.0) | BIO, 33 types in 6 groups | size and SHA-256 of each file | the release's own train, dev and test |

A directory that is not the recorded release is refused unless the run
passes `--unrecorded-corpus`, and its receipt then says
`"recorded": false`. Neither corpus is checked in.

Every run scores the test split against the majority baseline (each
word's commonest training label, the commonest label overall for an
unseen word). Tags score by word accuracy, also over unseen words;
BIO by exact span F1, per type and per coarse group, spans read the
conlleval way. A trained head is compared with the baseline by a paired
bootstrap over sentences (1,000 resamples, seeded).

```text
turbo-train-head check    --corpus <id> --dir <corpus dir>
turbo-train-head baseline --corpus <id> --dir <corpus dir> --out <new dir>
turbo-train-head train    --corpus <id> --dir <corpus dir> --bundle <encoder bundle> --out <new dir>
    [--device cpu|cuda|levelzero|metal] [--precision model|exact] [--pooling first|mean]
    [--crf] [--bilstm <hidden>x<layers>] [--seed N] [--epochs N] [--batch N] [--lr X]
    [--patience N] [--max-train N] [--commit <sha>]
```

`train` reads per-word features from the encoder through
`turbo_tagger::WordEncoder`, the path the annotators use, trains with
`turbo_trainer::fit`, predicts the test split through the core's
`TokenTagger`, and writes `head.safetensors` (the head's F32 tensors
under `heads.<name>.`), `head.json` (its `HeadSpec`) and `receipt.json`
(`receipt::Receipt`: command, commit, host, corpus record with digests
and counts, encoder model and manifest digest, backend and device, head,
settings, per-epoch loss and dev score, test scores of model and
baseline, the bootstrap, timings, and the outputs' digests).

Majority baselines on the test splits (CPU, 2026-10-28, at the recorded
releases): MASC Penn POS accuracy 0.8964 (85,292 words), MultiCoNER v2
English span F1 0.1404 (377,805 gold spans).

```text
cargo test -p turbo-trainer-recipes
TURBO_TEST_MASC_DIR=<MASC-3.0.0> TURBO_TEST_MULTICONER_DIR=<multiconer2023> \
    cargo test --release -p turbo-trainer-recipes
```

Not yet here: bundle output (a sealed bundle carrying the head), the
learned mix of the last layers, WNUT-17 and MASC named entities.
