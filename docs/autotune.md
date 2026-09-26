# Kernel choices and autotuning

A backend with more than one kernel for a step of the encoder chooses one
per session, when the session is made. This page says how a session's
choices are reported, forced, measured and cached, the same for every
backend that offers them; each backend's own page says what its choices
are and how it times them (docs/cuda.md, Kernel choices and Autotuning).

## What a session reports

`turbo_session_get_info` fills three fields:

- `choices`: the session's kernels as one line, in the form the
  backend's page gives. The core never parses it. A backend with one
  path reports it empty.
- `tuned`: where the choices came from. `TURBO_TUNED_DEFAULT`, the
  backend's own; `TURBO_TUNED_FORCED`, every choice named by the
  environment or a test, or in a tuned session every choice the
  backend would time; `TURBO_TUNED_MEASURED`, timed when this
  session was made; `TURBO_TUNED_CACHE`, timed for an earlier session
  with the same key and reused.
- `tune_ms`: the time spent measuring, 0 unless MEASURED.

A benchmark record carries the first two in `library.settings`, as
`TURBO_CUDA_TUNED` and `TURBO_CUDA_CHOICES` for the CUDA backend
(docs/benchmarks.md).

## The promise

The vectors are a function of the rows, the device, the driver, the
library build, the bundle and the session's kernel choices. The same
rows through sessions with the same choices on the same device, driver
and build give the same bits. A session's reported line, forced back
with the backend's `TURBO_<BACKEND>_CHOICES`, makes a session of the
same kernels.

Every kernel a precision may run computes in a numeric class the
precision allows (`TURBO_NUMERIC_*` in turbo_backend.h): F32 FMAs at
EXACT and MODEL; F16 operands with F32 sums, and F16 sums within a
chunk (a decision of 2026-09-26, recorded in docs/cuda.md), at FASTEST.
TF32 is in no precision's set; a backend's experiment switch widens
that session's set alone. So a different choice moves the bits within
the precision's bound, never past it.

## Switches

| Variable | Meaning |
|---|---|
| `TURBO_AUTOTUNE` | `off` (the default), `on`: the cached choice for the session's key, else measured now and cached; `retune`: measured now, the cached choice the one to beat. Read when `turbo_session_desc.tuning` is `TURBO_AUTOTUNE_RUNTIME`; any other value is refused, naming field 4. |
| `TURBO_AUTOTUNE_BUDGET_MS` | The time a session may spend measuring, when `turbo_session_desc.tuning_budget_ms` is 0: default 150 without a disk cache, 750 with one. Any value not a count of milliseconds is refused, naming field 5. |
| `TURBO_AUTOTUNE_CACHE` | A directory for the disk cache; unset, memory only. |
| `TURBO_<BACKEND>_CHOICES` | The backend's line: forces the items it names. |

A backend's own switches (for CUDA, `TURBO_CUDA_TILE` and the rest)
force their knob whether or not the session is tuned, and a forced
knob is never timed: forcing beats tuning. A tuned session with a
forced knob reports `forced=` naming it in its line.

## The cache

Each runtime keeps the choices its tuned sessions measured, in memory.
With `TURBO_AUTOTUNE_CACHE` naming a directory, each entry is also a
file there, `<backend>-<first 16 hex digits of the SHA-256 of the key's
JSON>.json`, written beside and renamed over, holding the key, the
choices, `tuned_at`, `tune_ms` and the timings the choice was made on.
The library never writes anywhere else: without the variable it writes
no file.

The key is everything that decides a choice or changes the bits: the
backend, the device's name and arch, the driver and runtime versions,
the library's build (its version and a hash of the backends' kernel
sources, so changed kernels are a new entry), the bundle's manifest and
artifact hashes, the precision and compute dtype, the precision's
numeric classes, the highest token bin the session has and max_seq
rounded up to a power of two.

Only a measured session with nothing forced and no class run beyond
its precision's is cached; any other is neither looked up nor stored.
A file that is missing, does not parse or holds another key is no
entry, and the session measures: never a failed load. A measurement
refused for a busy device is not cached, and the next session measures
again; a forced line, or `off`, removes that one way two unforced
sessions can differ. Across processes without a disk cache, a record's
choices line forced back reproduces a run.

## A backend's side

A backend offers `session_create_tuned` in its table
(turbo_backend.h) and fills the tuning struct's out fields: `tuned`,
`tune_ms`, its line in `choices`, its timings as `<bin>/<knob>/<variant>=
<least ms>` lines, and `numerics_used`, the precision's classes and any
class a kernel it chose computes in beyond them. It takes `cached`, when
the core hands one and the session forces and widens nothing, as its
incumbent: unmeasured under ON, the one to beat under RETUNE. A backend
without the function is made sessions through `session_create`, and
reports DEFAULT and no choices.
