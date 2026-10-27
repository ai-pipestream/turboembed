# #1433: lost device on concurrent Level Zero loads

Machine B (Arc Pro B70, Battlemage G31, kernel driver `xe`, compute runtime
26.27.39122, GuC 70.72.1), booked for this run from 22:25 to 23:40 UTC on
2026-10-26. Command, from `core/`, for each run:

```text
TURBO_TEST_REQUIRE_LEVELZERO=1 <levelzero_loads test binary> --test-threads=1
```

built with `cargo test --release --locked -p turbo --features levelzero --test levelzero_loads --no-run`.
Each run loads `testdata/tiny-bert-bundle` from two, then four threads at
once, each thread with its own runtime and context, and asserts that the
levelzero backend listed the device.

## What failed before

`kernel-timeouts.log` holds every `xe` engine reset on the card that day
(local time, UTC-4). There were nine, six in `levelzero_loads` and three in a
`device` test binary. Each is `Timedout job ... engine_class=ccs`: a compute
job that had not finished within the engine's 5000 ms job timeout, always at
`lrc_seqno` 4294967169 to 4294967186, one of the first 18 jobs of a fresh
queue. The test then saw `zeCommandListHostSynchronize failed with 0x70000001`.
No page fault or other error is in the log.

All nine fell inside the four days snap-store ran: `snap-store-stop.log`
shows its scope ending at 18:32:21 local (22:32:21 UTC), after 3 d 23 h. The
one load sample taken while it ran, about 22:25:50 UTC, had it at 61% of the
render engine (`card-load-before.log`, copied from the terminal). Render and
compute share the card's execution units.

So the resets coincided with that client; it is the likely cause, not a
reproduced one. No load was recorded at the reset times (05:13 to 05:16 and
10:13 to 10:22 local), and no run was made with it up as a control.

## The runs

| Log | Binary | Started (UTC) | Runs | Failed |
| --- | --- | --- | ---: | ---: |
| `main-400.log` | main `29214c70` | 22:32:42 | 400 | 0 |
| `main-500.log` | the same | 22:36:00 | 500 | 0 |
| `head-500.log` | this branch's Rust sources at 1d1e242f, built before that commit | 22:38:13 | 500 | 0 |

Each log starts with its source and exact command. Each run takes about 0.25 s. At about 1 failure in 81, 1,400 clean runs would
happen by chance with probability below 0.01%. After the runs, the kernel log
had no engine reset or queue reset since 22:32 UTC.

`card-load.log` samples, every 10 s from 22:32:42 UTC (after snap-store stopped), the busy share of each engine by client
from drm fdinfo, and the number of active CI jobs (`workers=`). CI runners on
the machine were not paused: three to five jobs were active throughout, CPU
tests at the time sampled. One other client used the compute engine (2.4%, a
`device_real` test binary, in two samples during `main-500.log`).

## The change

No load path changed. A lost device now says what it means (device lost, the
driver reset the queue, likely a job timeout, the context cannot be used again)
instead of only the status code, and `docs/levelzero.md` says how to read one.
