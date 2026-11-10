# Demo catalog filter receipt (#2118)

The filter over the Models catalog on the demo page
(`demo/web/src/catalogfilter.ts`). Machine: a 4-core x86_64 Linux
container, no GPU.

- `red.log`: `scripts/build-web.sh` with the new unit tests against the
  filter as it was on main at a4bd8d08. 23 pass, 4 fail: white space
  other than a space, the fold, both sharp s letters and both sigmas,
  and the enum spelling of a kind.
- `green.log`: the same run with the change. 27 pass.
- `desktop-filter.png`: the Models tab at 1280 px in Chromium through
  Playwright, against `turbo-kserve` on the CPU and a demo started with
  `--user`, after "Read the catalog" with `BGE`, a no-break space and
  `small` typed in the filter. Above, main: "0 of 46 entries". Below,
  the change: "4 of 46 entries".

`scripts/e2e-web.sh` passes 4 of 4 with the change.
