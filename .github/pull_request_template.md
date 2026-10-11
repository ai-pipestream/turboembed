Closes #<issue>

## What changed

Describe the change in full sentences: what it does, which files and
modules it touches, and why it is done this way.

## Proof

Paste the exact commands and their results. For a device claim, include
the `TURBO_TEST_REQUIRE_*` variable that was set, `TURBO_TEST_DEVICE`,
`TURBO_TEST_BUNDLE` and `TURBO_TEST_PRECISION`, the device name and class
the run reported, and the machine (Machine A, B or C, never a host name).
Link any benchmark record added under `benchmarks/records/`.

```text
<commands and output>
```

## Remaining work

Anything not finished, stubbed or deferred, with the issue that tracks it.
Write "none" if nothing remains.

## Checklist

- [ ] Every commit is authored and committed as `Kristian Rickert <krickert@gmail.com>`.
- [ ] No commit, comment, doc or this description carries a trailer, a
      footer or any line naming a tool or assistant, and any footer added
      when this pull request was opened has been removed.
- [ ] No em dash characters anywhere in the change.
- [ ] No real host names, user names or private paths.
- [ ] No silent fallback to another device, and no relaxed tolerance or
      refusal to make a test pass.
- [ ] Hardware proof is from the real device named in the issue.
- [ ] Adversary review requested (`adversary-review` label added).
- [ ] This will be merged with a merge commit, not squashed or rebased.
