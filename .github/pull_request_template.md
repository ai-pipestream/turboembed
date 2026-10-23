ROLE: IMPLEMENTER (<model>)

Closes #<issue>

## What changed

Describe the change in full sentences: what it does, which files and
modules it touches, and why it is done this way.

## Proof

Paste the exact commands and their results. For a device claim, include
the `TURBO_TEST_REQUIRE_*` variable that was set, `TURBO_TEST_DEVICE`,
`TURBO_TEST_BUNDLE` and `TURBO_TEST_PRECISION`, the device name and class
the run reported, and the machine (Machine A, B or C, never a host name).
For a change on an existing path, include its timing against main.
Link any benchmark record added under `benchmarks/records/`.

```text
<commands and output>
```

## Not done in this change

Anything not finished, stubbed or deferred, with the issue that tracks it.
Write "none" if nothing remains.

## Checklist

- [ ] Every commit is authored and committed as `Kristian Rickert <krickert@gmail.com>`.
- [ ] No generated attribution trailer or footer (such as `Co-Authored-By`
      or a "generated with" line) in any commit, comment, doc or this
      description. The `ROLE:` and `VERDICT:` records below are required,
      not attribution.
- [ ] No em dash characters anywhere in the change.
- [ ] No real host names, user names or private paths.
- [ ] Every touched `.rs` file is under 500 lines.
- [ ] No silent fallback to another device, and no relaxed tolerance or
      refusal to make a test pass.
- [ ] Hardware proof is from the real device named in the issue.
- [ ] Review: two adversary reviewers, each posting `ROLE: REVIEWER (<model>)`
      and `VERDICT: PASS (<model>)` naming this pull request's current head,
      with no open `VERDICT: BLOCK`; CI `check` is green on that head.
- [ ] This will be merged with a merge commit, not squashed or rebased.
