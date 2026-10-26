ROLE: IMPLEMENTER (<model>)

Closes #<issue>
<!-- or: Part of #<issue> -->

## What changed

Describe the change in full sentences: what it does, which files and
modules it touches, and why it is done this way.

## Red/Green receipt

**Red:** the test or measurement that fails, or the behaviour that is
wrong, on main or before the change, with the command that shows it.

**Green:** the same commands passing at `<head sha>`.

For a device claim, include the `TURBO_TEST_REQUIRE_*` variable that was
set, `TURBO_TEST_DEVICE`, `TURBO_TEST_BUNDLE` and `TURBO_TEST_PRECISION`,
the device name and class the run reported, and the machine (Machine A,
B or C, never a host name). For a change on an existing path, include its
timing against main. Link any benchmark record added under
`benchmarks/records/`.

```text
<commands and output>
```

## Not done in this change

Anything not finished, stubbed or deferred, with the issue that tracks it.
Write "none" if nothing remains.

## Checklist

Gates:

- [ ] `cargo fmt --all --check` passes (max_width 120).
- [ ] `cargo clippy -p <crate> --all-targets -- -D warnings` passes on every crate touched.
- [ ] `cargo test -p turbo-bench --test tree` passes.
- [ ] Every touched `.rs` file is under 500 lines.
- [ ] No new regex runtime dependency.
- [ ] README and docs paths are kept in step with any file moved or renamed.
- [ ] No em dash characters anywhere in the change.
- [ ] No real host names, user names or private paths.
- [ ] No silent fallback to another device, and no relaxed tolerance or
      refusal to make a test pass.
- [ ] Hardware proof is from the real device named in the issue.

Issue labels:

- [ ] The issue carries `agent:<model>`, and moved from `status:ready` to
      `status:in-progress` when it was claimed.
- [ ] With this pull request up, the issue is `status:needs-review`, with
      an "Opened PR #N at `<sha>`" comment.
- [ ] On merge: a `Closes` pull request sets the issue to `status:done`; a
      `Part of` pull request leaves a summary comment on the parent issue.

Review and merge:

- [ ] Two adversary reviewers each post `ROLE: REVIEWER (<model>)` and
      `VERDICT: PASS (<model>)` naming this pull request's current head,
      with no open `VERDICT: BLOCK`. A verdict does not count on a head
      that holds the reviewer's own commit, and a reviewer who pushes a
      fix voids their own verdict, so the implementer applies review fixes.
      The one exception is `krick-1`: its `PASS` counts on a head carrying
      its own commits, and the other `PASS` then comes from an agent with
      no commit on that head.
- [ ] CI `check` is green on that head. The second passing reviewer merges
      with a merge commit, never squash or rebase, and deletes the branch.
