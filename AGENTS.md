# Working rules for agents

This file is the set of rules every coding agent follows when it works on
TurboEmbed, whether it runs on one of the project's own machines or in a
cloud checkout of its own. It covers how work is
handed out, how agents talk to each other, how work is proven and how it
reaches `main`. It does not restate how the library works.

Read these first, in this order, before touching the tree:

1. [`README.md`](README.md), especially "Project principles" and "Speed
   over elegance". Nothing here overrides it.
2. The docs page for the area you are changing, listed in the README's
   Documentation table (for example [`docs/bundle.md`](docs/bundle.md),
   [`docs/conformance.md`](docs/conformance.md),
   [`docs/benchmarks.md`](docs/benchmarks.md), or the backend page such as
   [`docs/npu.md`](docs/npu.md)).
3. [`docs/decisions.md`](docs/decisions.md), the decisions already made.
   A pull request does not reopen one; a question about one goes on the
   issue first.
4. The GitHub issue you were assigned and, on a project machine, the
   `INSTRUCTIONS.md` in your worktree.

Work on the document analysis crates (`analysis/`) and the OpenNLP port
also follows section 14.

The earlier attempt in `ai-slop-generated-shit/` has its own `AGENTS.md`
and `CONTRIBUTING.md`. They are kept for reference only and describe a
tree that is no longer built. Where they agree with the README (no
attribution lines in commits, no silent device fallback, no private host
names in public text, no Python in the tree), they still hold. Where they
describe crates, scripts or a layout that the current tree does not have,
ignore them. If this file and the README ever disagree, the README wins
and this file is fixed.

## 1. How work is organized

GitHub is the only coordination layer. Every piece of work has exactly one
of each of the following, and they are tied together by the issue number:

| Thing | Where it lives | What it is for |
|---|---|---|
| Issue | GitHub Issues | The assignment: scope, decisions, blockers, final summary. The source of truth. |
| Branch | `origin` | The work itself, named `<issue>-<slug>`. |
| Worktree | The assigned machine | An isolated checkout of that branch, owned by one agent. |
| `INSTRUCTIONS.md` | The root of that worktree, untracked | The brief the agent works from. |
| Draft pull request | GitHub Pull Requests | Opened early; progress, proof and review happen here. |

Architecture notes, design writeups and research results live in `docs/`,
and the decision log is [`docs/decisions.md`](docs/decisions.md). They are
in the tree rather than the GitHub wiki so that every agent, including
ones that only see a cloud checkout, reads them, and so a change to them
is reviewed in a pull request like any other. Issues link to docs pages;
a decision entry links back to the issue or pull request that settled it.
A research issue (`type:research`) ends with a merged docs page, not code.

## 2. Worktrees and INSTRUCTIONS.md

The coordinator (the agent labelled `agent:turboembed`, or Kristian)
creates each worktree from a fresh `origin/main`:

```sh
git fetch origin
git worktree add -b <issue>-<slug> <worktrees-root>/<issue>-<slug> origin/main
```

Then it writes `INSTRUCTIONS.md` at the root of that worktree. The file is
listed in `.gitignore` and is never committed: it is a local brief, and
the issue holds the same content in public. If the brief changes, the
issue is updated first and the file second.

Agents that run in a cloud checkout of their own (for example a cloud
coding agent started from the issue) have no worktree on a project
machine and no `INSTRUCTIONS.md`: the issue body is their brief, and the
worktree rules below do not apply to them. Everything else in this file
does.

Rules for worktrees:

- One worktree, one branch, one issue, one owning agent.
- Never edit, build in, reset, stash or delete another agent's worktree,
  even to fix something obvious. Say what you found on that agent's issue
  or pull request instead.
- Never commit from the primary checkout, and never leave uncommitted
  changes there. It belongs to Kristian.
- Before you start, run `git fetch origin` and check that your branch is
  based on a recent `origin/main`. To pick up later changes from `main`,
  merge `origin/main` into your branch. Do not rebase a branch that has
  already been pushed.
- When the pull request is merged or closed, the coordinator removes the
  worktree with `git worktree remove`. Agents do not remove worktrees.

### INSTRUCTIONS.md template

Copy this into the worktree and fill in every field. A field that does not
apply says "none" and why, so the reader can tell it was considered.

```markdown
# <issue number>: <issue title>

Issue: https://github.com/ai-pipestream/turboembed/issues/<number>
Branch: <issue>-<slug>
Owning agent label: agent:<name>
Machine: <Machine A | Machine B | Machine C | a Hailo board>, class <rtx4080super | b70 | arl-npu | pi5-hailo10h | ...>
Merge: <coordinator merges | owning agent is cleared to merge>

## Scope
What this change does, in full sentences. Name the files or modules that
are expected to change.

## Out of scope
What this change must not touch, and related work that belongs to another
issue (link it).

## Definition of done
- [ ] The behaviour described in Scope works, with a test that fails
      without the change.
- [ ] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`
      and `cargo test --workspace` pass, plus the feature-gated lines from
      `.github/workflows/ci.yml` that apply.
- [ ] Docs updated (the docs page for the area, and `docs/decisions.md`
      if a decision was made).
- [ ] Hardware proof below is attached to the pull request.
- [ ] Adversary review finished with no open findings.

## Hardware proof required
The exact commands to run on the machine above, including every
`TURBO_TEST_REQUIRE_*`, `TURBO_TEST_DEVICE`, `TURBO_TEST_BUNDLE` and
`TURBO_TEST_PRECISION` value, and any benchmark record expected under
`benchmarks/records/`. Write "none: <reason>" only for work that makes no
device claim.

## Context
Links to docs pages, decisions, earlier issues and pull requests.
```

## 3. Branch names

A branch is named `<issue>-<slug>`: the issue number, a hyphen, and a few
lowercase words joined by hyphens that say what the change is. For
example, `142-chunk-planner-api` or `150-npu-two-models-one-context`. One
branch per issue. If the work turns out to need two independent changes,
open a second issue and a second branch rather than growing the first.

Some agents choose their own branch prefix and cannot be told otherwise
(`claude/`, `cursor/`, `codex/` and so on). They name the branch
`<prefix>/<issue>-<slug>`. The pull request's `Closes #<issue>` line, not
the branch name, is what ties the work to its issue.

## 4. Labels

Every issue carries one label from each of these groups. Pull requests
carry the same labels as their issue.

| Group | Labels | Meaning |
|---|---|---|
| Owner | `agent:<name>` | The one agent that owns the issue. `agent:turboembed` is the coordinator. There is exactly one owner label at a time. The names are `agent:claude`, `agent:cursor`, `agent:codex` (ChatGPT), `agent:junie`, `agent:kimi`, `agent:agy` (Antigravity), `agent:muse` and `agent:grok`. |
| State | `status:ready`, `status:in-progress`, `status:blocked`, `status:needs-review`, `status:needs-hardware-proof`, `status:ready-to-merge` | Where the work is. There is exactly one state label at a time. |
| Area | `area:analysis`, `area:chunking`, `area:embedding`, `area:core`, `area:bundle`, `area:bench`, `area:grpc`, `area:ffi`, `area:docs`, `area:ci` | Which part of the tree. More than one is allowed. `area:analysis` is the analysis crates and the OpenNLP port. |
| Stage | `stage:0` to `stage:7` | Analysis work only: the landing stage from the analysis design, which is also the issue's milestone. |
| Hardware | `hw:cuda`, `hw:levelzero`, `hw:npu`, `hw:metal`, `hw:cpu`, `hw:hailo` | Which backend the work touches and so which device must prove it. More than one is allowed. |
| Type | `type:feature`, `type:bug`, `type:research`, `type:chore`, `type:epic` | What kind of work it is. An epic is a stage's parent issue: it lists the work as sub-issues, belongs to the coordinator, and is never claimed by a coding agent. |
| Priority | `priority:p0`, `priority:p1`, `priority:p2` | Order of work. `p0` blocks other work. |
| Review | `adversary-review` | The work is waiting for, or under, adversary review. |

The state labels move in this order, and only the listed party moves them:

1. `status:ready`: the coordinator, once the issue is scoped and the
   worktree exists.
2. `status:in-progress`: the owning agent, when it claims the work.
3. `status:blocked`: the owning agent, whenever it cannot proceed (see
   section 7). It goes back to `status:in-progress` when unblocked.
4. `status:needs-hardware-proof`: the owning agent, when the code is done
   but the device run has not happened yet.
5. `status:needs-review`: the owning agent, when the code and its proof
   are both on the pull request. The agent also adds `adversary-review`.
6. `status:ready-to-merge`: the reviewer, once adversary review is clean.

## 5. Claiming work

An agent claims an issue only when the coordinator or Kristian has pointed
it at that issue. To claim it:

1. Check that the issue has `status:ready` and no other agent label.
2. Add your `agent:<name>` label, swap `status:ready` for
   `status:in-progress`, and comment on the issue with what you understood
   the task to be, the machine and worktree you are using, and your first
   step. If your reading of the scope differs from the issue, say so now,
   before writing code.
3. Push the branch and open a draft pull request within the first working
   session, even if it only has the first commit. Its body starts with
   `Closes #<issue>`.

## 6. Status updates and communication

Agents talk to each other only through issues and pull requests. There is
no other channel, and nothing an agent learns in a private session counts
until it is written down on GitHub.

- Post a progress comment on the pull request at the end of every working
  session: what changed, what was run and the result, and what is next.
- Put design questions on the issue, not the pull request, so the decision
  stays with the assignment. Record the answer there too.
- To ask another agent something, comment on that agent's issue or pull
  request and link your own. Do not change its code, its labels or its
  worktree.
- When a decision affects more than one issue, add an entry to
  `docs/decisions.md` in the pull request that acts on it, and link it
  from each issue.
- Write in complete sentences with enough context that a reader who has
  never seen your session can follow. Name the file, the function, the
  command and the result. Short status lines such as "done" or "fixed it"
  are not updates.

## 7. When you are blocked

Being blocked is normal. Hiding it is not. When you cannot proceed:

1. Swap your state label for `status:blocked`.
2. Comment on the issue with what you were trying to do, the exact command
   and its full error output, what you already tried, and what would
   unblock you (a decision, a device, a change in another issue).
3. If another issue is the cause, link it and comment there too.
4. Stop. Do not work around the blocker by weakening a test, skipping a
   device check, adding a fallback, or quietly narrowing the scope.

If the task cannot be finished at all, say what was done, what was not and
why, leave the branch pushed, and stop.

## 8. Commits

These rules are absolute. A pull request that breaks one is sent back.

- Every commit is authored and committed as
  `Kristian Rickert <krickert@gmail.com>`, both fields. Set it in the
  worktree before the first commit:

  ```sh
  git config user.name "Kristian Rickert"
  git config user.email "krickert@gmail.com"
  ```

- No trailers and no attribution of any kind: no `Co-authored-by` (not
  even one naming Kristian under another address), no `Made-with`, no
  `Generated-by`, no `Signed-off-by` from a tool, and no line that names a
  tool, an assistant or an agent product. The same rule applies to code
  comments, docs, issue comments and pull request bodies.
- Some machines have a `commit-msg` hook that appends a `Co-authored-by`
  line. Check every commit after you make it. If a trailer appeared, amend
  the commit with `git commit --amend --no-verify` before pushing, and use
  `--no-verify` for later commits in that worktree.
- Check before every push:

  ```sh
  git log --format='%an <%ae> | %cn <%ce>%n%B' origin/main..HEAD
  ```

  Every header line must read
  `Kristian Rickert <krickert@gmail.com> | Kristian Rickert <krickert@gmail.com>`,
  and no message may contain a trailer.
- Some tools add a footer to a pull request body when they open it, such
  as an "Open in ..." button or a "generated with" line. Edit the body
  right after creating the pull request and remove it.
- Commit messages are `area: a sentence`, where the area is the part of
  the tree (for example `npu`, `cuda`, `bundle`, `bench`, `core`, `docs`,
  `ci`) and the sentence says what the commit does, for example
  `npu: refuse a second graph in one context with the driver's code`.
  Add a body when the reason is not obvious from the diff.
- One logical change per commit.
- Never use the em dash character anywhere: commits, code, comments, docs,
  issues or pull requests. Use a comma, a colon, parentheses or a
  new sentence.

## 9. Engineering rules

These come from the README and the docs; they are repeated here because
they are the ones most often broken.

- Nothing is done until it works. A stub, a mock, a placeholder or a
  hard-coded result is never presented as a finished feature. If part of
  the work is a stub, the pull request says so in its first paragraph and
  the issue stays open.
- A device claim needs proof from that device. Run the tests with the
  matching `TURBO_TEST_REQUIRE_*` variable set (`TURBO_TEST_REQUIRE_CUDA`,
  `TURBO_TEST_REQUIRE_LEVELZERO`, `TURBO_TEST_REQUIRE_NPU`,
  `TURBO_TEST_REQUIRE_METAL` or `TURBO_TEST_REQUIRE_HAILO`, described in
  each backend's docs page) so a missing device fails the run instead of
  skipping it, run conformance on that device (docs/conformance.md), and attach the commands and output. A
  performance claim needs a record in `benchmarks/records/` written by
  `turbo-bench` (docs/benchmarks.md). Records are never typed or edited
  by hand.
- A selected device that is absent or cannot run the model is an error
  that names the reason. Never fall back to the CPU or to another device
  silently, and never add a code path that does.
- The NPU computes F16 only. It has no `EXACT` tier, and the library
  refuses `EXACT` on it. Do not change that refusal, and do not make a
  test pass by relaxing it.
- Seal rule: `turbo-bundle seal` runs only on files produced inside the
  pinned container named in the recipe, driven from Rust, never on files
  made by hand or on the host. Any container image built on a machine that
  may not survive (a cloud runner, a temporary VM) is saved with
  `docker save` or pushed to a registry in the same session that built it,
  and the pull request says where it went.
- No Python in the tree. Vendor Python tools run in pinned containers,
  driven from Rust (README, "Project principles").
- The C header is designed by hand. Changing `include/turbo/` is a design
  decision that goes on the issue and in `docs/decisions.md` first.

## 10. Public text and machine names

The repository, issues and pull requests are public.

- Never write a real host name, user name, IP address or private checkout
  path. Refer to machines as Machine A, Machine B or Machine C, or by
  device class (`rtx4080super`, `b70`, `arl-npu`, `m2`, `orin`,
  `pi5-hailo10h`), the same names the benchmark records use.
- Code comments must make sense to someone reading the file cold. Do not
  write comments that point at a conversation, a session, a review round
  or "the issue above" without a link. Say what the code does and why.
- Do not be overly terse. Commit bodies, comments and pull request text
  are written in plain, complete prose.

## 11. Shared machines

The project machines are shared with other agents and with Kristian.

- Never kill, renice or otherwise disturb a process you did not start,
  especially one holding a GPU or NPU. If a device is busy, wait or report
  it as a blocker.
- Do not change drivers, system packages or global configuration without
  an issue that asks for it.
- Keep build output inside your worktree's `target/`. Clean up temporary
  files you created outside it.

## 12. Review and the merge gate

- Before asking for review, review your own diff as an adversary would:
  look for silent fallbacks, unreported host copies, relaxed tolerances,
  tests that cannot fail, missing hardware proof, attribution markers and
  em dashes. Fix what you find.
- Adversary review is done by a different agent from the one that wrote
  the code, or by Kristian. The reviewer comments with every finding,
  each tied to a file and line or to a missing proof. The author answers
  each one on the pull request. Review is clean when no finding is open.
- There is one merge gate. The coordinator merges, unless Kristian has
  cleared the owning agent to merge its own pull request, which is
  recorded in the issue's `INSTRUCTIONS.md` "Merge" field and in an issue
  comment.
- Merge only with a merge commit. Never squash and never rebase-merge.
- Merge only when the pull request has `status:ready-to-merge`, CI is
  green, the hardware proof is attached and the branch is up to date with
  `main`.
- After merging, the merger comments on the issue with the merge commit
  hash and closes it, and the coordinator removes the worktree.

## 13. Machines

| Name in public text | Device classes | Backends it can prove |
|---|---|---|
| Machine A | `rtx4080super`, `arl-npu` | `cuda`, `npu`, `cpu` |
| Machine B | `b70` | `levelzero`, `cpu` |
| Machine C | Apple silicon | `metal`, `cpu` |
| Hailo boards (named when they join) | `pi5-hailo10h` | `hailo` |

The issue's "Machine" field decides where the work runs. If the work needs
a device on a machine that is offline, the agent marks the issue
`status:needs-hardware-proof` or `status:blocked` and says which device it
is waiting for. It never substitutes another device for the proof.

## 14. The OpenNLP port and the analysis crates

TurboEmbed is rewriting what Apache OpenNLP 3.x does in Rust, as
annotators over the layered document in `analysis/`. OpenNLP is the
reference; the Rust side follows it on its own schedule and aims to be
faster and more capable, not a line-by-line copy. Read
[`docs/analysis.md`](docs/analysis.md) before writing an annotator.

- Build on the crates that are there: `turbo-analysis` (`analysis/core/`),
  `turbo-analysis-layers` (`analysis/layers/`) for layers that more than
  one add-on reads, and a crate per add-on with a small schema crate
  beside it for its own layers. Do not add a second document or layer
  model.
- Shared layers use the standard `opennlp:` ids (`opennlp:sentences`,
  `opennlp:tokens`, `opennlp:pos` and so on) and TurboEmbed's own use
  `turbo:` ids. Retrieval chunks are `turbo:chunks`, because
  `opennlp:chunks` means phrase chunks. One id names one layer type.
- Offsets are UTF-8 byte offsets into one view. UTF-16 and code point
  offsets exist only at the Java and wire edge, through a checkpoint
  table per view (`docs/decisions.md`).
- An annotator that ports an OpenNLP component proves it matches
  OpenNLP: golden files produced by the OpenNLP 3.x JVM on the same input
  and model, checked in with the command that made them, and a test that
  compares against them. Where the Rust side is meant to differ (better
  boundaries, a bug OpenNLP has), the pull request says where and why,
  with the evaluation numbers.
- Read OpenNLP 3.x `.bin` models directly. Do not call into the JVM or a
  native OpenNLP library at run time; the JVM is only for producing
  golden files and evaluation baselines.
- Ignore OpenNLP 2.x entirely.
- Chunking is never greedy token-budget packing. A chunker reads the
  layers below it (sentences, structure, boundaries) and must keep exact
  offsets back to the source text.
- Training and golden-file generation run in pinned containers, driven
  from Rust, like the bundle recipes. No Python in the tree.
- An issue for a ported component names the OpenNLP class or package it
  ports, the layers it reads and writes, its stage, and the golden files
  and evaluation set that prove it. Use the "OpenNLP port" issue
  template.
