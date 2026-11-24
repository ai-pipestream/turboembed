# Self-hosted CI runner

ci.yml's Linux lanes run on the hosted `ubuntu-24.04` runner unless the
repository variable `TURBO_CI_LINUX` names runner labels as JSON:

    gh variable set TURBO_CI_LINUX --body '["self-hosted", "turbo-ci"]'

Deleting that variable sends its jobs back to the hosted runner. The Windows
lanes are hosted regardless of its value.

## Model training runs

The classical `models`, `models-ud`, `models-tatoeba`, `models-fineweb`
and `models-publish` workflows were retired by #1511. Their absence is
checked by `analysis/train-recipes/tests/retired_commands.rs`. The
`ci:models` label no longer starts training, and `TURBO_TRAIN_LINUX`
does not select runners for any current workflow.

The required check `check` comes from ci.yml (lint, the test shards,
parity and the other fast lanes), and runs on pull requests and merge
groups. It is not evidence of a full model-training or benchmark run.
Changes requiring that evidence must record the actual commands, data,
device and results separately; do not count the retired label or a
missing workflow as a model pass.

## What keeps outside code off these runners

The `runs-on` expressions send unchanged pull requests from forks to the
hosted runner, but a fork can edit them, so they are routing, not a
boundary. The boundary is the repository's Actions setting that requires
approval to run workflows from every outside collaborator (Settings,
Actions, General; the API's `approval_policy` is
`all_external_contributors`). With it, no fork's workflow runs anywhere
until a maintainer has read the change and approved the run; approve
only changes you would run on these hosts yourself.

Each runner also takes one job: supervise.sh asks GitHub for a
just-in-time configuration, runs the job in a new container and removes
it, so a job cannot leave anything for the next one.

## Starting runners

Build the image on each host:

    docker build -t turbo-ci-runner .github/runner

Install supervise.sh as /opt/turbo-ci/supervise.sh and
turbo-ci-runner@.service in /etc/systemd/system, set `User=` in the unit
to an account with docker access and a gh login that may create
runners, then give each runner a settings file and start it:

    # /etc/turbo-ci/1.env
    NAME=x86-64-1
    LABELS=turbo-ci,x86-64
    DOCKER_OPTS=--cpus 8 --memory 16g --memory-swap 16g -e CARGO_BUILD_JOBS=8

    systemctl enable --now turbo-ci-runner@1

A host with docker but no gh login is driven from one that has both, with
`DOCKER=ssh -T <host> docker` in the settings file; the image must be
loaded on that host.

## Disk

Give every runner a work volume of its own. Without one, a job's
checkout, `target/` and temporary files go to the container's writable
layer under /var/lib/docker, on the host's root filesystem, shared with
every other runner and everything else on the host. When that filesystem
fills, every running job fails at once (#2552).

work-volumes.sh, run as root, makes a preallocated ext4 image per runner
under BASE, adds a loop mount for it to /etc/fstab and mounts it:

    sudo sh .github/runner/work-volumes.sh /work/turbo-ci 150G x86-64-1 x86-64-2

Then name the mount in each runner's settings file:

    WORK_DIR=/work/turbo-ci/x86-64-1

supervise.sh mounts WORK_DIR as the runner's `_work` directory, points
TMPDIR into it, and empties it before and after every job. A job that
fills it fails alone, and the host's root filesystem is untouched. The
volume's root holds a `.turbo-ci-volume` marker; when it is missing, as
when the image is not mounted, the runner takes no job rather than
falling back to the root filesystem. Keep BASE on a local disk, never on
a network share: a Rust build writes many small files and takes file
locks. With `DOCKER=ssh ...`, BASE and the image are on that host; where
its firmware owns /etc/fstab, mount the image from a systemd unit
instead.

The images are preallocated, so each one's size is reserved on BASE's
filesystem whether or not a job is running.

Name runners by architecture, not by host name: runner names show in the
public job logs.

Runners can be added without root, as transient user units of the same
account (with `loginctl enable-linger` so they outlive the login):

    systemd-run --user --unit=turbo-ci-runner-extra-1 --property=Restart=always \
      /opt/turbo-ci/supervise.sh x86-64-extra-1 turbo-ci,x86-64 \
      --cpus 8 --memory 16g --memory-swap 16g -e CARGO_BUILD_JOBS=8

A transient unit is gone after a reboot. To keep one, make it a system
unit as in the steps above, which needs root: write
/etc/turbo-ci/extra-1.env with its NAME, LABELS and DOCKER_OPTS, run
`systemctl enable --now turbo-ci-runner@extra-1`, then, as the runner's
account, stop the transient one with
`systemctl --user stop turbo-ci-runner-extra-1`.

Each runner takes one job at a time. A host whose runners leave its
cores idle can take more runners; the `--cpus` and `--memory` of all its
runners may add up to more than the host has, since few jobs use all of
theirs at once.

## GPUs

On a host with a GPU, add the device to `DOCKER_OPTS` and a label for
it to `LABELS`. An NVIDIA GPU goes in through the NVIDIA Container
Toolkit's CDI spec, with the label `cuda`:

    --device nvidia.com/gpu=all

An Intel GPU goes in as its DRM nodes, with the ids of the groups that own
them on the host and the label `intel-gpu`:

    --device /dev/dri --group-add <render gid> --group-add <video gid>
