# Self-hosted CI runner

ci.yml's Linux lanes and the training jobs of the models-* workflows run
on the hosted `ubuntu-24.04` runner unless the repository variable
`TURBO_CI_LINUX` names runner labels as JSON:

    gh variable set TURBO_CI_LINUX --body '["self-hosted", "turbo-ci"]'

Deleting the variable sends the jobs back to the hosted runner. The
Windows lanes and the publishing jobs are hosted whatever it says.

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

Name runners by architecture, not by host name: runner names show in the
public job logs.

## GPUs

On a host with a GPU, add the device to `DOCKER_OPTS` and a label for
it to `LABELS`. An NVIDIA GPU goes in through the NVIDIA Container
Toolkit's CDI spec, with the label `cuda`:

    --device nvidia.com/gpu=all

An Intel GPU goes in as its DRM nodes, with the ids of the groups that own
them on the host and the label `intel-gpu`:

    --device /dev/dri --group-add <render gid> --group-add <video gid>
