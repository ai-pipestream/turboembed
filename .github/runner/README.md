# Self-hosted CI runner

ci.yml's Linux lanes and the training jobs of the models-* workflows run
on the hosted `ubuntu-24.04` runner unless the repository variable
`TURBO_CI_LINUX` names runner labels as JSON:

    gh variable set TURBO_CI_LINUX --body '["self-hosted", "turbo-ci"]'

Deleting the variable sends the jobs back to the hosted runner. Pull
requests from forks run hosted, and the Windows lanes and the publishing
jobs are hosted, whatever the variable says.

A runner is a container of this image, with a volume for /home/runner:

    docker build -t turbo-ci-runner .github/runner
    docker run -d --name turbo-ci-1 --restart unless-stopped \
      --cpus 8 --memory 16g --memory-swap 16g -e CARGO_BUILD_JOBS=8 \
      -e RUNNER_NAME=x86-64-1 -e RUNNER_LABELS=turbo-ci,x86-64 \
      -e RUNNER_TOKEN="$(gh api -X POST repos/ai-pipestream/turboembed/actions/runners/registration-token --jq .token)" \
      -v turbo-ci-1:/home/runner turbo-ci-runner

On a host with a GPU, add the device and a label for it. An NVIDIA GPU
goes in through the NVIDIA Container Toolkit's CDI spec, with the label
`cuda`:

    --device nvidia.com/gpu=all

An Intel GPU goes in as its DRM nodes, with the groups that own them on
the host and the label `intel-gpu`:

    --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
      --group-add "$(getent group video | cut -d: -f3)"

The token is read once, on the first start; later starts reuse the
registration kept in the volume. Name runners by architecture, not by
host name: runner names show in the public job logs.
