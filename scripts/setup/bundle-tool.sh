#!/usr/bin/env bash
# The bundle tool, turbo-bundle, and the pinned Python it runs in a
# container (docs/setup/bundles.md).
#
#   bundle-tool.sh [--check|--install|--dry-run] [--hailo]
#
# Checks Docker, the reference image each recipe pins, network access to
# huggingface.co for `fetch`, and the tools to rejoin a prebuilt bundle.
# --install builds the reference image from bundle/reference and prints
# its id; it never edits a recipe. With --hailo it also checks for Hailo's
# Dataflow Compiler wheel, which needs a Hailo account: the script says
# where the file goes and builds the compiler image once it is there.
set -u
. "$(dirname "$0")/lib.sh"
parse_mode "$@"
hailo=0
set -- "${TURBO_SETUP_ARGS[@]+"${TURBO_SETUP_ARGS[@]}"}"
while [ $# -gt 0 ]; do
    case "$1" in
        --hailo) hailo=1 ;;
        -h|--help) usage_common "[--hailo]"; exit 0 ;;
        *) usage_common "[--hailo]"; exit 2 ;;
    esac
    shift
done

echo "TurboEmbed setup: bundle tool ($(os_id) $(os_version), $(uname -m))"
check_rust

section "Docker (the reference and the conversions run in pinned containers)"
docker_ok=0
if have docker; then
    if docker info >/dev/null 2>&1; then
        ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null), and $(id -un) can reach the daemon"
        docker_ok=1
    else
        missing "docker is installed, but $(id -un) cannot reach the daemon (start it, or add the user to the docker group)"
    fi
else
    missing "docker (turbo-bundle calls the docker command; podman works only through a docker shim)"
    case "$(os_id)" in ubuntu|debian|raspbian) apt_install docker.io ;; macos) note "fix: install Docker Desktop" ;; esac
fi
for t in id tar; do if have $t; then ok "$t"; else missing "$t"; fi; done
if [ "$(uname -m)" != x86_64 ] && [ "$(uname -m)" != amd64 ]; then
    warn "$(uname -m): the reference image installs x86_64 and arm64 wheels, but the Hailo compiler is x86_64 only"
fi

section "Reference images the recipes pin"
pins=$(grep -ho '"turbo-reference@sha256:[0-9a-f]*"' "$TURBO_ROOT"/bundle/recipes/*.json | sort -u | tr -d '"')
any=0
for p in $pins; do
    used=$(grep -l "$p" "$TURBO_ROOT"/bundle/recipes/*.json | xargs -n1 basename | tr '\n' ' ')
    if [ $docker_ok = 1 ] && docker image inspect "${p#*@}" >/dev/null 2>&1; then
        ok "${p#*@} present ($used)"; any=1
    else
        warn "${p#*@} not present: $used"
    fi
done
if [ $any = 0 ]; then
    note "The pins name images built on another machine; a local build gets its own id."
    note "Build the image, then put turbo-reference@<id> in each recipe you make, in"
    note "manifest.reference.produced_by.container (bundle/README.md)."
    if [ $docker_ok = 1 ]; then
        install_step "docker build -t turbo-reference '$TURBO_ROOT/bundle/reference' && docker image inspect --format 'pin: turbo-reference@{{.Id}}' turbo-reference"
    else
        note "fix: docker build -t turbo-reference bundle/reference"
    fi
fi
if [ $docker_ok = 1 ] && docker image inspect turbo-reference >/dev/null 2>&1; then
    ok "a local turbo-reference image: pin turbo-reference@$(docker image inspect --format '{{.Id}}' turbo-reference)"
fi

section "Network (fetch reads the model files from huggingface.co)"
if have curl; then
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 15 https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2 || true)
    case "$code" in
        200) ok "huggingface.co answers" ;;
        *) warn "huggingface.co answered ${code:-nothing}: fetch will fail here; fill <upstream-dir> another way, then run 'turbo-bundle reference' (the offline path)" ;;
    esac
else
    warn "curl is not installed, so the network was not checked"
fi

section "Prebuilt bundles (split zip archives on the GitHub releases page)"
for t in zip unzip sha256sum; do
    if have $t; then ok "$t"
    elif [ $t = sha256sum ] && have shasum; then ok "shasum (use: shasum -a 256)"
    else
        missing "$t"
        case "$(os_id)" in ubuntu|debian|raspbian) [ $t = sha256sum ] || apt_install $t ;; esac
    fi
done

if [ $hailo = 1 ]; then
    section "Hailo Dataflow Compiler (for a HEF; needs a Hailo account)"
    wheels=$(ls "$TURBO_ROOT"/bundle/hailo/*.whl 2>/dev/null || true)
    if [ -z "$wheels" ]; then
        missing "no Dataflow Compiler wheel in bundle/hailo/"
        note "Download it from Hailo's Developer Zone (Software Downloads, Dataflow Compiler,"
        note "Linux x86_64) and put the .whl file in $TURBO_ROOT/bundle/hailo/ (git ignores it)."
        note "A Hailo-10H takes compiler 5.x; a Hailo-8 or Hailo-8L takes 3.x (docs/setup/hailo-8.md)."
    else
        for w in $wheels; do
            v=$(basename "$w" | sed -E 's/^hailo_dataflow_compiler-([0-9.]+)-.*/\1/')
            case "$v" in
                5.*) ok "$(basename "$w"): compiler $v, for the Hailo-10H" ;;
                3.*) ok "$(basename "$w"): compiler $v, for the Hailo-8 and Hailo-8L" ;;
                *) warn "$(basename "$w"): version not recognised" ;;
            esac
        done
        if [ $docker_ok = 1 ]; then
            if docker image inspect turbo-hailo-dfc >/dev/null 2>&1; then
                ok "a local turbo-hailo-dfc image: pin turbo-hailo-dfc@$(docker image inspect --format '{{.Id}}' turbo-hailo-dfc)"
            else
                w=$(basename "$(echo "$wheels" | head -n1)")
                missing "no local turbo-hailo-dfc image"
                install_step "docker build --build-arg WHEEL=$w -t turbo-hailo-dfc '$TURBO_ROOT/bundle/hailo' && docker image inspect --format 'pin: turbo-hailo-dfc@{{.Id}}' turbo-hailo-dfc"
            fi
        fi
    fi
fi

finish
