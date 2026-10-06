#!/usr/bin/env bash
# NVIDIA GPUs through CUDA, desktop or Jetson (docs/setup/cuda.md).
#
#   cuda.sh [--check|--install|--dry-run] [--cublas] [--toolkit 12.8]
#
# --install adds NVIDIA's apt repository and the toolkit packages the build
# needs (nvcc, the runtime's headers and static library, CCCL) on Ubuntu
# x86_64, as the release workflow does. It never installs or changes the
# GPU driver: that needs a reboot and is the machine owner's call; the
# check says what is missing.
set -u
. "$(dirname "$0")/lib.sh"
parse_mode "$@"
cublas=0
toolkit=12.8
set -- "${TURBO_SETUP_ARGS[@]+"${TURBO_SETUP_ARGS[@]}"}"
while [ $# -gt 0 ]; do
    case "$1" in
        --cublas) cublas=1 ;;
        --toolkit) toolkit=${2:?--toolkit needs a version such as 12.8}; shift ;;
        -h|--help) usage_common "[--cublas] [--toolkit 12.8]"; exit 0 ;;
        *) usage_common "[--cublas] [--toolkit 12.8]"; exit 2 ;;
    esac
    shift
done

machine=$(uname -m)
jetson=0
[ -r /etc/nv_tegra_release ] && jetson=1
echo "TurboEmbed setup: cuda ($(os_id) $(os_version), $machine$( [ $jetson = 1 ] && echo ', Jetson'))"
if [ "$(uname -s)" != Linux ]; then
    missing "the cuda backend builds and runs on Linux only"
    finish; exit
fi

check_rust
check_c_compiler

# The toolkit directory, in the order core/build.rs looks.
root=""
for v in TURBO_CUDA_ROOT CUDA_PATH CUDA_HOME; do
    eval "val=\${$v:-}"
    if [ -n "$val" ]; then root=$val; src=$v; break; fi
done
if [ -z "$root" ]; then root=/usr/local/cuda; src="the default"; fi

section "CUDA toolkit (build time), from $src: $root"
toolkit_major=""
if [ -x "$root/bin/nvcc" ]; then
    nv=$("$root/bin/nvcc" --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
    toolkit_major=${nv%%.*}
    case "$toolkit_major" in
        12|13) ok "nvcc $nv" ;;
        *) missing "nvcc $nv: the build wants CUDA 12.x or 13.x" ;;
    esac
else
    missing "$root/bin/nvcc (set TURBO_CUDA_ROOT to the toolkit's directory; a distribution toolkit with nvcc in /usr/bin is TURBO_CUDA_ROOT=/usr)"
    if [ $jetson = 1 ]; then
        note "fix (Jetson): sudo apt-get install nvidia-jetpack   # JetPack's CUDA toolkit, from NVIDIA's L4T repository"
    else
        case "$(os_id):$machine" in
            ubuntu:x86_64)
                rel=$(os_version | tr -d .)
                pkg=$(echo "$toolkit" | tr . -)
                keyring=0
                dpkg -s cuda-keyring >/dev/null 2>&1 ||
                    install_step "cd /tmp && curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$rel/x86_64/cuda-keyring_1.1-1_all.deb && $(as_root dpkg -i cuda-keyring_1.1-1_all.deb) && $(as_root apt-get update)" ||
                    keyring=1
                [ $keyring = 1 ] || apt_install "cuda-nvcc-$pkg cuda-cudart-dev-$pkg cuda-cccl-$pkg$( [ $cublas = 1 ] && echo " libcublas-dev-$pkg")"
                note "then: export TURBO_CUDA_ROOT=/usr/local/cuda-$toolkit"
                ;;
            *) note "install the CUDA toolkit 12.x or 13.x from NVIDIA (developer.nvidia.com/cuda-downloads) and set TURBO_CUDA_ROOT" ;;
        esac
    fi
fi

libdir=""
for d in lib64 lib "targets/$machine-linux/lib" "lib/$machine-linux-gnu"; do
    if [ -e "$root/$d/libcudart_static.a" ] && { [ $cublas = 0 ] || [ -e "$root/$d/libcublas.so" ]; }; then libdir=$root/$d; break; fi
done
if [ -n "$libdir" ]; then
    ok "libcudart_static.a$( [ $cublas = 1 ] && echo ' and libcublas.so') in $libdir"
elif [ -x "$root/bin/nvcc" ]; then
    missing "libcudart_static.a$( [ $cublas = 1 ] && echo ' and libcublas.so') under $root/lib64, lib, targets/$machine-linux/lib or lib/$machine-linux-gnu"
fi

arch_list=${TURBO_CUDA_ARCH:-89}
note "TURBO_CUDA_ARCH=$arch_list$( [ -z "${TURBO_CUDA_ARCH:-}" ] && echo ' (the default)')"

section "NVIDIA driver and GPU (run time)"
driver=""
if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    ok "driver $driver"
    nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader | while IFS=, read -r i name cap; do
        cap=$(echo "$cap" | tr -d ' .')
        name=$(echo "$name" | sed 's/^ *//')
        case ",$(echo "$arch_list" | tr -d ' ' | sed 's/sm_//g')," in
            *",$cap,"*) printf '  ok       GPU %s: %s, sm_%s, in TURBO_CUDA_ARCH\n' "$i" "$name" "$cap" ;;
            *) printf '  warning  GPU %s: %s, sm_%s, not in TURBO_CUDA_ARCH=%s: build with TURBO_CUDA_ARCH=%s\n' "$i" "$name" "$cap" "$arch_list" "$cap" ;;
        esac
    done
    major=${driver%%.*}
    if [ "$toolkit_major" = 12 ] && [ "$major" -lt 525 ] 2>/dev/null; then
        missing "driver $driver is older than 525, the first for CUDA 12.x"
    elif [ "$toolkit_major" = 13 ] && [ "$major" -lt 580 ] 2>/dev/null; then
        missing "driver $driver is older than 580, the first for CUDA 13.x"
    fi
elif [ $jetson = 1 ]; then
    ok "Jetson: the driver comes with L4T ($(head -n1 /etc/nv_tegra_release | sed 's/^# //'))"
    note "Orin is sm_87: build with TURBO_CUDA_ARCH=87"
    case ",$arch_list," in *",87,"*) ;; *) warn "TURBO_CUDA_ARCH=$arch_list has no 87" ;; esac
else
    missing "the NVIDIA driver (nvidia-smi does not run): install the driver your distribution packages for the GPU, then reboot; the library lists no CUDA device without it"
fi
if [ $jetson = 0 ] && ! ls /dev/nvidia0 >/dev/null 2>&1; then
    warn "/dev/nvidia0 is not there: the driver's kernel module is not loaded"
fi

finish
