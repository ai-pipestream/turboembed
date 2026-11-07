#!/bin/zsh
# Five paired rounds, interleaved shape by shape, first side alternating by shape and round.
S=${0:a:h}; M=$S/main; O=$S/gemm-out
cd $M
gpu() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2 }
la() { sysctl -n vm.loadavg | tr -d '{}' | xargs }
src=$(cd <checkout> && git rev-parse origin/main)
sh() { shasum -a 256 $1 | cut -d' ' -f1 }
print -r -- "{\"source_commit\":\"$src\",\"kernels_sha256\":\"$(sh core/metal/kernels.metal)\",\"gemm_time_swift_sha256\":\"$(sh bench/metal-peers/gemm_time.swift)\",\"gemm_time_binary_sha256\":\"$(sh $S/gemm_time)\",\"mlx_gemm_py_sha256\":\"$(sh bench/metal-peers/mlx_gemm.py)\",\"macos\":\"$(sw_vers -productVersion) ($(sw_vers -buildVersion))\",\"device\":\"Apple M2\",\"turbo_command\":\"GEMM_ONLY=<h>,<m>,<shape>,bare ./gemm_time core/metal/kernels.metal\",\"mlx_command\":\"GEMM_ONLY=<h>,<m>,<shape> python3 bench/metal-peers/mlx_gemm.py\",\"build\":\"swiftc -O bench/metal-peers/gemm_time.swift -o gemm_time\"}" > $O/schedule.jsonl
: > $O/turbo.jsonl; : > $O/mlx.jsonl
run() { # round side shape
  local r=$1 side=$2 sr=$3 lb gb st
  lb=$(la); gb=$(gpu); st=$(date -u +%FT%TZ)
  if [[ $side == turbo ]]; then GEMM_ONLY=$sr,bare $S/gemm_time core/metal/kernels.metal | sed "s/^{/{\"round\":$r,/" >> $O/turbo.jsonl
  else GEMM_ONLY=$sr python3 bench/metal-peers/mlx_gemm.py | sed "s/^{/{\"round\":$r,/" >> $O/mlx.jsonl; fi
  local ex=${pipestatus[1]}
  print -r -- "{\"round\":$r,\"side\":\"$side\",\"shape_run\":\"$sr\",\"start\":\"$st\",\"end\":\"$(date -u +%FT%TZ)\",\"loadavg_before\":\"$lb\",\"gpu_util_pct_before\":$gb,\"loadavg_after\":\"$(la)\",\"gpu_util_pct_after\":$(gpu),\"exit\":$ex}" >> $O/schedule.jsonl
  local five=$(sysctl -n vm.loadavg | awk '{print $3}')
  if (( five > 3.5 )); then echo "ABORT: 5-min load $five" | tee $O/ABORTED; exit 3; fi
}
for r in 1 2 3 4 5; do i=0
  for h in 384 768; do for m in 128 512 1024 4096 16384; do for s in qkv attn_out ffn_in ffn_out; do
    if (( (r + i) % 2 == 1 )); then run $r mlx $h,$m,$s; run $r turbo $h,$m,$s; else run $r turbo $h,$m,$s; run $r mlx $h,$m,$s; fi
    i=$((i+1))
  done; done; done
  echo "round $r done $(la)"
done
echo DONE
