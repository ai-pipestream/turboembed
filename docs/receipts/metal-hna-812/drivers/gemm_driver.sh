#!/bin/zsh
# #812 GEMM receipt: base auto (main's kernels and harness), head auto and tiles, MLX float16,
# one shape a process, the side order rotating by shape and round.
G=<scratch>/gemm; B=<base tree>; H=<head tree>
la() { sysctl -n vm.loadavg | tr -d '{}' | xargs }
i=0
for r in 1 2 3; do for h in 384 768; do for m in 4096 16384; do for s in qkv attn_out ffn_in ffn_out; do
  sides=(base head mlx); k=$(( i % 3 )); sides=(${sides[@]:$k} ${sides[@]:0:$k}); i=$((i+1))
  for side in $sides; do
    echo "{\"round\":$r,\"hidden\":$h,\"m\":$m,\"shape\":\"$s\",\"side\":\"$side\",\"load\":\"$(la)\"}" >> $G/schedule.jsonl
    case $side in
      base) GEMM_ONLY=$h,$m,$s,bare GEMM_TILES=auto $G/gemm_time_base $B/core/metal/kernels.metal | tail -n +2 >> $G/base.jsonl ;;
      head) GEMM_ONLY=$h,$m,$s,bare GEMM_TILES=auto,h32n32a,h64n32a $G/gemm_time_head $H/core/metal/kernels.metal | tail -n +2 >> $G/head.jsonl ;;
      mlx) GEMM_ONLY=$h,$m,$s python3 $H/bench/metal-peers/mlx_gemm.py | tail -n +2 >> $G/mlx.jsonl ;;
    esac
  done
done; done; done; done
GEMM_TILES=auto $G/gemm_time_base $B/core/metal/kernels.metal 384 | head -1 > $G/base-header.json
GEMM_ONLY=384,128,qkv,bare GEMM_TILES=auto $G/gemm_time_head $H/core/metal/kernels.metal | head -1 > $G/head-header.json
GEMM_ONLY=384,128,qkv python3 $H/bench/metal-peers/mlx_gemm.py | head -1 > $G/mlx-header.json
echo DONE
