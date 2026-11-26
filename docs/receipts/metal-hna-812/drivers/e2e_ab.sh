#!/bin/zsh
# #812 end-to-end A/B: base (origin/main) and head turbo-bench at FASTEST, each with MLX float16,
# interleaved per size, the side order alternating by round, size order reversed in even rounds.
S=<scratch>; O=$S/e2e; mkdir -p $O
BASE=<base target>/release/turbo-bench; HEAD=<head target>/release/turbo-bench
gpu() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2 }
la() { sysctl -n vm.loadavg | tr -d '{}' | xargs }
sizes=(${=SIZES:-32x256 32x128 16x128 8x128 4x128 2x128 1x128 1x32})
for r in $(seq 1 ${ROUNDS:-5}); do
  list=($sizes); (( r % 2 == 0 )) && list=(${(Oa)sizes})
  sides=(base head); (( r % 2 == 0 )) && sides=(head base)
  for sz in $list; do b=${sz%x*}; s=${sz#*x}
    for side in $sides; do bin=$BASE; [[ $side == head ]] && bin=$HEAD; d=$O/out/r$r/$side/$sz; mkdir -p $d
      { echo "round $r size $sz side $side"; echo "before: load $(la) gpu $(gpu)% at $(date -u +%FT%TZ)"
        $bin record --bundle <bundles>/all-minilm-l6-v2 --device metal --precision fastest --batch $b --seq $s --rows dense --warmup 5 --iterations 20 --no-tei --mlx-model <bundles>/upstream-minilm --out $d 2>&1 | sed "s|$HOME|~|g"
        echo "exit: ${pipestatus[1]}"; echo "after: load $(la) gpu $(gpu)% at $(date -u +%FT%TZ)"; } > $O/r$r-$sz-$side.log
    done
  done; echo "round $r done $(la)"
done; echo DONE
