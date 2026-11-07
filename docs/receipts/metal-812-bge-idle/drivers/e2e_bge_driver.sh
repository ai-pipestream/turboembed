#!/bin/zsh
S=${0:a:h}; R=$S/mainrepo; O=$S/e2e-bge-out; mkdir -p $O; cd $R
gpu() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2 }
la() { sysctl -n vm.loadavg | tr -d '{}' | xargs }
sizes=(32x256 32x128 16x128 8x128 4x128 2x128 1x128 1x32 1x512 8x512 32x512)
for r in 1 2 3 4 5; do
  list=($sizes); (( r % 2 == 0 )) && list=(${(Oa)sizes})
  for sz in $list; do b=${sz%x*}; s=${sz#*x}; d=$O/r$r/$sz; mkdir -p $d
    cmd="target/release/turbo-bench record --bundle <bundles>/bge-base-en-v1.5-r1 --device metal --precision fastest --batch $b --seq $s --rows dense --warmup 5 --iterations 20 --no-tei --mlx-model <upstream bge-base-en-v1.5> --out <out>"
    { echo "round $r size $sz commit $(git rev-parse HEAD)"; echo "command: $cmd"; echo "before: load $(la) gpu $(gpu)% at $(date -u +%FT%TZ)"
      target/release/turbo-bench record --bundle ~/turbo-bundles/bge-base-en-v1.5-r1 --device metal --precision fastest --batch $b --seq $s --rows dense --warmup 5 --iterations 20 --no-tei --mlx-model $HOME/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a --out $d 2>&1 | sed "s|$HOME|~|g; s|$S|<scratch>|g"
      echo "exit: ${pipestatus[1]}"; echo "after: load $(la) gpu $(gpu)% at $(date -u +%FT%TZ)"; } > $O/r$r-$sz.log
    five=$(sysctl -n vm.loadavg | awk '{print $3}')
    if (( five > 3.5 )); then echo "ABORT: 5-min load $five" | tee $O/ABORTED; exit 3; fi
  done; echo "round $r done $(la)"
done; echo DONE
