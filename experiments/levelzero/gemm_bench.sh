#!/bin/bash
# r2.sh TM TN WM WN KSTEP LARGE shape:splits...
TM=$1 TN=$2 WM=$3 WN=$4 KS=$5 L=$6; shift 6
o=k2_$TM_$TN_$KS_$$.spv
clang -cl-std=CL3.0 --target=spirv64 -O2 -mllvm --spirv-ext=+SPV_INTEL_subgroups -c -DTM=$TM -DTN=$TN -DWM=$WM -DWN=$WN -DKSTEP=$KS $XD -o $o g2.cl || exit 1
O="$XO"; [ "$L" = 1 ] && O="-ze-opt-large-register-file"
for s in "$@"; do sh=${s%:*}; sp=${s#*:}; OPTS="$O" ./g2 $o ${sh//x/ } $TM $TN $WM $WN $sp; done
rm -f $o
