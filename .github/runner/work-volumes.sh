#!/bin/sh
# work-volumes.sh BASE SIZE NAME... (run as root)
# Creates BASE/NAME.img (preallocated, ext4) per runner, mounts it at
# BASE/NAME through /etc/fstab, and drops the .turbo-ci-volume marker.
set -eu
base=$1 size=$2; shift 2
mkdir -p "$base"
stamp=$(date +%Y%m%d-%H%M%S)
cp -a /etc/fstab "/etc/fstab.bak-turbo-ci-$stamp"
for n in "$@"; do
  img=$base/$n.img mnt=$base/$n
  mkdir -p "$mnt"
  if ! mountpoint -q "$mnt"; then
    [ -f "$img" ] || { fallocate -l "$size" "$img"; mkfs.ext4 -q -F -m 0 -L "tci-$(echo $n | tail -c 12)" -E nodiscard,root_owner=1001:1001 "$img"; }
    grep -q " $mnt ext4 " /etc/fstab || printf '# turbo-ci runner %s work volume (%s cap), see /opt/turbo-ci/supervise.sh\n%s %s ext4 loop,nofail,noatime,x-systemd.requires-mounts-for=%s 0 0\n' "$n" "$size" "$img" "$mnt" "$base" >> /etc/fstab
    mount "$mnt"
  fi
  touch "$mnt/.turbo-ci-volume"
  mkdir -p "$mnt/_tmp"; chown 1001:1001 "$mnt" "$mnt/_tmp"; chmod 1777 "$mnt/_tmp"
  df -h "$mnt" | tail -1
done
