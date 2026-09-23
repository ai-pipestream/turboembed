#!/usr/bin/env bash
# Build the Turbo Java binding into the local Maven repository, build this
# demo, and run it. Needs JDK 25 on JAVA_HOME (or on PATH) and Maven, plus
# libturbo (cargo build -p turbo-shared from the repository root).
#
#   demo/java/run.sh [--provider-lib <so>] [--provider <id> --ordinal <n>] --bundle <dir> text...
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
lib="${TURBO_LIBRARY:-$root/target/debug/libturbo.so}"
[[ -f "$lib" ]] || { echo "missing $lib; run: cargo build -p turbo-shared" >&2; exit 2; }
mvn -q -B -f "$root/bindings/java/pom.xml" install -DskipTests
mvn -q -B -f "$root/demo/java/pom.xml" package -DskipTests
exec "${JAVA_HOME:+$JAVA_HOME/bin/}java" --enable-native-access=ALL-UNNAMED -Dturbo.library="$lib" \
    -jar "$root/demo/java/target/turbo-demo-java-0.1.0.jar" "$@"
