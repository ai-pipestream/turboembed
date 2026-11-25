#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/maven-ci.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/bin"
cat > "$WORK/bin/mvn" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_LOG"
count="$(cat "$FAKE_COUNT")"
count=$((count + 1))
printf '%s\n' "$count" > "$FAKE_COUNT"
if ((count <= FAKE_FAILURES)); then
  echo "${FAKE_ERROR:-Could not find artifact org.example:demo:jar:1 in central}" >&2
  exit 1
fi
FAKE
cat > "$WORK/bin/sleep" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_SLEEP_LOG"
FAKE
chmod +x "$WORK/bin/mvn" "$WORK/bin/sleep"
export PATH="$WORK/bin:$PATH"
export FAKE_LOG="$WORK/mvn.log" FAKE_COUNT="$WORK/count" FAKE_SLEEP_LOG="$WORK/sleep.log"
# Retries clear recorded misses; keep them out of the real ~/.m2.
export MAVEN_REPO_LOCAL="$WORK/repo-default"
printf '0\n' > "$FAKE_COUNT"
: > "$FAKE_LOG"
: > "$FAKE_SLEEP_LOG"

export MAVEN_RESOLVE_ATTEMPTS=3 FAKE_FAILURES=2
"$SCRIPT" resolve -f clients/pom.xml -pl java-jni -am
[[ "$(cat "$FAKE_COUNT")" == 3 ]]
DEP="org.apache.maven.plugins:maven-dependency-plugin:3.9.0"
[[ "$(sed -n '1p' "$FAKE_LOG")" == "-B -ntp -DskipTests -f clients/pom.xml -pl java-jni -am $DEP:go-offline $DEP:get -Dartifact=org.apache.maven.surefire:surefire-junit-platform:3.5.4 package" ]]
[[ "$(sed -n '$p' "$FAKE_LOG")" == "$(sed -n '1p' "$FAKE_LOG")" ]]
[[ "$(cat "$FAKE_SLEEP_LOG")" == $'5\n10' ]]
echo "ok   a transient miss retries only resolution"

# No attempt forces update checks, and no goal is a bare prefix: either
# makes a warm run read metadata from Central.
if grep -Eq -- '(^| )-U( |$)|updateCheckManager|(^| )dependency:' "$FAKE_LOG"; then
  echo "FAIL resolution forces update checks or names the plugin by prefix: $(cat "$FAKE_LOG")" >&2
  exit 1
fi
[[ "$("$SCRIPT" dependency-plugin)" == "$DEP" ]]
echo "ok   resolution names the dependency plugin by its pinned coordinates"

# Before a retry, the failure Maven recorded is removed, so the retry asks
# for what failed again instead of reusing the recorded miss.
mkdir -p "$WORK/retry-repo/org/example"
cat > "$WORK/bin/mvn" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
count="$(cat "$FAKE_COUNT")"
count=$((count + 1))
printf '%s\n' "$count" > "$FAKE_COUNT"
if ((count == 1)); then
  printf 'x\n' > "$MAVEN_REPO_LOCAL/org/example/demo.jar.lastUpdated"
  printf 'maven-metadata-central.xml.error=HTTP 429\n' > "$MAVEN_REPO_LOCAL/org/example/resolver-status.properties"
  echo "Could not transfer metadata org.example:demo from central" >&2
  exit 1
fi
if [[ -e "$MAVEN_REPO_LOCAL/org/example/demo.jar.lastUpdated" || -e "$MAVEN_REPO_LOCAL/org/example/resolver-status.properties" ]]; then
  echo "retry saw the recorded miss" >&2
  exit 1
fi
FAKE
printf '0\n' > "$FAKE_COUNT"
: > "$FAKE_SLEEP_LOG"
MAVEN_REPO_LOCAL="$WORK/retry-repo" MAVEN_RESOLVE_ATTEMPTS=2 "$SCRIPT" resolve -f clients/pom.xml
[[ "$(cat "$FAKE_COUNT")" == 2 ]]
echo "ok   a retry does not reuse the miss Maven recorded"

cat > "$WORK/bin/mvn" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_LOG"
count="$(cat "$FAKE_COUNT")"
count=$((count + 1))
printf '%s\n' "$count" > "$FAKE_COUNT"
if ((count <= FAKE_FAILURES)); then
  echo "${FAKE_ERROR:-Could not find artifact org.example:demo:jar:1 in central}" >&2
  exit 1
fi
FAKE

printf '0\n' > "$FAKE_COUNT"
: > "$FAKE_SLEEP_LOG"
export FAKE_FAILURES=3
if "$SCRIPT" resolve -f clients/pom.xml >"$WORK/out" 2>&1; then
  echo "FAIL persistent resolution failure returned success" >&2
  exit 1
fi
[[ "$(cat "$FAKE_COUNT")" == 3 ]]
[[ "$(sed -n '$p' "$WORK/out")" == "error: Maven dependency resolution failed after 3 attempts" ]]
echo "ok   a persistent miss is bounded and fails the job"

printf '0\n' > "$FAKE_COUNT"
export FAKE_FAILURES=1 FAKE_ERROR="Fatal error compiling: release version 17 not supported"
if "$SCRIPT" resolve -f clients/pom.xml >"$WORK/out" 2>&1; then
  echo "FAIL compiler failure returned success" >&2
  exit 1
fi
[[ "$(cat "$FAKE_COUNT")" == 1 ]]
[[ "$(sed -n '$p' "$WORK/out")" == "$FAKE_ERROR" ]]
unset FAKE_ERROR
echo "ok   a compiler failure is returned without a retry"

printf '0\n' > "$FAKE_COUNT"
export MAVEN_RESOLVE_ATTEMPTS=0
if "$SCRIPT" resolve >"$WORK/out" 2>&1; then
  echo "FAIL invalid retry bound returned success" >&2
  exit 1
fi
[[ "$(cat "$FAKE_COUNT")" == 0 ]]
echo "ok   an invalid retry bound cannot hide a resolution failure"

mkdir -p "$WORK/repository/org/example"
printf 'marker\n' > "$WORK/repository/org/example/example.lastUpdated"
printf 'jar\n' > "$WORK/repository/org/example/example.jar"
printf 'maven-metadata-central.xml.error=HTTP 429\n' > "$WORK/repository/org/example/resolver-status.properties"
printf '<metadata/>\n' > "$WORK/repository/org/example/maven-metadata-central.xml"
MAVEN_REPO_LOCAL="$WORK/repository" "$SCRIPT" clean-cache
[[ ! -e "$WORK/repository/org/example/example.lastUpdated" ]]
[[ ! -e "$WORK/repository/org/example/resolver-status.properties" ]]
[[ "$(cat "$WORK/repository/org/example/example.jar")" == jar ]]
[[ "$(cat "$WORK/repository/org/example/maven-metadata-central.xml")" == "<metadata/>" ]]
MAVEN_REPO_LOCAL="$WORK/repository/missing" "$SCRIPT" clean-cache
if "$SCRIPT" clean-cache unexpected >"$WORK/out" 2>&1; then
  echo "FAIL cache cleanup accepted an argument" >&2
  exit 1
fi
echo "ok   cache cleanup removes only transient resolution markers"
