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
printf '0\n' > "$FAKE_COUNT"
: > "$FAKE_LOG"
: > "$FAKE_SLEEP_LOG"

export MAVEN_RESOLVE_ATTEMPTS=3 FAKE_FAILURES=2
"$SCRIPT" resolve -f clients/pom.xml -pl java-jni -am
[[ "$(cat "$FAKE_COUNT")" == 3 ]]
[[ "$(sed -n '1p' "$FAKE_LOG")" == *"-B -ntp -U -Daether.updateCheckManager.sessionState=bypass -DskipTests -f clients/pom.xml -pl java-jni -am dependency:go-offline dependency:get -Dartifact=org.apache.maven.surefire:surefire-junit-platform:3.5.4 package"* ]]
[[ "$(sed -n '$p' "$FAKE_LOG")" == *"dependency:go-offline dependency:get -Dartifact=org.apache.maven.surefire:surefire-junit-platform:3.5.4 package" ]]
[[ "$(cat "$FAKE_SLEEP_LOG")" == $'5\n10' ]]
echo "ok   a transient miss retries only resolution with fresh update checks"

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
MAVEN_REPO_LOCAL="$WORK/repository" "$SCRIPT" clean-cache
[[ ! -e "$WORK/repository/org/example/example.lastUpdated" ]]
[[ "$(cat "$WORK/repository/org/example/example.jar")" == jar ]]
MAVEN_REPO_LOCAL="$WORK/repository/missing" "$SCRIPT" clean-cache
if "$SCRIPT" clean-cache unexpected >"$WORK/out" 2>&1; then
  echo "FAIL cache cleanup accepted an argument" >&2
  exit 1
fi
echo "ok   cache cleanup removes only transient resolution markers"
