#!/usr/bin/env bash
# Resolves and compiles a Maven reactor through package with tests skipped,
# then keeps transient resolution markers out of the setup-java cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve() {
  local attempts="${MAVEN_RESOLVE_ATTEMPTS:-3}"
  if [[ ! "$attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: MAVEN_RESOLVE_ATTEMPTS must be a positive integer" >&2
    exit 2
  fi
  local surefire_version
  surefire_version="$(sed -n '/<artifactId>maven-surefire-plugin<\/artifactId>/{n;s|.*<version>\(.*\)</version>.*|\1|p;}' \
    "$ROOT/clients/java-api/pom.xml")"
  if [[ ! "$surefire_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "error: could not read the pinned Surefire version" >&2
    exit 2
  fi
  local log attempt=1
  log="$(mktemp)"
  while true; do
    local statuses=(0 0)
    mvn -B -ntp -U -Daether.updateCheckManager.sessionState=bypass -DskipTests \
      "$@" dependency:go-offline dependency:get \
      -Dartifact="org.apache.maven.surefire:surefire-junit-platform:$surefire_version" \
      package 2>&1 | tee "$log" || statuses=("${PIPESTATUS[@]}")
    if ((statuses[0] == 0 && statuses[1] == 0)); then
      rm -f "$log"
      return 0
    fi
    # A compile or plugin execution failure is not a transient repository
    # miss. Return it immediately instead of repeating the build.
    if ((statuses[0] == 0)) || ! grep -Eq \
      'Could not (resolve|find|transfer)|Failed to read artifact descriptor|Unable to resolve artifact|Non-resolvable|No plugin found for prefix|Plugin .*could not be resolved' \
      "$log"; then
      rm -f "$log"
      return 1
    fi
    if ((attempt == attempts)); then
      echo "error: Maven dependency resolution failed after $attempts attempts" >&2
      rm -f "$log"
      return 1
    fi
    local delay=$((attempt * 5))
    echo "Maven dependency resolution failed; retrying in $delay seconds (attempt $((attempt + 1))/$attempts)" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
  done
}

clean_cache() {
  local repository="${MAVEN_REPO_LOCAL:-$HOME/.m2/repository}"
  if [[ ! -d "$repository" ]]; then
    return 0
  fi
  find "$repository" -type f -name '*.lastUpdated' -print -delete
}

case "${1:-}" in
  resolve)
    shift
    resolve "$@"
    ;;
  clean-cache)
    shift
    if (($# != 0)); then
      echo "error: clean-cache takes no arguments" >&2
      exit 2
    fi
    clean_cache
    ;;
  *)
    echo "usage: $0 resolve [Maven project arguments...] | clean-cache" >&2
    exit 2
    ;;
esac
