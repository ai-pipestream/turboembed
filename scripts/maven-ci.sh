#!/usr/bin/env bash
# Resolves and compiles a Maven reactor through package with tests skipped,
# then keeps transient resolution markers out of the saved Maven cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# maven-dependency-plugin by full coordinates. A bare `dependency:` prefix
# makes Maven read plugin-group metadata from Central to find the plugin
# and its latest version, on every run and with a warm cache too. Pinned,
# a warm run reads nothing from Central. 3.9.0 is the version
# analysis/goldens-tool/java/pom.xml pins.
DEPENDENCY_PLUGIN="org.apache.maven.plugins:maven-dependency-plugin:3.9.0"

# Output that means Maven could not get an artifact or plugin, as opposed
# to a compile or plugin execution failure. "in offline mode" is how an
# offline build reports what the local repository lacks.
RESOLUTION_FAILURE='Could not (resolve|find|transfer)|in offline mode|Failed to read artifact descriptor|Unable to resolve artifact|Non-resolvable|No plugin found for prefix|Plugin .*could not be resolved'

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
  local goals=("$DEPENDENCY_PLUGIN:go-offline" "$DEPENDENCY_PLUGIN:get"
    -Dartifact="org.apache.maven.surefire:surefire-junit-platform:$surefire_version" package)
  local log attempt=1
  log="$(mktemp)"
  # A restored cache usually holds the whole reactor. Offline, Maven makes
  # no request to Central at all, not even the remote repository filter's
  # .meta/prefixes.txt that an online build reads on every run. Central is
  # asked only for what the local repository lacks.
  local offline=(0 0)
  mvn -B -ntp -o -DskipTests "$@" "${goals[@]}" 2>&1 | tee "$log" || offline=("${PIPESTATUS[@]}")
  if ((offline[0] == 0 && offline[1] == 0)); then
    rm -f "$log"
    return 0
  fi
  if ((offline[0] == 0)) || ! grep -Eq "$RESOLUTION_FAILURE" "$log"; then
    rm -f "$log"
    return 1
  fi
  echo "The local Maven repository is incomplete; resolving from the remote repositories" >&2
  while true; do
    local statuses=(0 0)
    mvn -B -ntp -DskipTests "$@" "${goals[@]}" 2>&1 | tee "$log" || statuses=("${PIPESTATUS[@]}")
    if ((statuses[0] == 0 && statuses[1] == 0)); then
      rm -f "$log"
      return 0
    fi
    # A compile or plugin execution failure is not a transient repository
    # miss. Return it immediately instead of repeating the build.
    if ((statuses[0] == 0)) || ! grep -Eq "$RESOLUTION_FAILURE" "$log"; then
      rm -f "$log"
      return 1
    fi
    if ((attempt == attempts)); then
      echo "error: Maven dependency resolution failed after $attempts attempts" >&2
      rm -f "$log"
      return 1
    fi
    # The miss is recorded in the local repository, and Maven would not
    # ask for it again until its update interval passes. Removing the
    # record makes the next attempt ask again for what failed, and only
    # for that: nothing already resolved is checked again.
    clean_cache >/dev/null
    local delay=$((attempt * 5))
    echo "Maven dependency resolution failed; retrying in $delay seconds (attempt $((attempt + 1))/$attempts)" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
  done
}

# Removes the records of failed transfers: `*.lastUpdated` for artifacts,
# and `resolver-status.properties`, where Maven 3.9 records failed
# metadata reads. Both only say when a remote was last asked; deleting
# one never deletes an artifact or its metadata.
clean_cache() {
  local repository="${MAVEN_REPO_LOCAL:-$HOME/.m2/repository}"
  if [[ ! -d "$repository" ]]; then
    return 0
  fi
  find "$repository" -type f \( -name '*.lastUpdated' -o -name 'resolver-status.properties' \) -print -delete
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
  dependency-plugin)
    echo "$DEPENDENCY_PLUGIN"
    ;;
  *)
    echo "usage: $0 resolve [Maven project arguments...] | clean-cache | dependency-plugin" >&2
    exit 2
    ;;
esac
