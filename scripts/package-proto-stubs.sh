#!/usr/bin/env bash
# Packages the versioned turbo.*.v1 proto archive, generated Java Maven
# artifact, and generated Python wheel with SHA-256 sums, and optionally runs
# minimal Java and Python clients compiled only against the packaged stubs
# against a live turbo-kserve endpoint (issue #1297).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-v0.1.0}"
OUT_DIR="${2:-$ROOT/target/proto-stubs-dist}"
SMOKE_ADDR="${3:-}"

CLEAN_TAG="$TAG"
if [[ ! "$CLEAN_TAG" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  echo "error: unsupported release tag '${TAG}'; expected vX.Y.Z or vX.Y.Z-<prerelease>" >&2
  exit 1
fi
VERSION="${CLEAN_TAG#v}"
CLEAN_TAG="v${VERSION}"

GRPC_JAVA_VERSION="1.68.2"
PROTOC_VERSION="3.25.5"
GRPCIO_PYTHON_VERSION="1.70.0"
PROTOBUF_PYTHON_VERSION="5.29.3"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# 1. Proto archive: turboembed-proto-<tag>.tar.gz + .sha256
PROTO_ARCHIVE_DIR="turboembed-proto-${CLEAN_TAG}"
mkdir -p "$WORK_DIR/$PROTO_ARCHIVE_DIR/proto" "$WORK_DIR/$PROTO_ARCHIVE_DIR/licenses"
cp "$ROOT"/server/proto/*.proto "$ROOT/server/proto/buf.yaml" "$WORK_DIR/$PROTO_ARCHIVE_DIR/proto/"
cp "$ROOT/LICENSE" "$WORK_DIR/$PROTO_ARCHIVE_DIR/licenses/LICENSE"
cp "$ROOT/NOTICE" "$WORK_DIR/$PROTO_ARCHIVE_DIR/licenses/NOTICE"
tar -C "$WORK_DIR" -czf "$OUT_DIR/${PROTO_ARCHIVE_DIR}.tar.gz" "$PROTO_ARCHIVE_DIR"
(cd "$OUT_DIR" && sha256sum "${PROTO_ARCHIVE_DIR}.tar.gz" > "${PROTO_ARCHIVE_DIR}.tar.gz.sha256")

# 2. Java Maven stub artifact: turboembed-grpc-<version>.jar, -sources.jar, .pom + .sha256
JAVA_BUILD_DIR="$WORK_DIR/java-build"
mkdir -p "$JAVA_BUILD_DIR"
sed "s|<version>0.1.0</version>|<version>${VERSION}</version>|" \
  "$ROOT/clients/java/pom.xml" > "$JAVA_BUILD_DIR/pom.xml"
mvn -B -q -f "$JAVA_BUILD_DIR/pom.xml" \
  -Dproto.source.dir="$ROOT/server/proto" \
  -Dlicense.source.dir="$ROOT" \
  -DskipTests \
  package dependency:build-classpath -DincludeScope=test -Dmdep.outputFile="$WORK_DIR/java-runtime.cp"

JAR_NAME="turboembed-grpc-${VERSION}.jar"
SOURCES_JAR_NAME="turboembed-grpc-${VERSION}-sources.jar"
POM_NAME="turboembed-grpc-${VERSION}.pom"
cp "$JAVA_BUILD_DIR/target/${JAR_NAME}" "$OUT_DIR/$JAR_NAME"
cp "$JAVA_BUILD_DIR/target/${SOURCES_JAR_NAME}" "$OUT_DIR/$SOURCES_JAR_NAME"
cp "$JAVA_BUILD_DIR/pom.xml" "$OUT_DIR/$POM_NAME"
(
  cd "$OUT_DIR"
  sha256sum "$JAR_NAME" > "${JAR_NAME}.sha256"
  sha256sum "$SOURCES_JAR_NAME" > "${SOURCES_JAR_NAME}.sha256"
  sha256sum "$POM_NAME" > "${POM_NAME}.sha256"
)

# 3. Python wheel stub artifact: turboembed_proto-<version>-py3-none-any.whl + .sha256
VENV_DIR="$WORK_DIR/py-venv"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --disable-pip-version-check \
  "grpcio==${GRPCIO_PYTHON_VERSION}" \
  "grpcio-tools==${GRPCIO_PYTHON_VERSION}" \
  "protobuf==${PROTOBUF_PYTHON_VERSION}"

WHEEL_PATH="$("$VENV_DIR/bin/python" "$ROOT/clients/python/build_wheel.py" \
  --proto-dir "$ROOT/server/proto" \
  --out-dir "$OUT_DIR" \
  --version "$VERSION")"
WHEEL_NAME="$(basename "$WHEEL_PATH")"
(cd "$OUT_DIR" && sha256sum "$WHEEL_NAME" > "${WHEEL_NAME}.sha256")

# 4. Write release BUILD metadata
{
  echo "turboembed proto and client stubs ${CLEAN_TAG}, commit ${GITHUB_SHA:-local}"
  echo "proto archive: ${PROTO_ARCHIVE_DIR}.tar.gz"
  echo "java maven artifact: ai.pipestream:turboembed-grpc:${VERSION} (protoc ${PROTOC_VERSION}, grpc-java ${GRPC_JAVA_VERSION})"
  echo "python wheel: ${WHEEL_NAME} (grpcio-tools ${GRPCIO_PYTHON_VERSION}, protobuf ${PROTOBUF_PYTHON_VERSION})"
} > "$OUT_DIR/proto-stubs.BUILD"

# 5. Optional end-to-end smoke check against a running turbo-kserve instance
if [[ -n "$SMOKE_ADDR" ]]; then
  # 5a. Compile and run the minimal Java client ONLY against the packaged stub JAR
  JAVA_SMOKE_CLASSES="$WORK_DIR/java-smoke-classes"
  mkdir -p "$JAVA_SMOKE_CLASSES"
  RUNTIME_CP="$(cat "$WORK_DIR/java-runtime.cp")"
  javac --release 17 \
    -cp "$OUT_DIR/$JAR_NAME:$RUNTIME_CP" \
    -d "$JAVA_SMOKE_CLASSES" \
    "$ROOT/clients/java/src/test/java/ai/pipestream/turboembed/ClientSmoke.java"
  java -cp "$JAVA_SMOKE_CLASSES:$OUT_DIR/$JAR_NAME:$RUNTIME_CP" \
    ai.pipestream.turboembed.ClientSmoke "$SMOKE_ADDR"

  # 5b. Install ONLY the packaged wheel in the clean venv and run the Python client outside the repo
  "$VENV_DIR/bin/pip" install --quiet --no-deps "$OUT_DIR/$WHEEL_NAME"
  (cd "$WORK_DIR" && "$VENV_DIR/bin/python" "$ROOT/clients/python/smoke_test.py" "$SMOKE_ADDR")
fi
