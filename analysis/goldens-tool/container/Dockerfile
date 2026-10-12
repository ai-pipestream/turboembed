# The container that makes the golden layer files: a pinned JDK, OpenNLP built
# from a pinned commit of the fork, and the generator in ../java.
#
# Build it with `turbo-goldens image` (analysis/goldens-tool/README.md), which
# passes ../java as the build context. The base image digest and the commit
# are also Rust constants in ../src/recipe.rs; a test checks they agree.
# Models are not in the image: `turbo-goldens generate` mounts them, with the
# inputs and the checked in fixtures, read only, and runs the container without
# a network.

FROM eclipse-temurin:21-jdk@sha256:3e3c176ffed168beb42c607be9bc1639b466cf00261a0fb04425562c9d0c5c2b

ARG OPENNLP_REPO=https://github.com/ai-pipestream/opennlp
ARG OPENNLP_COMMIT=cdf33c653d5d61d5af7d37ffee189a83bd8392ea

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Fetch exactly the pinned commit and refuse anything else. The commit read
# back from git is what the generator writes into every golden's header.
WORKDIR /opennlp
RUN git init -q . \
 && git remote add origin "$OPENNLP_REPO" \
 && git fetch -q --depth 1 origin "$OPENNLP_COMMIT" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "$OPENNLP_COMMIT" \
 && git rev-parse HEAD > /opennlp.commit

# The runtime and the modules it depends on, installed into the local Maven
# repository. The forbidden API check is skipped because it fails on classes
# of this commit that the generator does not use (NameSample and Parse in
# opennlp-api use java.util.regex); it only reads the compiled classes.
# -DskipITs is needed as well: the integration tests download models.
RUN ./mvnw -q -DskipTests -DskipITs -Dforbiddenapis.skip=true -pl opennlp-core/opennlp-runtime -am install

# The generator, built with the same Maven as OpenNLP. Its tests run here.
COPY . /generator
RUN /opennlp/mvnw -q -f /generator/pom.xml package

# The entrypoint passes any arguments given after the image name on to the
# generator; the driver passes --dumps so the decision point dumps are written
# beside the goldens.
RUN printf '%s\n' '#!/bin/sh' \
      'exec java -jar /generator/target/turbo-goldens-java.jar /inputs /models /fixtures /out "$(cat /opennlp.commit)" "$@"' \
      > /usr/local/bin/turbo-goldens-java \
 && chmod 0755 /usr/local/bin/turbo-goldens-java

ENTRYPOINT ["/usr/local/bin/turbo-goldens-java"]
