# The container that makes the golden layer files: a pinned JDK, OpenNLP built
# from a pinned commit of the fork, the numeric and geo modules of the OpenNLP addons
# built from a pinned commit, and the generator in ../java.
# The liblinear add-on of the addons is compiled into a jar of its own beside them,
# and their neural-postag module is built from a commit of its own branch.
#
# Build it with `turbo-goldens image` (analysis/goldens-tool/README.md), which
# passes ../java as the build context. The base image digest and the commits
# are also Rust constants in ../src/recipe.rs; a test checks they agree.
# Models are not in the image: `turbo-goldens generate` mounts them, with the
# inputs and the checked in fixtures, read only, and runs the container without
# a network.

FROM eclipse-temurin:21-jdk@sha256:3e3c176ffed168beb42c607be9bc1639b466cf00261a0fb04425562c9d0c5c2b

ARG OPENNLP_REPO=https://github.com/ai-pipestream/opennlp
ARG OPENNLP_COMMIT=cdf33c653d5d61d5af7d37ffee189a83bd8392ea
ARG ADDONS_REPO=https://github.com/ai-pipestream/opennlp-addons
ARG ADDONS_COMMIT=33e95a4956f1535cdc074056b29246fe88b3c81a

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

# The runtime, the formats module (the sample conversion streams), the
# LIBSVM document categorizer module and the modules they depend on,
# installed into the local Maven repository. The forbidden API check is
# skipped because it fails on classes of this commit (NameSample and Parse in
# opennlp-api use java.util.regex); it only reads the compiled classes and
# changes nothing in them.
# -DskipITs is needed as well: the integration tests download models.
RUN ./mvnw -q -DskipTests -DskipITs -Dforbiddenapis.skip=true -pl opennlp-core/opennlp-runtime,opennlp-core/opennlp-formats,opennlp-core/opennlp-ml/opennlp-ml-libsvm -am install

# The code point StringPattern of the fork's ready/string-pattern-unicode
# branch: its StringPattern and FeatureGeneratorUtil, and its ParamChecks,
# which the commit above does not have, compiled against the runtime installed
# above into a jar of their own. The generator loads the jar in a class loader
# of its own for the code point pattern goldens (StringPatternGolden.java) and
# checks the commit written beside it.
ARG STRING_PATTERN_COMMIT=b2473ca80d65ae2ef10a24b08b1d0ae6a9231109
WORKDIR /string-pattern
RUN git -C /opennlp fetch -q --depth 1 origin "$STRING_PATTERN_COMMIT" \
 && mkdir src classes \
 && git -C /opennlp show "$STRING_PATTERN_COMMIT:opennlp-api/src/main/java/opennlp/tools/util/ParamChecks.java" > src/ParamChecks.java \
 && for f in StringPattern FeatureGeneratorUtil; do \
      git -C /opennlp show "$STRING_PATTERN_COMMIT:opennlp-core/opennlp-runtime/src/main/java/opennlp/tools/util/featuregen/$f.java" > "src/$f.java"; \
    done \
 && m2=/root/.m2/repository/org/apache/opennlp \
 && javac -encoding UTF-8 -d classes \
      -cp "$m2/opennlp-api/3.0.0-SNAPSHOT/opennlp-api-3.0.0-SNAPSHOT.jar:$m2/opennlp-runtime/3.0.0-SNAPSHOT/opennlp-runtime-3.0.0-SNAPSHOT.jar" \
      src/*.java \
 && jar cf string-pattern.jar -C classes . \
 && echo "$STRING_PATTERN_COMMIT" > commit

# The Hunspell classes of the fork's OPENNLP-1927-hunspell-compatibility branch,
# which are later than the commit above, compiled with HunspellBridge (../java/hunspell,
# which calls the package private HunspellStemmer.analyze) against the runtime
# installed above into a jar of their own. The generator loads the jar in a class
# loader of its own for the Hunspell goldens (HunspellGolden.java) and checks the
# commit written beside it, so the commit's own Hunspell classes are never used.
ARG HUNSPELL_COMMIT=fb299cf32ccf6f5dbc81b39448d2d7bea1cf66dc
WORKDIR /hunspell
COPY hunspell/HunspellBridge.java src/HunspellBridge.java
RUN git -C /opennlp fetch -q --depth 1 origin "$HUNSPELL_COMMIT" \
 && mkdir classes \
 && for f in AffixCondition HunspellAffixTable HunspellCompoundRule HunspellConversion HunspellDictionary HunspellStemmer; do \
      git -C /opennlp show "$HUNSPELL_COMMIT:opennlp-core/opennlp-runtime/src/main/java/opennlp/tools/stemmer/hunspell/$f.java" > "src/$f.java"; \
    done \
 && m2=/root/.m2/repository/org/apache/opennlp \
 && javac -encoding UTF-8 -d classes \
      -cp "$m2/opennlp-api/3.0.0-SNAPSHOT/opennlp-api-3.0.0-SNAPSHOT.jar:$m2/opennlp-runtime/3.0.0-SNAPSHOT/opennlp-runtime-3.0.0-SNAPSHOT.jar" \
      src/*.java \
 && jar cf hunspell.jar -C classes . \
 && echo "$HUNSPELL_COMMIT" > commit

# The numeric module of the OpenNLP addons (NumberScan), built against the
# runtime installed above. -am installs the addons' parent pom with it. Its
# tests are skipped; they are the addons' own. Git LFS files are not needed.
WORKDIR /addons
RUN git init -q . \
 && git remote add origin "$ADDONS_REPO" \
 && GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin "$ADDONS_COMMIT" \
 && GIT_LFS_SKIP_SMUDGE=1 git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "$ADDONS_COMMIT"
RUN /opennlp/mvnw -q -DskipTests -f /addons/pom.xml -pl numeric -am install

# The geo module of the same addons commit (the gazetteers), installed the
# same way. Two of its readers drop a byte order mark from the first line with
# StringUtil.stripByteOrderMark, which the OpenNLP fork added after the commit
# above; the sed writes out what that method does at both calls, and the grep
# checks that no call is left. Its tests are not compiled: they need DuckDB and
# Jackson for the addons' data tooling.
#
# PlaceProfiles (PlaceProfilesGolden.java) is not in that commit. Its source
# is taken from the commit below, on the ready/geo-place-profiles branch, and
# compiled into the module; every other class stays that of the commit above.
# It calls stripByteOrderMark once, which the sed replaces as well.
ARG PLACE_PROFILES_COMMIT=07e1f670818917d74e9a94367080f51ae6365bfa
RUN GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin "$PLACE_PROFILES_COMMIT" \
 && git show "$PLACE_PROFILES_COMMIT:geo/src/main/java/opennlp/geo/PlaceProfiles.java" \
      > geo/src/main/java/opennlp/geo/PlaceProfiles.java \
 && sed -i 's/StringUtil\.stripByteOrderMark(line)/(line.startsWith("\\uFEFF") ? line.substring(1) : line)/' \
      geo/src/main/java/opennlp/geo/GazetteerIndex.java geo/src/main/java/opennlp/geo/UserGazetteer.java \
      geo/src/main/java/opennlp/geo/PlaceProfiles.java \
 && ! grep -rq stripByteOrderMark geo/src/main \
 && /opennlp/mvnw -q -Dmaven.test.skip=true -f /addons/pom.xml -pl geo install

# The neural part-of-speech taggers of the OpenNLP addons (neural-postag: the
# BiLSTM-CRF and the feedforward tagger, PostagNeuralGolden.java), which the
# commit above does not have. The module's pom and main sources are taken from
# the commit below, on the ready/neural-postag-bilstm branch, and installed
# against the runtime above with the parent pom of the commit above. Its tests
# are not compiled. The empty .mvn directory makes /addons the project
# directory Maven reports when it builds the module on its own, where the
# parent's checkstyle configuration looks for checkstyle.xml.
ARG NEURAL_POSTAG_COMMIT=88767f8d09ffd350bd045e532bdd6f9d76cd9dd8
RUN GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin "$NEURAL_POSTAG_COMMIT" \
 && git archive "$NEURAL_POSTAG_COMMIT" neural-postag/pom.xml neural-postag/src/main | tar -x \
 && mkdir -p .mvn \
 && /opennlp/mvnw -q -Dmaven.test.skip=true -f /addons/neural-postag/pom.xml install

# The generator, built with the same Maven as OpenNLP. Its tests run here.
COPY . /generator
RUN /opennlp/mvnw -q -f /generator/pom.xml package

# The liblinear add-on of the OpenNLP addons, on their main branch: its three
# classes are in the default package, which the generator cannot import, so
# they are compiled from this commit, against the jars the generator's build
# copied to target/lib (OpenNLP and liblinear-java), into a jar of their own.
# The generator loads it in a class loader of its own for the liblinear
# trainer goldens (LiblinearTrainGolden.java) and checks the commit written
# beside it.
ARG LIBLINEAR_ADDON_COMMIT=84631d0111629110ebf094eadb6e938bbf7d738f
WORKDIR /liblinear-addon
RUN GIT_LFS_SKIP_SMUDGE=1 git -C /addons fetch -q --depth 1 origin "$LIBLINEAR_ADDON_COMMIT" \
 && mkdir src classes \
 && for f in LiblinearModel LiblinearModelSerializer LiblinearTrainer; do \
      git -C /addons show "$LIBLINEAR_ADDON_COMMIT:liblinear-addon/src/main/java/$f.java" > "src/$f.java"; \
    done \
 && javac -encoding UTF-8 -d classes -cp '/generator/target/lib/*' src/*.java \
 && jar cf liblinear-addon.jar -C classes . \
 && echo "$LIBLINEAR_ADDON_COMMIT" > commit

# The entrypoint passes any arguments given after the image name on to the
# generator; the driver passes --dumps so the decision point dumps are written
# beside the goldens.
RUN printf '%s\n' '#!/bin/sh' \
      'exec java -Dturbo.goldens.stringPatternJar=/string-pattern/string-pattern.jar -Dturbo.goldens.hunspellJar=/hunspell/hunspell.jar -Dturbo.goldens.liblinearJar=/liblinear-addon/liblinear-addon.jar -jar /generator/target/turbo-goldens-java.jar /inputs /models /fixtures /out "$(cat /opennlp.commit)" "$@"' \
      > /usr/local/bin/turbo-goldens-java \
 && chmod 0755 /usr/local/bin/turbo-goldens-java

ENTRYPOINT ["/usr/local/bin/turbo-goldens-java"]
