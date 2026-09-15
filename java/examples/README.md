# Standalone Java example

`Embed.java` uses the installed native SDK and the Java API and FFM adapter
JARs. It does not download a model. Supply a prepared bundle for the pinned
`sentence-transformers/all-MiniLM-L6-v2` revision whose tokenizer SHA-256 is
checked by the example before it uses the model-specific prepared token IDs.

Build the Java JARs from the repository with JDK 25:

```sh
mvn -f java/pom.xml package
```

Copy `Embed.java` anywhere, then compile and run it with paths to those JARs:

```sh
API_JAR=/path/to/turboembed-api-0.1.0-SNAPSHOT.jar
FFM_JAR=/path/to/turboembed-ffm-0.1.0-SNAPSHOT.jar
javac --release 25 -cp "$API_JAR:$FFM_JAR" Embed.java
java --enable-native-access=ALL-UNNAMED -cp ".:$API_JAR:$FFM_JAR" \
  Embed /path/to/installed-sdk /path/to/prepared-model-bundle
```

The example explicitly selects OpenVINO GPU device ordinal zero. It embeds
`hello world`, checks the 384-float output is normalized, then uploads the
pinned WordPiece IDs and verifies that both paths agree.
