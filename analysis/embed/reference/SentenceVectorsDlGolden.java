/*
 * Licensed under the Apache License, Version 2.0.
 */

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;

import opennlp.dl.vectors.SentenceVectorsDL;

/**
 * Generates {@code analysis/embed/tests/data/sentence_vectors_dl_golden.json}
 * by running OpenNLP's {@link SentenceVectorsDL} in the pinned JVM container
 * ({@code turbo-goldens-java:opennlp-cdf33c653d5d-addons-33e95a4956f1}) over:
 * <ul>
 *   <li>the 12 test sentences of {@code analysis/embed/tests/vectors.rs}, and</li>
 *   <li>the 76 golden inputs of the {@code edge}, {@code en}, {@code multi},
 *       {@code subwords} and {@code surrogate} sets under
 *       {@code analysis/goldens/data/inputs},</li>
 * </ul>
 * using two ONNX exports of {@code testdata/tiny-bert-bundle/weights/model.safetensors}
 * with the shared vocabulary {@code analysis/goldens/data/fixtures/all-minilm-l6-v2-vocab.txt}:
 * <ol>
 *   <li>{@code mean_l2.onnx}: the 2-layer BERT encoder followed by masked mean
 *       pooling over all live tokens ({@code [CLS] + pieces + [SEP]}) and L2
 *       normalization, shaped {@code [batch, 1, 32]} so {@code getVectors}
 *       returns the whole-row mean-pooled unit vector;</li>
 *   <li>{@code cls.onnx}: the raw 2-layer BERT {@code last_hidden_state}
 *       shaped {@code [batch, seq, 32]} so {@code getVectors} returns
 *       {@code last_hidden_state[0][0]}, the {@code [CLS]} position vector.</li>
 * </ol>
 *
 * <p>Arguments: {@code <mean_l2.onnx> <cls.onnx> <vocab.txt> <weights.safetensors>
 * <inputs_dir> <opennlp_commit> <out.json>}.</p>
 */
public final class SentenceVectorsDlGolden {

  private static final String[] SENTENCES = {
    "Embedding models turn a passage of text into a vector.",
    "Passages with similar meaning land near each other.",
    "A search system embeds every document once.",
    "It stores the vectors in an index for later.",
    "Each query is embedded as it arrives.",
    "The nearest documents by cosine are the candidates.",
    "The model reads a fixed number of tokens.",
    "Long documents are split into chunks first.",
    "Each chunk then gets a vector of its own.",
    "Pooling averages the token states into one vector.",
    "Normalization scales that vector to unit length.",
    "The rest of the pipeline reads those vectors."
  };

  private static final String[] SETS = {"edge", "en", "multi", "subwords", "surrogate"};

  private SentenceVectorsDlGolden() {
  }

  public static void main(String[] args) throws Exception {
    File meanModel = new File(args[0]);
    File clsModel = new File(args[1]);
    File vocab = new File(args[2]);
    Path weights = Path.of(args[3]);
    Path inputsRoot = Path.of(args[4]);
    String commit = args[5].trim();
    Path outPath = Path.of(args[6]);

    List<String> names = new ArrayList<>();
    List<String> texts = new ArrayList<>();
    for (String set : SETS) {
      List<Path> files = new ArrayList<>();
      try (DirectoryStream<Path> ds = Files.newDirectoryStream(inputsRoot.resolve(set), "*.txt")) {
        for (Path p : ds) {
          files.add(p);
        }
      }
      Collections.sort(files);
      for (Path p : files) {
        String id = p.getFileName().toString().replaceFirst("\\.txt$", "");
        names.add(set + "/" + id);
        texts.add(Files.readString(p, StandardCharsets.UTF_8));
      }
    }

    StringBuilder sb = new StringBuilder("{\n");
    sb.append("  \"opennlp_commit\": ").append(quote(commit)).append(",\n");
    sb.append("  \"vocab_file\": \"all-minilm-l6-v2-vocab.txt\",\n");
    sb.append("  \"vocab_sha256\": ").append(quote(sha256(vocab.toPath()))).append(",\n");
    sb.append("  \"weights_file\": \"testdata/tiny-bert-bundle/weights/model.safetensors\",\n");
    sb.append("  \"weights_sha256\": ").append(quote(sha256(weights))).append(",\n");
    sb.append("  \"sentences\": [\n");
    try (SentenceVectorsDL meanDl = new SentenceVectorsDL(meanModel, vocab, true);
         SentenceVectorsDL clsDl = new SentenceVectorsDL(clsModel, vocab, true)) {
      for (int i = 0; i < SENTENCES.length; i++) {
        float[] m = meanDl.getVectors(SENTENCES[i]);
        float[] c = clsDl.getVectors(SENTENCES[i]);
        sb.append("    {\"text\": ").append(quote(SENTENCES[i]))
            .append(", \"mean_l2\": ").append(vec(m))
            .append(", \"cls\": ").append(vec(c)).append("}")
            .append(i + 1 < SENTENCES.length ? ",\n" : "\n");
      }
      sb.append("  ],\n  \"golden_inputs\": [\n");
      for (int i = 0; i < names.size(); i++) {
        float[] m = meanDl.getVectors(texts.get(i));
        float[] c = clsDl.getVectors(texts.get(i));
        sb.append("    {\"name\": ").append(quote(names.get(i)))
            .append(", \"mean_l2\": ").append(vec(m))
            .append(", \"cls\": ").append(vec(c)).append("}")
            .append(i + 1 < names.size() ? ",\n" : "\n");
      }
      sb.append("  ]\n}\n");
    }
    Files.writeString(outPath, sb.toString(), StandardCharsets.UTF_8);
  }

  private static String sha256(Path file) throws Exception {
    MessageDigest md = MessageDigest.getInstance("SHA-256");
    return HexFormat.of().formatHex(md.digest(Files.readAllBytes(file)));
  }

  private static String vec(float[] v) {
    StringBuilder b = new StringBuilder("[");
    for (int i = 0; i < v.length; i++) {
      if (i > 0) {
        b.append(", ");
      }
      b.append(Float.toString(v[i]));
    }
    return b.append("]").toString();
  }

  private static String quote(String s) {
    StringBuilder b = new StringBuilder("\"");
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == '"' || c == '\\') {
        b.append('\\').append(c);
      } else if (c < 0x20) {
        b.append(String.format("\\u%04x", (int) c));
      } else {
        b.append(c);
      }
    }
    return b.append('"').toString();
  }
}
