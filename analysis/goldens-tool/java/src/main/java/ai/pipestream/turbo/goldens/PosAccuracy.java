package ai.pipestream.turbo.goldens;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;

/**
 * Scores a part-of-speech model on a CoNLL-U file, the way the Rust port's accuracy test
 * (analysis/postag/tests/accuracy.rs) does, so the two can be compared token by token:
 *
 * <pre>java -cp turbo-goldens-java.jar ai.pipestream.turbo.goldens.PosAccuracy \
 *     &lt;model.bin&gt; &lt;test.conllu&gt; [&lt;tags.txt&gt;]</pre>
 *
 * <p>The file is read as OpenNLP's {@code ConlluStream} and {@code ConlluPOSSampleStream}
 * read it for {@code POSTaggerEvaluator}: a sentence per blank-line block, comment lines
 * skipped, every other line ten tab-separated fields; a multiword token line ({@code 3-4})
 * keeps its form and takes the UPOS tags of its parts joined with {@code +}, and the part
 * lines are dropped; an empty node ({@code 8.1}) stays a token with the tag {@code _}. Both
 * of the last two are tokens the tagger can never get right, so the tagger is scored once over
 * every token, as {@code POSTaggerEvaluator} scores it, and once over the plain word lines
 * only. The treebank text is read where it is and never copied anywhere.</p>
 *
 * <p>With a third argument the tags are written there, one sentence per line, tags separated
 * by spaces, for the Rust test to compare against.</p>
 */
public final class PosAccuracy {

  private PosAccuracy() {}

  /** One sentence: the forms, the reference tags and whether each token is a plain word line. */
  record Sentence(List<String> forms, List<String> tags, List<Boolean> plain) {}

  public static void main(String[] args) throws IOException {
    if (args.length < 2 || args.length > 3) {
      System.err.println("usage: PosAccuracy <model.bin> <test.conllu> [<tags.txt>]");
      System.exit(2);
    }
    POSTaggerME tagger;
    try (var in = Files.newInputStream(Path.of(args[0]))) {
      tagger = new POSTaggerME(new POSModel(in));
    }
    List<Sentence> sentences = read(Path.of(args[1]));
    PrintWriter tagsOut = args.length == 3
        ? new PrintWriter(Files.newBufferedWriter(Path.of(args[2]), StandardCharsets.UTF_8))
        : null;
    long tokens = 0;
    long correct = 0;
    long plainTokens = 0;
    long plainCorrect = 0;
    for (Sentence s : sentences) {
      String[] tagged = tagger.tag(s.forms().toArray(new String[0]));
      for (int i = 0; i < tagged.length; i++) {
        boolean right = s.tags().get(i).equals(tagged[i]);
        tokens++;
        correct += right ? 1 : 0;
        if (s.plain().get(i)) {
          plainTokens++;
          plainCorrect += right ? 1 : 0;
        }
      }
      if (tagsOut != null) {
        tagsOut.println(String.join(" ", tagged));
      }
    }
    if (tagsOut != null) {
      tagsOut.close();
    }
    System.out.printf("sentences\t%d%n", sentences.size());
    System.out.printf("tokens\t%d\tcorrect\t%d\taccuracy\t%.6f%n", tokens, correct, (double) correct / tokens);
    System.out.printf("plain word lines\t%d\tcorrect\t%d\taccuracy\t%.6f%n", plainTokens, plainCorrect,
        (double) plainCorrect / plainTokens);
  }

  /** Reads the sentences of a CoNLL-U file as described on the class. */
  static List<Sentence> read(Path conllu) throws IOException {
    List<Sentence> sentences = new ArrayList<>();
    List<String[]> block = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(conllu, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isEmpty()) {
          if (!block.isEmpty()) {
            sentences.add(sentence(block));
            block.clear();
          }
          continue;
        }
        if (line.trim().startsWith("#")) {
          continue;
        }
        String[] fields = line.split("\t", -1);
        if (fields.length != 10) {
          throw new IOException("line [" + line + "] must have exactly 10 fields");
        }
        block.add(fields);
      }
    }
    if (!block.isEmpty()) {
      sentences.add(sentence(block));
    }
    return sentences;
  }

  /** One blank-line block's word lines, with every multiword range merged into its line. */
  private static Sentence sentence(List<String[]> lines) throws IOException {
    List<String> forms = new ArrayList<>();
    List<String> tags = new ArrayList<>();
    List<Boolean> plain = new ArrayList<>();
    int i = 0;
    while (i < lines.size()) {
      String[] line = lines.get(i);
      String id = line[0];
      int dash = id.indexOf('-');
      if (dash != -1) {
        int start = Integer.parseInt(id.substring(0, dash));
        int end = Integer.parseInt(id.substring(dash + 1));
        if (end < start) {
          throw new IOException("multiword token id runs backwards: " + id);
        }
        // The parts follow their range line, one per id. ConlluStream finds them by id
        // anywhere in the sentence and drops them; a file where they are not right behind
        // the range line is refused here instead.
        int parts = end - start + 1;
        StringBuilder upos = new StringBuilder();
        for (int k = 0; k < parts; k++) {
          String[] part = i + 1 + k < lines.size() ? lines.get(i + 1 + k) : null;
          if (part == null || !part[0].equals(Integer.toString(start + k))) {
            throw new IOException("multiword token " + id + " is not followed by its word line " + (start + k));
          }
          if (!"_".equals(part[3])) {
            if (upos.length() > 0) {
              upos.append('+');
            }
            upos.append(part[3]);
          }
        }
        forms.add(line[1]);
        tags.add(upos.toString());
        plain.add(false);
        i += 1 + parts;
        continue;
      }
      forms.add(line[1]);
      tags.add(line[3]);
      plain.add(id.indexOf('.') == -1);
      i++;
    }
    return new Sentence(forms, tags, plain);
  }
}
