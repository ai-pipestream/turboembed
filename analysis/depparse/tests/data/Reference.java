package opennlp.tools.depparse;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import opennlp.tools.ml.maxent.quasinewton.QNModel;
import opennlp.tools.ml.model.Context;

/** Run against the pinned, unmodified OpenNLP sources identified in README.md. */
public final class Reference {
  private static final DependencyContextGenerator CONTEXT = new DependencyContextGenerator();
  private static final DependencyParserME PARSER = new DependencyParserME(new QNModel(
      new Context[] {new Context(new int[] {0, 1, 2}, new double[] {0, 1, 2})},
      new String[] {"s0t=NOUN"}, new String[] {"SHIFT", "LEFT_ARC:dep", "RIGHT_ARC:dep"}));
  private static final DependencyParserME TIES = new DependencyParserME(new QNModel(
      new Context[] {new Context(new int[] {0, 1}, new double[] {0, 0})},
      new String[] {"never"}, new String[] {"RIGHT_ARC:dep", "SHIFT"}));

  private static void line(String kind, String[] values) {
    System.out.println(kind + "\t" + String.join("\t", values));
  }
  private static void graph(String kind, DependencyGraph g) {
    String[] heads = new String[g.size()];
    String[] labels = new String[g.size()];
    for (int i = 0; i < g.size(); i++) {
      heads[i] = Integer.toString(g.headOf(i));
      labels[i] = g.relationOf(i);
    }
    line(kind + "H", heads);
    line(kind + "R", labels);
  }
  private static void sample(String[] words, String[] tags, int[] heads, String[] labels) {
    DependencyGraph gold = DependencyGraph.of(heads, labels);
    line("W", words); line("P", tags); graph("G", gold);
    List<Transition> steps;
    try { steps = ArcStandardOracle.transitions(gold); }
    catch (IllegalArgumentException e) { steps = List.of(); }
    System.out.println("O\t" + steps.size());
    ArcStandardState state = new ArcStandardState(words.length);
    for (Transition step : steps) {
      System.out.println("T\t" + step.encode() + "\t" + String.join("\t", CONTEXT.getContext(state, words, tags)));
      state.apply(step);
    }
    graph("M", PARSER.parse(words, tags));
    graph("Z", TIES.parse(words, tags));
  }
  public static void main(String[] args) throws Exception {
    sample(new String[] {"I", "see", "birds"}, new String[] {"PRON", "VERB", "NOUN"},
        new int[] {1, -1, 1}, new String[] {"nsubj", "root", "obj"});
    sample(new String[] {"a", "b", "c"}, new String[] {"X", "X", "X"},
        new int[] {2, -1, 1}, new String[] {"dep", "root", "dep"});
    String[] words = {"😀", "é", "中", "d", "e", "f", "g", "h"};
    String[] tags = new String[8]; Arrays.fill(tags, "NOUN");
    String[] labels = new String[8]; Arrays.fill(labels, "dep:part"); labels[0] = "root";
    sample(words, tags, new int[] {-1, 0, 0, 0, 0, 0, 0, 0}, labels);
    for (String arg : args) {
      for (String block : Files.readString(Path.of(arg)).strip().split("\n\n")) {
        List<String[]> rows = new ArrayList<>();
        for (String row : block.split("\n")) {
          String[] cols = row.split("\t");
          if (cols.length == 10 && cols[0].chars().allMatch(Character::isDigit)) rows.add(cols);
        }
        if (rows.isEmpty()) continue;
        sample(rows.stream().map(r -> r[1]).toArray(String[]::new),
            rows.stream().map(r -> r[3]).toArray(String[]::new),
            rows.stream().mapToInt(r -> Integer.parseInt(r[6]) - 1).toArray(),
            rows.stream().map(r -> r[7]).toArray(String[]::new));
      }
    }
  }
}
