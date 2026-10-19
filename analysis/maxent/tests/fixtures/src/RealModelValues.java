import java.io.File;

import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.tokenize.DefaultTokenContextGenerator;
import opennlp.tools.tokenize.TokenizerModel;

/**
 * Prints, for one TokenizerME model, its predicate count and the
 * probabilities its maxent model gives for ten contexts, in the form
 * tests/models.rs holds them. The contexts are made by OpenNLP's own
 * DefaultTokenContextGenerator, plus a context with a predicate no model
 * has and an empty context.
 */
public final class RealModelValues {
  public static void main(String[] args) throws Exception {
    TokenizerModel model = new TokenizerModel(new File(args[0]));
    MaxentModel maxent = model.getMaxentModel();
    Object[] data = ((AbstractModel) maxent).getDataStructures();
    System.out.println("predicates " + ((java.util.Map<?, ?>) data[1]).size());
    StringBuilder outcomes = new StringBuilder("outcomes");
    for (int i = 0; i < maxent.getNumOutcomes(); i++) {
      outcomes.append(' ').append(maxent.getOutcome(i));
    }
    System.out.println(outcomes);
    DefaultTokenContextGenerator generator = new DefaultTokenContextGenerator();
    String[][] units = {
        {"Hola,", "4"}, {"dijo:", "4"}, {"(España)", "1"}, {"l'homme", "2"}, {"aujourd'hui", "8"},
        {"3,5%", "3"}, {"\"Sí\".", "4"}, {"etc.", "3"}};
    for (String[] unit : units) {
      print(maxent, generator.getContext(unit[0], Integer.parseInt(unit[1])));
    }
    print(maxent, new String[] {"no-such-predicate", "p=H"});
    print(maxent, new String[0]);
  }

  static void print(MaxentModel maxent, String[] context) {
    double[] probs = maxent.eval(context);
    StringBuilder sb = new StringBuilder("(&[");
    for (int i = 0; i < context.length; i++) {
      sb.append(i > 0 ? ", " : "").append('"').append(context[i].replace("\\", "\\\\").replace("\"", "\\\""))
          .append('"');
    }
    sb.append("], [");
    for (int i = 0; i < probs.length; i++) {
      sb.append(i > 0 ? ", " : "").append(Double.toString(probs[i]));
    }
    System.out.println(sb.append("]),"));
  }
}
