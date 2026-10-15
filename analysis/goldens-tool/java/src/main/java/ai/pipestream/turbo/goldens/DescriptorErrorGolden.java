package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.List;
import java.util.Set;

import opennlp.tools.util.featuregen.AdaptiveFeatureGenerator;
import opennlp.tools.util.featuregen.GeneratorFactory;

/**
 * What {@code GeneratorFactory.create} does with a malformed feature generator descriptor: each
 * input of the {@code descriptor-errors} set is one descriptor, built with a resource manager that
 * has nothing. The golden is the error record of what it threw, or, should it build, the layer
 * {@code turbo:generator} with the class of the generator it returned.
 */
final class DescriptorErrorGolden implements AnnotatorGolden {

  static final String SET = Runs.DESCRIPTOR_ERRORS;

  @Override
  public String annotator() {
    return "features";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return "errors";
  }

  @Override
  public String annotatorClass() {
    return GeneratorFactory.class.getName();
  }

  @Override
  public String method() {
    return "create";
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    AdaptiveFeatureGenerator g;
    try {
      g = GeneratorFactory.create(new ByteArrayInputStream(input.bytes()), key -> null);
    } catch (IOException e) {
      throw new AnnotatorException(e);
    } catch (RuntimeException e) {
      throw new AnnotatorException(e);
    }
    return List.of(new Table("turbo:generator", "class:str").row(g.getClass().getName()).layer());
  }
}
