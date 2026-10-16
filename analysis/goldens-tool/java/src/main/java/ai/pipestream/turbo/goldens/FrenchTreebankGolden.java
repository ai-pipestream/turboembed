package ai.pipestream.turbo.goldens;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import opennlp.tools.cmdline.StreamFactoryRegistry;
import opennlp.tools.formats.DirectorySampleStream;
import opennlp.tools.formats.frenchtreebank.ConstitParseSampleStream;
import opennlp.tools.parser.Parse;
import opennlp.tools.util.ObjectStream;

/**
 * The goldens of OpenNLP's French Treebank reader ({@code ConstitParseSampleStream} with its
 * {@code ConstitDocumentHandler}) over the {@code frenchtreebank} input set, each input a bundle of
 * a data directory (written out as {@link MascGolden#writeBundle} writes it). The stream is made by
 * the {@code frenchtreebank} factory of the registry, given the directory as {@code -data}: every
 * file of the directory, not of its subdirectories, parsed as XML.
 *
 * <p>The stream is read until {@code read} returns null, with an exception recorded and the reading
 * going on. The layers are those {@link ParserGolden} writes for parses, then
 * {@code turbo:listing}, the files in the order the factory's {@code DirectorySampleStream} gives
 * them, relative to the directory. The message of an XML error is the JDK's parser's.</p>
 */
final class FrenchTreebankGolden implements AnnotatorGolden {

  /** The input set the bundles are read from. */
  static final String SET = Runs.FRENCH_TREEBANK;

  /** The stopping point for a stream that throws without moving on, which would otherwise loop. */
  private static final int MAX_READS = 100_000;

  static List<FrenchTreebankGolden> all() {
    return List.of(new FrenchTreebankGolden());
  }

  @Override
  public String annotator() {
    return "frenchtreebank";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String annotatorClass() {
    return ConstitParseSampleStream.class.getName();
  }

  @Override
  public String method() {
    return "read";
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public String modelDir() {
    return "parse";
  }

  @Override
  public List<Layer> layers(Input input) {
    Path root = null;
    try {
      root = Files.createTempDirectory("frenchtreebank-golden");
      MascGolden.writeBundle(input, root);
      return read(input, root);
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the corpus could not be written", e);
    } finally {
      if (root != null) {
        MascGolden.delete(root);
      }
    }
  }

  private static List<Layer> read(Input input, Path root) {
    List<Parse> parses = new ArrayList<>();
    FormatGolden.Columns errors = new FormatGolden.Columns();
    String[] args = {"-data", root.toString()};
    try (ObjectStream<Parse> stream = StreamFactoryRegistry.getFactory(Parse.class, "frenchtreebank")
        .create(args)) {
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        Parse p;
        try {
          p = stream.read();
        } catch (IOException | RuntimeException e) {
          errors.u32("after", parses.size()).str("class", e.getClass().getName())
              .str("message", OntoNotesGolden.message(e, root));
          continue;
        }
        if (p == null) {
          break;
        }
        parses.add(p);
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the stream could not be closed", e);
    }
    List<Layer> layers = new ArrayList<>(ParserGolden.layers(input, parses, errors));
    FormatGolden.Columns files = new FormatGolden.Columns();
    try (DirectorySampleStream d = new DirectorySampleStream(root.toFile(), null, false)) {
      for (File f = d.read(); f != null; f = d.read()) {
        files.str("file", root.relativize(f.toPath()).toString());
      }
    } catch (IOException e) {
      throw new IllegalStateException("could not list " + root, e);
    }
    layers.add(files.layer("turbo:listing", "document", "file"));
    return layers;
  }
}
