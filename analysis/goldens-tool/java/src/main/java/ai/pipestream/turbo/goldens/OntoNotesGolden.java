package ai.pipestream.turbo.goldens;

import java.io.File;
import java.io.IOException;
import java.io.FileFilter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.cmdline.StreamFactoryRegistry;
import opennlp.tools.formats.DirectorySampleStream;
import opennlp.tools.formats.convert.FileToStringSampleStream;
import opennlp.tools.formats.ontonotes.DocumentToLineStream;
import opennlp.tools.formats.ontonotes.OntoNotesNameSampleStream;
import opennlp.tools.formats.ontonotes.OntoNotesParseSampleStream;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.parser.Parse;
import opennlp.tools.postag.POSSample;
import opennlp.tools.formats.convert.ParseToPOSSampleStream;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.WhitespaceMode;

/**
 * The goldens of OpenNLP's OntoNotes readers ({@code opennlp.tools.formats.ontonotes}) over the
 * {@code ontonotes} input set, each input a bundle of a corpus directory (written out as
 * {@link MascGolden#writeBundle} writes it). The streams are made by the {@code ontonotes} factories
 * of the registry, given the directory as {@code -ontoNotesDir}: the name samples of the
 * {@code .name} files ({@code names}, also with the legacy whitespace, {@code names+legacy}), the
 * parses of the {@code .parse} files ({@code parse}) and the part-of-speech samples made of them
 * ({@code pos}). The {@code lines} run reads the {@code .parse} files with a
 * {@code DocumentToLineStream}, as the parse factory does before it parses them.
 *
 * <p>A stream is read until {@code read} returns null, with an exception recorded and the reading
 * going on. The layers, anchored to the document: {@code turbo:samples} and
 * {@code turbo:read-errors}; for names {@code turbo:sample-tokens} and {@code turbo:names}, for
 * parses those {@link ParserGolden} writes, for part-of-speech samples {@code turbo:sample-tokens}
 * and {@code turbo:sample-tags}, for lines {@code turbo:lines}; and {@code turbo:listing}, the
 * files of the corpus, relative to it, in the order the factory's {@code DirectorySampleStream}
 * gives them. A message that names the corpus directory writes it {@code <corpus>}.</p>
 */
final class OntoNotesGolden implements AnnotatorGolden {

  /** The input set the bundles are read from. */
  static final String SET = Runs.ONTONOTES;

  /** The stopping point for a stream that throws without moving on, which would otherwise loop. */
  private static final int MAX_READS = 100_000;

  enum Kind {
    NAMES("names", OntoNotesNameSampleStream.class, ".name"),
    PARSE("parse", OntoNotesParseSampleStream.class, ".parse"),
    POS("pos", ParseToPOSSampleStream.class, ".parse"),
    LINES("lines", DocumentToLineStream.class, ".parse");

    final String dir;
    final Class<?> streamClass;
    final String suffix;

    Kind(String dir, Class<?> streamClass, String suffix) {
      this.dir = dir;
      this.streamClass = streamClass;
      this.suffix = suffix;
    }
  }

  private final Kind kind;
  private final WhitespaceMode whitespace;

  private OntoNotesGolden(Kind kind, WhitespaceMode whitespace) {
    this.kind = kind;
    this.whitespace = whitespace;
  }

  static List<OntoNotesGolden> all() {
    List<OntoNotesGolden> runs = new ArrayList<>();
    for (Kind k : Kind.values()) {
      runs.add(new OntoNotesGolden(k, WhitespaceMode.UNICODE));
    }
    runs.add(new OntoNotesGolden(Kind.NAMES, WhitespaceMode.LEGACY));
    return runs;
  }

  @Override
  public String annotator() {
    return "ontonotes";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String annotatorClass() {
    return kind.streamClass.getName();
  }

  @Override
  public String method() {
    return "read";
  }

  @Override
  public Map<String, String> config() {
    return Whitespace.config(whitespace);
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public String modelDir() {
    return kind.dir + Whitespace.dirSuffix(whitespace);
  }

  /** The factory's filter: the files with the run's suffix, and the directories. */
  private FileFilter filter() {
    return file -> file.isFile() ? file.getName().endsWith(kind.suffix) : file.isDirectory();
  }

  @SuppressWarnings("unchecked")
  private ObjectStream<Object> open(Path root) {
    String[] args = {"-ontoNotesDir", root.toString()};
    return switch (kind) {
      case NAMES -> (ObjectStream<Object>) (ObjectStream<?>)
          StreamFactoryRegistry.getFactory(NameSample.class, "ontonotes").create(args);
      case PARSE -> (ObjectStream<Object>) (ObjectStream<?>)
          StreamFactoryRegistry.getFactory(Parse.class, "ontonotes").create(args);
      case POS -> (ObjectStream<Object>) (ObjectStream<?>)
          StreamFactoryRegistry.getFactory(POSSample.class, "ontonotes").create(args);
      case LINES -> (ObjectStream<Object>) (ObjectStream<?>) new DocumentToLineStream(
          new FileToStringSampleStream(new DirectorySampleStream(root.toFile(), filter(), true),
              StandardCharsets.UTF_8));
    };
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    Path root = null;
    try {
      root = Files.createTempDirectory("ontonotes-golden");
      MascGolden.writeBundle(input, root);
      Path dir = root;
      return Whitespace.in(whitespace, () -> read(input, dir));
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the corpus could not be written", e);
    } finally {
      if (root != null) {
        MascGolden.delete(root);
      }
    }
  }

  private List<Layer> read(Input input, Path root) {
    List<Object> samples = new ArrayList<>();
    FormatGolden.Columns errors = new FormatGolden.Columns();
    try (ObjectStream<Object> stream = open(root)) {
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        Object sample;
        try {
          sample = stream.read();
        } catch (IOException | RuntimeException e) {
          errors.u32("after", samples.size()).str("class", e.getClass().getName())
              .str("message", message(e, root));
          continue;
        }
        if (sample == null) {
          break;
        }
        samples.add(sample);
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the stream could not be closed", e);
    }

    List<Layer> layers = new ArrayList<>();
    if (kind == Kind.PARSE) {
      List<Parse> parses = new ArrayList<>();
      for (Object s : samples) {
        parses.add((Parse) s);
      }
      layers.addAll(ParserGolden.layers(input, parses, errors));
    } else {
      FormatGolden.Columns written = new FormatGolden.Columns();
      for (Object s : samples) {
        written.str("written", s.toString());
      }
      layers.add(written.layer("turbo:samples", "document", "written"));
      layers.add(errors.layer("turbo:read-errors", "document", "after", "class", "message"));
      switch (kind) {
        case NAMES -> FormatGolden.names(samples, layers);
        case POS -> FormatGolden.tagged(samples, layers);
        case LINES -> {
          FormatGolden.Columns lines = new FormatGolden.Columns();
          for (Object s : samples) {
            lines.str("line", (String) s);
          }
          layers.add(lines.layer("turbo:lines", "document", "line"));
        }
        case PARSE -> throw new IllegalStateException("unreachable");
      }
    }
    layers.add(listing(root));
    return layers;
  }

  /** The files the factory's {@code DirectorySampleStream} gives, relative to {@code root}. */
  private Layer listing(Path root) {
    FormatGolden.Columns files = new FormatGolden.Columns();
    try (DirectorySampleStream d = new DirectorySampleStream(root.toFile(), filter(), true)) {
      for (File f = d.read(); f != null; f = d.read()) {
        files.str("file", root.relativize(f.toPath()).toString());
      }
    } catch (IOException e) {
      throw new IllegalStateException("could not list " + root, e);
    }
    return files.layer("turbo:listing", "document", "file");
  }

  /** {@code message} with the corpus directory written {@code <corpus>}. */
  static String message(Exception e, Path root) {
    return FormatGolden.message(e).replace(root.toString(), "<corpus>");
  }
}
