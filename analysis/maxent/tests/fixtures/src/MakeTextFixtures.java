import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.TreeMap;
import java.util.TreeSet;

import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.dictionary.serializer.DictionaryEntryPersistor;
import opennlp.tools.util.StringList;

/**
 * Loads the hand-written properties and dictionary files in this directory
 * with java.util.Properties and OpenNLP, and writes what they read, so the
 * Rust readers can be compared with them.
 */
public final class MakeTextFixtures {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0], "text");

    Properties props = new Properties();
    try (InputStream in = Files.newInputStream(dir.resolve("tricky.properties"))) {
      props.load(in);
    }
    try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(dir.resolve("tricky.properties.expected"),
        StandardCharsets.UTF_8))) {
      out.println("# key <TAB> value, sorted by key, \\uXXXX escaped");
      for (var e : new TreeMap<>(props).entrySet()) {
        out.println(escape((String) e.getKey()) + "\t" + escape((String) e.getValue()));
      }
    }

    byte[] xml = Files.readAllBytes(dir.resolve("tricky.dictionary"));
    try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(dir.resolve("tricky.dictionary.expected"),
        StandardCharsets.UTF_8))) {
      Dictionary d = new Dictionary(new ByteArrayInputStream(xml));
      out.println("case_sensitive " + d.isCaseSensitive());
      out.println("size " + d.size());
      out.println("min_tokens " + d.getMinTokenCount());
      out.println("max_tokens " + d.getMaxTokenCount());
      TreeSet<String> entries = new TreeSet<>();
      for (StringList s : d) {
        List<String> tokens = new ArrayList<>();
        for (String t : s) {
          tokens.add(escape(t));
        }
        entries.add(String.join("\t", tokens));
      }
      for (String e : entries) {
        out.println("set\t" + e);
      }
      String[][] probes = {{"Mr."}, {"mr."}, {"MR."}, {"New", "York"}, {"new", "york"}, {"New York"},
          {"a&b"}, {"<tag>"}, {"café"}, {"CAFÉ"}, {"line one\nline two"}, {"spaced"}, {"x"},
          {"e.g."}, {"E.G."}, {"ß"}, {"𐐀"}, {"𐐨"}, {"leading"}};
      for (String[] probe : probes) {
        List<String> tokens = new ArrayList<>();
        for (String t : probe) {
          tokens.add(escape(t));
        }
        out.println("contains\t" + d.contains(new StringList(probe)) + "\t" + String.join("\t", tokens));
      }
      DictionaryEntryPersistor.create(new ByteArrayInputStream(xml), entry -> {
        StringBuilder sb = new StringBuilder("entry");
        TreeMap<String, String> attributes = new TreeMap<>();
        for (var it = entry.attributes().iterator(); it.hasNext();) {
          String k = it.next();
          attributes.put(k, entry.attributes().getValue(k));
        }
        for (var a : attributes.entrySet()) {
          sb.append("\t@").append(escape(a.getKey())).append('=').append(escape(a.getValue()));
        }
        for (String t : entry.tokens()) {
          sb.append('\t').append(escape(t));
        }
        out.println(sb);
      });
    }
  }

  static String escape(String s) {
    StringBuilder sb = new StringBuilder();
    for (char c : s.toCharArray()) {
      if (c < 0x21 || c > 0x7e || c == '\\') {
        sb.append(String.format("\\u%04X", (int) c));
      } else {
        sb.append(c);
      }
    }
    return sb.toString();
  }
}
