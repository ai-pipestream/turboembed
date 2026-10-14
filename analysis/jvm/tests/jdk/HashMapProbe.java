import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.CharBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs the HashMap scenarios of {@code tests/jdk_hash_map.rs} and prints,
 * for each, the map's entries in iteration order.
 *
 * <p>A scenario is one line: {@code kind ctor keys op...}, separated by
 * spaces. {@code kind} is the key class: {@code s} (String), {@code c}
 * (CharBuffer), {@code o} (a class that is not Comparable, with a given
 * hash code) or {@code i} (Integer). {@code ctor} is {@code d} for
 * {@code new HashMap<>()} or {@code n} and a capacity for
 * {@code new HashMap<>(capacity)}. {@code keys} is the keys, separated by
 * commas, no two equal: for {@code s} and {@code c} the UTF-16 units of each in hex
 * separated by dots ({@code _} for the empty string, {@code ~} for a
 * {@code null} String), and for {@code o}
 * and {@code i} a decimal int. An op is {@code p} and a key index
 * ({@code put(key, op number)}), {@code m} and a key index
 * ({@code merge(key, op number, Integer::sum)}), {@code a} and key indexes
 * separated by dots ({@code putAll} of a LinkedHashMap of those keys, each
 * with the op number) or {@code x} ({@code clear()}).
 *
 * <p>Each output line is the entries as {@code index=value}, separated by
 * commas.
 */
public final class HashMapProbe {

  /** A key that is not Comparable, equal only to a key with its index. */
  static final class Opaque {
    final int hash;
    final int index;

    Opaque(int hash, int index) {
      this.hash = hash;
      this.index = index;
    }

    @Override
    public int hashCode() {
      return hash;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof Opaque other && other.index == index;
    }
  }

  public static void main(String[] args) throws IOException {
    BufferedWriter out = new BufferedWriter(new OutputStreamWriter(System.out, StandardCharsets.US_ASCII));
    try (BufferedReader in = Files.newBufferedReader(Path.of(args[0]), StandardCharsets.US_ASCII)) {
      String line;
      while ((line = in.readLine()) != null) {
        out.write(run(line));
        out.write('\n');
      }
    }
    out.flush();
  }

  static String run(String line) {
    String[] parts = line.split(" ");
    String kind = parts[0];
    String ctor = parts[1];
    String[] texts = parts[2].split(",", -1);
    List<Object> keys = new ArrayList<>();
    Map<Object, Integer> indexOf = new HashMap<>();
    for (int i = 0; i < texts.length; i++) {
      Object key = switch (kind) {
        case "s" -> texts[i].equals("~") ? null : string(texts[i]);
        case "c" -> CharBuffer.wrap(string(texts[i]));
        case "o" -> new Opaque(Integer.parseInt(texts[i]), i);
        case "i" -> Integer.valueOf(Integer.parseInt(texts[i]));
        default -> throw new IllegalArgumentException(kind);
      };
      keys.add(key);
    }
    HashMap<Object, Integer> map = ctor.equals("d") ? new HashMap<>() : new HashMap<>(Integer.parseInt(ctor.substring(1)));
    for (int op = 3; op < parts.length; op++) {
      String o = parts[op];
      switch (o.charAt(0)) {
        case 'p' -> map.put(keys.get(Integer.parseInt(o.substring(1))), op);
        case 'm' -> map.merge(keys.get(Integer.parseInt(o.substring(1))), op, Integer::sum);
        case 'x' -> map.clear();
        case 'a' -> {
          Map<Object, Integer> source = new LinkedHashMap<>();
          for (String k : o.substring(1).split("\\.")) {
            source.put(keys.get(Integer.parseInt(k)), op);
          }
          map.putAll(source);
        }
        default -> throw new IllegalArgumentException(o);
      }
    }
    // The keys are distinct, so each is found by equality.
    for (int i = 0; i < keys.size(); i++) {
      indexOf.put(keys.get(i), i);
    }
    StringBuilder sb = new StringBuilder();
    for (Map.Entry<Object, Integer> e : map.entrySet()) {
      if (sb.length() > 0) {
        sb.append(',');
      }
      sb.append(indexOf.get(e.getKey())).append('=').append(e.getValue());
    }
    return sb.toString();
  }

  static String string(String text) {
    if (text.equals("_")) {
      return "";
    }
    StringBuilder sb = new StringBuilder();
    for (String unit : text.split("\\.")) {
      sb.append((char) Integer.parseInt(unit, 16));
    }
    return sb.toString();
  }
}
