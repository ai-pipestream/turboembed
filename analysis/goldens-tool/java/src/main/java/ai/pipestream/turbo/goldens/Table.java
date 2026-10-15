package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;

/**
 * A layer built a row at a time: its id and anchor, and its columns declared up front as
 * {@code name:u32}, {@code name:f64} or {@code name:str}. Each row gives one cell per column, in
 * order: a {@code long} or {@code int} for {@code u32}, a {@code double} for {@code f64} and a
 * {@code String} for {@code str}, checked as it is added, so a mistake fails where it is made
 * rather than in the file.
 */
final class Table {

  private final String id;
  private final String anchor;
  private final String[] names;
  private final char[] types;
  private final List<Object[]> rows = new ArrayList<>();

  /** A layer anchored to the document. */
  Table(String id, String... columns) {
    this(id, "document", columns);
  }

  /** A layer whose row i describes row i of the layer {@code of}. */
  static Table aligned(String id, String of, String... columns) {
    return new Table(id, "aligned:" + of, columns);
  }

  private Table(String id, String anchor, String[] columns) {
    this.id = id;
    this.anchor = anchor;
    this.names = new String[columns.length];
    this.types = new char[columns.length];
    for (int i = 0; i < columns.length; i++) {
      int colon = columns[i].lastIndexOf(':');
      names[i] = columns[i].substring(0, colon);
      types[i] = switch (columns[i].substring(colon + 1)) {
        case "u32" -> 'u';
        case "f64" -> 'f';
        case "str" -> 's';
        default -> throw new IllegalArgumentException(id + ": " + columns[i] + " is not name:u32, name:f64 or name:str");
      };
    }
  }

  /** Adds a row, one cell per column. */
  Table row(Object... cells) {
    if (cells.length != names.length) {
      throw new IllegalArgumentException(id + ": a row of " + cells.length + " cells for " + names.length + " columns");
    }
    for (int i = 0; i < cells.length; i++) {
      boolean ok = switch (types[i]) {
        case 'u' -> cells[i] instanceof Long || cells[i] instanceof Integer;
        case 'f' -> cells[i] instanceof Double;
        default -> cells[i] instanceof String;
      };
      if (!ok) {
        throw new IllegalArgumentException(id + ": column " + names[i] + " does not take " + cells[i]);
      }
    }
    rows.add(cells);
    return this;
  }

  /** The number of rows added so far. */
  int size() {
    return rows.size();
  }

  Layer layer() {
    List<Column> columns = new ArrayList<>(names.length);
    for (int c = 0; c < names.length; c++) {
      int n = rows.size();
      switch (types[c]) {
        case 'u' -> {
          long[] v = new long[n];
          for (int r = 0; r < n; r++) {
            v[r] = ((Number) rows.get(r)[c]).longValue();
          }
          columns.add(new Column.U32(names[c], v));
        }
        case 'f' -> {
          double[] v = new double[n];
          for (int r = 0; r < n; r++) {
            v[r] = (Double) rows.get(r)[c];
          }
          columns.add(new Column.F64(names[c], v));
        }
        default -> {
          String[] v = new String[n];
          for (int r = 0; r < n; r++) {
            v[r] = (String) rows.get(r)[c];
          }
          columns.add(new Column.Str(names[c], v));
        }
      }
    }
    return new Layer(id, anchor, columns);
  }

  /** 1 for true, 0 for false: how a flag is written in a {@code u32} column. */
  static long flag(boolean b) {
    return b ? 1 : 0;
  }
}
