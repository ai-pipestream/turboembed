package ai.pipestream.turbo.goldens;

/**
 * One column of a golden layer: a name and its values, one per row, of one of the three column
 * types the format knows. The writer turns each value into its text form.
 */
sealed interface Column permits Column.U32, Column.F64, Column.Str {

  /** The column's name, as written in the layer's column list. */
  String name();

  /** The column's type, as written after the colon in the layer's column list. */
  String type();

  /** The number of values, which is the layer's row count. */
  int size();

  /** Unsigned 32 bit integers, held as {@code long} so the whole range fits; each is checked. */
  record U32(String name, long[] values) implements Column {
    public U32 {
      for (long v : values) {
        if (v < 0 || v > 0xffff_ffffL) {
          throw new IllegalArgumentException("column " + name + ": " + v + " is not a u32");
        }
      }
    }

    @Override
    public String type() {
      return "u32";
    }

    @Override
    public int size() {
      return values.length;
    }
  }

  /** Doubles, written as their raw IEEE 754 bits followed by the decimal form. */
  record F64(String name, double[] values) implements Column {
    @Override
    public String type() {
      return "f64";
    }

    @Override
    public int size() {
      return values.length;
    }
  }

  /** Strings, written with the format's escapes. */
  record Str(String name, String[] values) implements Column {
    @Override
    public String type() {
      return "str";
    }

    @Override
    public int size() {
      return values.length;
    }
  }
}
