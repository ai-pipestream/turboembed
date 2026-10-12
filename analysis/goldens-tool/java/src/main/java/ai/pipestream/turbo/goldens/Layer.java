package ai.pipestream.turbo.goldens;

import java.util.List;

/**
 * One layer block of a golden file: the layer id, its anchor ({@code text},
 * {@code aligned:<layer id>} or {@code document}) and its columns in the order the format lists
 * them. Every column has the same number of rows.
 */
record Layer(String id, String anchor, List<Column> columns) {

  Layer {
    columns = List.copyOf(columns);
    if (columns.isEmpty()) {
      throw new IllegalArgumentException("layer " + id + " has no columns");
    }
    int rows = columns.get(0).size();
    for (Column c : columns) {
      if (c.size() != rows) {
        throw new IllegalArgumentException("layer " + id + ": column " + c.name() + " has " + c.size()
            + " rows, column " + columns.get(0).name() + " has " + rows);
      }
    }
  }

  Layer(String id, String anchor, Column... columns) {
    this(id, anchor, List.of(columns));
  }

  /** The number of rows, the same in every column. */
  int rows() {
    return columns.get(0).size();
  }
}
