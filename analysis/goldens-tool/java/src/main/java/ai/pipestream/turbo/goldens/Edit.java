package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import opennlp.tools.util.normalizer.AlignedText;
import opennlp.tools.util.normalizer.Alignment;

/**
 * One edit of a normalized view: the half open UTF-16 range {@code [start, end)} of the source
 * that is replaced, and the replacement. An empty range is an insertion and an empty replacement
 * is a deletion. A list of edits in text order, applied to the source, gives the normalized text.
 *
 * <p>Two ways make the list. {@link #fromAlignment} reads it from the {@link Alignment} an
 * OpenNLP normalizer reports. {@link #fromDiff} reconstructs it by comparing source and result on
 * code points, for the normalizers that report no alignment.</p>
 */
record Edit(int start, int end, String with) {

  /**
   * Reads the edits from an aligned result. Each normalized character reports the original range it
   * came from. Characters that share one range came from one replacement and form one edit;
   * characters that each have their own range of one character are separate edits, one per
   * character, as the alignment gives them. A range of the source that no normalized character
   * covers is a deletion. A block that reproduces its source text unchanged is not an edit.
   *
   * @throws IllegalStateException when the ranges are not in text order
   */
  static List<Edit> fromAlignment(String original, AlignedText aligned) {
    String normalized = aligned.normalizedString();
    Alignment a = aligned.alignment();
    if (a.normalizedLength() != normalized.length() || a.originalLength() != original.length()) {
      throw new IllegalStateException("the alignment does not match the texts' lengths");
    }
    List<Edit> edits = new ArrayList<>();
    int cursor = 0;
    int k = 0;
    while (k < normalized.length()) {
      var span = a.toOriginalSpan(k, k + 1);
      int s = span.getStart();
      int e = span.getEnd();
      int j = k + 1;
      while (j < normalized.length() && sameRange(a, j, s, e)) {
        j++;
      }
      if (s < cursor) {
        throw new IllegalStateException("original range [" + s + ", " + e + ") starts before " + cursor);
      }
      if (s > cursor) {
        edits.add(new Edit(cursor, s, ""));
      }
      String text = normalized.substring(k, j);
      if (!original.substring(s, e).equals(text)) {
        edits.add(new Edit(s, e, text));
      }
      cursor = e;
      k = j;
    }
    if (cursor < original.length()) {
      edits.add(new Edit(cursor, original.length(), ""));
    }
    return edits;
  }

  private static boolean sameRange(Alignment a, int k, int s, int e) {
    var span = a.toOriginalSpan(k, k + 1);
    return span.getStart() == s && span.getEnd() == e;
  }

  /** The shortest run of matched code points {@link #fromDiff} keeps inside the text. */
  private static final int MIN_ANCHOR = 3;

  /**
   * Reconstructs the edits from a longest common subsequence of the two texts' code points, in
   * three steps. The subsequence is walked from the start: equal code points match, and where they
   * differ a replacement of one code point by one is taken when that loses nothing from the
   * subsequence, otherwise a deletion or an insertion, the earlier match first. Runs of matched
   * code points shorter than {@value #MIN_ANCHOR} are then dropped, unless they touch the start or
   * the end of both texts, because one or two equal code points inside a replaced region are a
   * coincidence (the space a URL is replaced by matching a space after it) and not text the
   * normalizer kept. What lies between the remaining runs is the edits: a stretch with as many code
   * points on both sides is written as one edit per code point that differs, the rest as one edit
   * for the whole stretch.
   */
  static List<Edit> fromDiff(String original, String normalized) {
    int[] a = original.codePoints().toArray();
    int[] b = normalized.codePoints().toArray();
    int[][] lcs = new int[a.length + 1][b.length + 1];
    for (int i = a.length - 1; i >= 0; i--) {
      for (int j = b.length - 1; j >= 0; j--) {
        lcs[i][j] = a[i] == b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
      }
    }
    int[] pair = new int[a.length];
    Arrays.fill(pair, -1);
    int i = 0;
    int j = 0;
    while (i < a.length && j < b.length) {
      if (a[i] == b[j]) {
        pair[i] = j;
        i++;
        j++;
      } else if (lcs[i + 1][j + 1] == lcs[i][j]) {
        i++;
        j++;
      } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
        i++;
      } else {
        j++;
      }
    }
    dropShortRuns(pair, b.length);
    int[] aUnits = unitOffsets(a);
    List<Edit> edits = new ArrayList<>();
    int ai = 0;
    int bi = 0;
    while (ai < a.length) {
      int k = ai;
      while (k < a.length && pair[k] < 0) {
        k++;
      }
      int be = k < a.length ? pair[k] : b.length;
      if (k > ai || be > bi) {
        addStretch(edits, a, b, aUnits, ai, k, bi, be);
      }
      int end = k;
      while (end < a.length && pair[end] == be + (end - k)) {
        end++;
      }
      ai = end;
      bi = be + (end - k);
    }
    if (bi < b.length) {
      addStretch(edits, a, b, aUnits, a.length, a.length, bi, b.length);
    }
    return edits;
  }

  /** Unmatches the runs of consecutive pairs that are shorter than the anchor and not at an end. */
  private static void dropShortRuns(int[] pair, int bLength) {
    int k = 0;
    while (k < pair.length) {
      if (pair[k] < 0) {
        k++;
        continue;
      }
      int end = k;
      while (end < pair.length && pair[end] == pair[k] + (end - k)) {
        end++;
      }
      boolean atStart = k == 0 && pair[k] == 0;
      boolean atEnd = end == pair.length && pair[end - 1] == bLength - 1;
      if (end - k < MIN_ANCHOR && !atStart && !atEnd) {
        Arrays.fill(pair, k, end, -1);
      }
      k = end;
    }
  }

  private static void addStretch(List<Edit> edits, int[] a, int[] b, int[] aUnits, int ai, int ae, int bi, int be) {
    if (ae - ai == be - bi && ae - ai > 1) {
      for (int k = 0; k < ae - ai; k++) {
        if (a[ai + k] != b[bi + k]) {
          edits.add(new Edit(aUnits[ai + k], aUnits[ai + k + 1], new String(b, bi + k, 1)));
        }
      }
      return;
    }
    edits.add(new Edit(aUnits[ai], aUnits[ae], new String(b, bi, be - bi)));
  }

  /** The UTF-16 index of each code point, and the text's length after the last. */
  private static int[] unitOffsets(int[] cps) {
    int[] at = new int[cps.length + 1];
    for (int i = 0; i < cps.length; i++) {
      at[i + 1] = at[i] + Character.charCount(cps[i]);
    }
    return at;
  }

  /** Applies edits in text order to the source and returns the result. */
  static String apply(String original, List<Edit> edits) {
    StringBuilder out = new StringBuilder(original.length());
    int at = 0;
    for (Edit e : edits) {
      if (e.start < at || e.end < e.start) {
        throw new IllegalStateException("edits out of order at " + e);
      }
      out.append(original, at, e.start).append(e.with);
      at = e.end;
    }
    return out.append(original, at, original.length()).toString();
  }
}
