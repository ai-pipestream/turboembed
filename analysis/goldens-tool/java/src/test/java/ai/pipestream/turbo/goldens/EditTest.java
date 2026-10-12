package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import opennlp.tools.util.normalizer.AlignedText;
import opennlp.tools.util.normalizer.Alignment;
import org.junit.jupiter.api.Test;

class EditTest {

  private static AlignedText aligned(String original, String normalized, Alignment a) {
    return new AlignedText(original, normalized, a);
  }

  @Test
  void aCollapseIsOneEdit() {
    Alignment a = new Alignment.Builder().equal(1).replace(3, 1).equal(1).build(5);
    assertEquals(List.of(new Edit(1, 4, " ")), Edit.fromAlignment("axyzb", aligned("axyzb", "a b", a)));
  }

  @Test
  void adjacentOneToOneReplacementsAreSeparateEdits() {
    Alignment a = new Alignment.Builder().replace(1, 1).replace(1, 1).equal(1).build(3);
    assertEquals(List.of(new Edit(0, 1, "x"), new Edit(1, 2, "y")), Edit.fromAlignment("abc", aligned("abc", "xyc", a)));
  }

  @Test
  void aDeletionIsAnEmptyReplacementAndAnInsertionAnEmptyRange() {
    Alignment a = new Alignment.Builder().equal(1).replace(2, 0).equal(1).replace(0, 2).build(4);
    assertEquals(List.of(new Edit(1, 3, ""), new Edit(4, 4, "ZZ")),
        Edit.fromAlignment("abcd", aligned("abcd", "adZZ", a)));
  }

  @Test
  void aBlockThatKeepsItsTextIsNotAnEdit() {
    Alignment a = new Alignment.Builder().replace(2, 2).build(2);
    assertEquals(List.of(), Edit.fromAlignment("ab", aligned("ab", "ab", a)));
  }

  @Test
  void theDiffFindsTheChangedStretches() {
    assertEquals(List.of(new Edit(1, 2, "XY")), Edit.fromDiff("abc", "aXYc"));
    assertEquals(List.of(new Edit(0, 1, "x"), new Edit(1, 2, "y")), Edit.fromDiff("abc", "xyc"));
    assertEquals(List.of(new Edit(1, 3, "")), Edit.fromDiff("abcd", "ad"));
    assertEquals(List.of(), Edit.fromDiff("same", "same"));
    assertEquals(List.of(new Edit(0, 1, " ")), Edit.fromDiff("0 vs", "  vs"));
  }

  @Test
  void theDiffCountsCodePointsNotUnits() {
    String original = "a😀b";
    assertEquals(List.of(new Edit(1, 3, ":)")), Edit.fromDiff(original, "a:)b"));
  }

  @Test
  void applyingTheEditsGivesTheResult() {
    List<Edit> edits = Edit.fromDiff("The cat  sat", "The cat sat!");
    assertEquals("The cat sat!", Edit.apply("The cat  sat", edits));
  }
}
