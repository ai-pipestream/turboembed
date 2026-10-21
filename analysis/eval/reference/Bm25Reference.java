// Lucene's BM25 over the terms `turbo-eval run --retrieval bm25
// --dump-terms <dir>` wrote, as a reference for turbo-eval's own BM25: the
// same terms indexed with Lucene's BM25Similarity, every query searched,
// a document scored by its best chunk, its own id left out, and nDCG@10
// taken over the queries the qrels judge, with BEIR's linear gain.
//
// javac -cp lucene-core-9.12.3.jar:lucene-analysis-common-9.12.3.jar Bm25Reference.java
// java -cp .:lucene-core-9.12.3.jar:lucene-analysis-common-9.12.3.jar Bm25Reference \
//     <set>.chunks.tsv <set>.queries.tsv qrels/test.tsv [k1 b]

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;

public class Bm25Reference {
  public static void main(String[] args) throws Exception {
    float k1 = args.length > 3 ? Float.parseFloat(args[3]) : 1.2f;
    float b = args.length > 4 ? Float.parseFloat(args[4]) : 0.75f;
    var dir = new ByteBuffersDirectory();
    var config = new IndexWriterConfig(new WhitespaceAnalyzer()).setSimilarity(new BM25Similarity(k1, b));
    try (var w = new IndexWriter(dir, config)) {
      for (String line : Files.readAllLines(Path.of(args[0]))) {
        int tab = line.indexOf('\t');
        var d = new Document();
        d.add(new StoredField("id", line.substring(0, tab)));
        d.add(new TextField("t", line.substring(tab + 1), Field.Store.NO));
        w.addDocument(d);
      }
    }
    Map<String, Map<String, Integer>> qrels = new HashMap<>();
    for (String line : Files.readAllLines(Path.of(args[2]))) {
      String[] f = line.split("\t");
      if (f.length < 3 || f[0].equals("query-id")) continue;
      int grade = Integer.parseInt(f[2]);
      if (grade > 0) qrels.computeIfAbsent(f[0], k -> new HashMap<>()).put(f[1], grade);
    }
    IndexSearcher.setMaxClauseCount(1 << 20);
    try (var reader = DirectoryReader.open(dir)) {
      var searcher = new IndexSearcher(reader);
      searcher.setSimilarity(new BM25Similarity(k1, b));
      StoredFields stored = searcher.storedFields();
      double sum = 0;
      int judged = 0;
      for (String line : Files.readAllLines(Path.of(args[1]))) {
        int tab = line.indexOf('\t');
        String qid = line.substring(0, tab);
        Map<String, Integer> rel = qrels.get(qid);
        if (rel == null) continue;
        var q = new BooleanQuery.Builder();
        for (String term : line.substring(tab + 1).split(" ")) {
          if (!term.isEmpty()) q.add(new TermQuery(new Term("t", term)), BooleanClause.Occur.SHOULD);
        }
        Map<String, Float> best = new HashMap<>();
        for (ScoreDoc sd : searcher.search(q.build(), Math.max(1, reader.maxDoc())).scoreDocs) {
          String id = stored.document(sd.doc).get("id");
          if (!id.equals(qid)) best.merge(id, sd.score, Math::max);
        }
        List<Map.Entry<String, Float>> ranked = new ArrayList<>(best.entrySet());
        // Score descending, then document id descending, as turbo-eval ranks.
        ranked.sort((x, y) -> {
          int c = Float.compare(y.getValue(), x.getValue());
          return c != 0 ? c : y.getKey().compareTo(x.getKey());
        });
        double dcg = 0;
        for (int i = 0; i < Math.min(10, ranked.size()); i++) {
          dcg += rel.getOrDefault(ranked.get(i).getKey(), 0) / (Math.log(i + 2) / Math.log(2));
        }
        List<Integer> grades = new ArrayList<>(rel.values());
        grades.sort(Comparator.reverseOrder());
        double ideal = 0;
        for (int i = 0; i < Math.min(10, grades.size()); i++) ideal += grades.get(i) / (Math.log(i + 2) / Math.log(2));
        sum += dcg / ideal;
        judged++;
      }
      System.out.printf("queries=%d ndcg_at_10=%.4f%n", judged, sum / judged);
    }
  }
}
