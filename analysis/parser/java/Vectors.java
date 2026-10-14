// Writes the JVM reference test vectors for the turbo-parser crate.
//
// Run from the repository root inside the pinned Temurin 21 container:
//
//   docker run --rm -u $(id -u):$(id -g) \
//     -v "$PWD":/work -v "$HOME/.m2":/m2:ro -w /work \
//     eclipse-temurin:21-jdk@sha256:3e3c176ffed168beb42c607be9bc1639b466cf00261a0fb04425562c9d0c5c2b \
//     java -cp /m2/repository/org/apache/opennlp/opennlp-api/3.0.0-SNAPSHOT/opennlp-api-3.0.0-SNAPSHOT.jar:/m2/repository/org/apache/opennlp/opennlp-runtime/3.0.0-SNAPSHOT/opennlp-runtime-3.0.0-SNAPSHOT.jar \
//     analysis/parser/java/Vectors.java analysis/parser/tests/data analysis/parser/tests/vectors

import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.parser.ChunkSampleStream;
import opennlp.tools.parser.HeadRules;
import opennlp.tools.parser.Parse;
import opennlp.tools.parser.ParseSampleStream;
import opennlp.tools.parser.PosSampleStream;
import opennlp.tools.parser.lang.es.AncoraSpanishHeadRules;
import opennlp.tools.postag.POSSample;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Span;

public class Vectors {

    private static final String PARSE_STRING =
        "(TOP  (S (S (NP-SBJ (PRP She)  )(VP (VBD was)  "
        + "(ADVP (RB just)  )(NP-PRD (NP (DT another)  (NN freighter)  )(PP (IN from)  (NP (DT the)  "
        + "(NNPS States)  )))))(, ,)  (CC and) (S (NP-SBJ (PRP she)  )(VP (VBD seemed)  "
        + "(ADJP-PRD (ADJP (RB as)  (JJ commonplace)  )(PP (IN as)  (NP (PRP$ her)  "
        + "(NN name)  )))))(. .)  ))";

    private static final String TOKEN_REPLACEMENT_TREE =
        "(TOP  (S-CLF (NP-SBJ (PRP It)  )(VP (VBD was) "
        + " (NP-PRD (NP (DT the)  (NN trial)  )(PP (IN of) "
        + " (NP (NP (NN oleomargarine)  (NN heir)  )(NP (NNP Minot) "
        + " (PRN (-LRB- -LRB-) (NNP Mickey) "
        + " (-RRB- -RRB-) )(NNP Jelke)  )))(PP (IN for) "
        + " (NP (JJ compulsory)  (NN prostitution) "
        + " ))(PP-LOC (IN in)  (NP (NNP New)  (NNP York) "
        + " )))(SBAR (WHNP-1 (WDT that)  )(S (VP (VBD put) "
        + " (NP (DT the)  (NN spotlight)  )(PP (IN on)  (NP (DT the) "
        + " (JJ international)  (NN play-girl)  ))))))(. .)  ))";

    private static final String EN_GAP_SUBJ_TREE =
        "(TOP (S (NP (DT The) (NN man) (SBAR (WHNP-1 (WP who)) (S (NP-SBJ (-NONE- *T*-1)) (VP (VBD left))))) (VP (VBD returned)) (. .)))";

    private static final String EN_GAP_OBJ_TREE =
        "(TOP (S (NP (DT The) (NN book) (SBAR (WHNP-1 (WDT that)) (S (NP-SBJ (PRP she)) (VP (VBD read) (NP (-NONE- *T*-1)))))) (VP (VBD vanished)) (. .)))";

    private static final String ES_TREE_1 =
        "(TOP (S (SN (SPEC (DA0MS0 El)) (GRUP.NOM (NCMS000 coche) (S.A (GRUP.A (AQ0MS0 rojo))))) (GRUP.VERB (VMIP3S0 corre)) (fp .)))";

    private static final String ES_TREE_2 =
        "(TOP (S (SN (GRUP.NOM (NP0000P Madrid))) (GRUP.VERB (VSIP3S0 es)) (SA (GRUP.A (AQ0CS0 grande))) (fp .)))";

    private static final String ES_GAP_SUBJ_TREE =
        "(TOP (S (SN (GRUP.NOM (NCMS000 hombre) (S (SN (-NONE- *T*)) (GRUP.VERB (VMIS3S0 vino))))) (fp .)))";

    private static final String ES_GAP_OBJ_TREE =
        "(TOP (S (SN (GRUP.NOM (NCMS000 libro) (S (GRUP.VERB (VMIS3S0 leyo) (SN (-NONE- *T*)))))) (fp .)))";

    private static final String PRUNE_TREE =
        "(TOP (S (NP (NP (DT The) (NN cat))) (VP (VP (VBD sat))) (. .)))";

    private static final String POSSESSIVE_TREE =
        "(TOP (S (NP (DT The) (NN dog) (POS 's) (NN bone) (CC and) (NNS cats) (POS ') (NNS toys)) (VP (VBP are) (ADJP (JJ here))) (. .)))";

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("usage: java Vectors.java <data-dir> <out-dir>");
            System.exit(2);
        }
        Path dataDir = Path.of(args[0]);
        Path outDir = Path.of(args[1]);
        Files.createDirectories(outDir);

        opennlp.tools.parser.lang.en.HeadRules enRules;
        try (Reader r = Files.newBufferedReader(dataDir.resolve("en_head_rules"), StandardCharsets.UTF_8)) {
            enRules = new opennlp.tools.parser.lang.en.HeadRules(r);
        }
        AncoraSpanishHeadRules esRules;
        try (Reader r = Files.newBufferedReader(dataDir.resolve("es_head_rules_legacy"), StandardCharsets.UTF_8)) {
            esRules = new AncoraSpanishHeadRules(r);
        }

        List<String> testParseLines = new ArrayList<>();
        for (String line : Files.readAllLines(dataDir.resolve("test.parse"), StandardCharsets.UTF_8)) {
            if (!line.trim().isEmpty()) {
                testParseLines.add(line);
            }
        }
        testParseLines.add(PARSE_STRING);
        testParseLines.add(TOKEN_REPLACEMENT_TREE);

        Path outFile = outDir.resolve("parse_vectors.tsv");
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(outFile, StandardCharsets.US_ASCII))) {
            w.println("# generator=Vectors.java\tjava.version=" + System.getProperty("java.version")
                + "\tjava.vm.name=" + System.getProperty("java.vm.name")
                + "\tjava.vm.version=" + System.getProperty("java.vm.version"));
            w.println("# columns: kind\tinput\toutput");

            for (String raw : testParseLines) {
                Parse.useFunctionTags(false);
                Parse p = Parse.parseParse(raw);
                StringBuffer sb = new StringBuffer();
                p.show(sb);
                w.println("show\t" + escape(raw) + "\t" + escape(sb.toString()));
                w.println("text\t" + escape(raw) + "\t" + escape(p.getText()));

                try (ObjectStream<POSSample> posStream = new PosSampleStream(
                        new ParseSampleStream(ObjectStreamUtils.createObjectStream(raw)))) {
                    POSSample pos = posStream.read();
                    w.println("pos\t" + escape(raw) + "\t" + escape(formatPos(pos)));
                }

                try (ObjectStream<ChunkSample> chunkStream = new ChunkSampleStream(
                        new ParseSampleStream(ObjectStreamUtils.createObjectStream(raw)))) {
                    ChunkSample chunk = chunkStream.read();
                    w.println("chunk\t" + escape(raw) + "\t" + escape(formatChunk(chunk)));
                }

                p.updateHeads(enRules);
                w.println("en_heads\t" + escape(raw) + "\t" + escape(formatHeads(p)));

                Parse.useFunctionTags(true);
                Parse pf = Parse.parseParse(raw);
                StringBuffer sbf = new StringBuffer();
                pf.show(sbf);
                w.println("show_ftags\t" + escape(raw) + "\t" + escape(sbf.toString()));
                Parse.useFunctionTags(false);
            }

            for (String esRaw : List.of(ES_TREE_1, ES_TREE_2)) {
                Parse p = Parse.parseParse(esRaw);
                p.updateHeads(esRules);
                w.println("es_heads\t" + escape(esRaw) + "\t" + escape(formatHeads(p)));
            }

            for (String gapTree : List.of(EN_GAP_SUBJ_TREE, EN_GAP_OBJ_TREE)) {
                Parse p = Parse.parseParse(gapTree, enRules);
                StringBuffer sb = new StringBuffer();
                p.show(sb);
                w.println("en_gap\t" + escape(gapTree) + "\t" + escape(sb.toString()));
            }

            for (String gapTree : List.of(ES_GAP_SUBJ_TREE, ES_GAP_OBJ_TREE)) {
                Parse p = Parse.parseParse(gapTree, esRules);
                StringBuffer sb = new StringBuffer();
                p.show(sb);
                w.println("es_gap\t" + escape(gapTree) + "\t" + escape(sb.toString()));
            }

            Parse pruned = Parse.parseParse(PRUNE_TREE);
            Parse.pruneParse(pruned);
            StringBuffer prunedSb = new StringBuffer();
            pruned.show(prunedSb);
            w.println("prune\t" + escape(PRUNE_TREE) + "\t" + escape(prunedSb.toString()));

            Parse poss = Parse.parseParse(POSSESSIVE_TREE);
            Parse.fixPossesives(poss);
            StringBuffer possSb = new StringBuffer();
            poss.show(possSb);
            w.println("possessives\t" + escape(POSSESSIVE_TREE) + "\t" + escape(possSb.toString()));

            Parse named = Parse.parseParse(PARSE_STRING);
            Parse.addNames("location", new Span[] { new Span(7, 8) }, named.getTagNodes());
            StringBuffer namedSb = new StringBuffer();
            named.show(namedSb);
            w.println("add_names\t" + escape(PARSE_STRING) + "\t" + escape(namedSb.toString()));
        }
    }

    private static String formatPos(POSSample sample) {
        StringBuilder sb = new StringBuilder();
        String[] sentence = sample.getSentence();
        String[] tags = sample.getTags();
        for (int i = 0; i < sentence.length; i++) {
            if (i > 0) {
                sb.append(' ');
            }
            sb.append(sentence[i]).append('/').append(tags[i]);
        }
        return sb.toString();
    }

    private static String formatChunk(ChunkSample sample) {
        StringBuilder sb = new StringBuilder();
        String[] sentence = sample.getSentence();
        String[] tags = sample.getTags();
        String[] preds = sample.getPreds();
        for (int i = 0; i < sentence.length; i++) {
            if (i > 0) {
                sb.append(' ');
            }
            sb.append(sentence[i]).append('/').append(tags[i]).append('/').append(preds[i]);
        }
        return sb.toString();
    }

    private static String formatHeads(Parse root) {
        StringBuilder sb = new StringBuilder();
        appendHeads(root, sb);
        return sb.toString();
    }

    private static void appendHeads(Parse node, StringBuilder sb) {
        if (sb.length() > 0) {
            sb.append(';');
        }
        Parse head = node.getHead();
        Span span = node.getSpan();
        sb.append(node.getType())
            .append('[')
            .append(span.getStart())
            .append(',')
            .append(span.getEnd())
            .append("]->")
            .append(node.getHeadIndex())
            .append(':')
            .append(head.getType())
            .append(':')
            .append(head.getCoveredText());
        for (Parse child : node.getChildren()) {
            appendHeads(child, sb);
        }
    }

    private static String escape(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= 0x20 && c <= 0x7e && c != '\\') {
                sb.append(c);
            } else {
                sb.append(String.format("\\u%04x", (int) c));
            }
        }
        return sb.toString();
    }
}
