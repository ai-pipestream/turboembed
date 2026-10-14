// Writes the test vectors of the turbo-jvm crate from the running JVM.
//
// Run from the repository root with the JDK 21 source launcher:
//
//     java analysis/jvm/java/Vectors.java analysis/jvm/tests/vectors
//
// Each file starts with a header line naming this program and the JVM that
// ran it. Records are one per line with tab separated fields. A double is
// written as its 16 hex digit IEEE 754 bit pattern
// (Double.doubleToRawLongBits), a float widened to a double (which is
// exact) the same way, an integer in decimal, and a string with every
// character outside printable ASCII, and the backslash, written as a
// backslash, the letter u and four hex digits, so the files are ASCII.

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.Consumer;

public class Vectors {
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.err.println("usage: java Vectors.java <out-dir>");
            System.exit(2);
        }
        Path out = Path.of(args[0]);
        Files.createDirectories(out);
        write(out.resolve("strict_math_exp.tsv"), "input,StrictMath.exp(input)", w -> {
            for (double x : expInputs()) {
                w.println(hex(x) + "\t" + hex(StrictMath.exp(x)));
            }
        });
        write(out.resolve("strict_math_log.tsv"), "input,StrictMath.log(input)", w -> {
            for (double x : logInputs()) {
                w.println(hex(x) + "\t" + hex(StrictMath.log(x)));
            }
        });
        write(out.resolve("random.tsv"), "seed,method,argument,index,value", Vectors::random);
        write(out.resolve("compare.tsv"), "a,b,a.compareTo(b)", Vectors::compare);
        write(out.resolve("text.tsv"), "method,input,result", Vectors::text);
        write(out.resolve("whitespace.tsv"), "code point (hex) for which Character.isWhitespace is true", w -> {
            for (int cp = 0; cp <= Character.MAX_CODE_POINT; cp++) {
                if (Character.isWhitespace(cp)) {
                    w.println(Integer.toHexString(cp));
                }
            }
        });
    }

    static void write(Path path, String columns, Consumer<PrintWriter> body) throws IOException {
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.US_ASCII))) {
            w.println("# generator=Vectors.java\tjava.version=" + System.getProperty("java.version")
                    + "\tjava.vm.version=" + System.getProperty("java.vm.version")
                    + "\tjava.vm.name=" + System.getProperty("java.vm.name")
                    + "\tunicode=not exposed at run time"
                    + "\tcolumns=" + columns);
            body.accept(w);
        }
    }

    static String hex(double x) {
        return String.format("%016x", Double.doubleToRawLongBits(x));
    }

    /** The double with the given high and low 32-bit words. */
    static double words(int hi, int lo) {
        return Double.longBitsToDouble(((long) hi << 32) | (lo & 0xffff_ffffL));
    }

    /** x, its neighbours up to n ulps away on each side, and their negations. */
    static void around(Set<Double> s, double x, int n) {
        double up = x;
        double down = x;
        s.add(x);
        s.add(-x);
        for (int i = 0; i < n; i++) {
            up = Math.nextUp(up);
            down = Math.nextDown(down);
            s.add(up);
            s.add(down);
            s.add(-up);
            s.add(-down);
        }
    }

    /** Both sides of a threshold on the high word of |x|: the doubles with
     *  high word hi - 1, hi and hi + 1 and low words 0, 1 and all ones, and
     *  their negations. */
    static void highWord(Set<Double> s, int hi) {
        for (int h = hi - 1; h <= hi + 1; h++) {
            for (int lo : new int[] {0, 1, 0xffff_fffe, 0xffff_ffff}) {
                s.add(words(h, lo));
                s.add(-words(h, lo));
            }
        }
    }

    static void specials(Set<Double> s) {
        s.add(0.0);
        s.add(-0.0);
        s.add(Double.NaN);
        s.add(Double.longBitsToDouble(0xfff8_0000_0000_0000L));
        s.add(Double.longBitsToDouble(0x7ff0_0000_0000_0001L));
        s.add(Double.POSITIVE_INFINITY);
        s.add(Double.NEGATIVE_INFINITY);
        around(s, Double.MIN_VALUE, 3);
        around(s, Double.MIN_NORMAL, 3);
        around(s, Double.MAX_VALUE, 3);
        around(s, 1.0, 3);
    }

    static List<Double> expInputs() {
        Set<Double> s = new LinkedHashSet<>();
        specials(s);
        // The thresholds on the high word of |x| in e_exp.c: overflow and
        // underflow filter, non-finite, 0.5 ln2, 1.5 ln2 and 2^-28.
        for (int hi : new int[] {0x4086_2e42, 0x7ff0_0000, 0x3fd6_2e42, 0x3ff0_a2b2, 0x3e30_0000}) {
            highWord(s, hi);
        }
        // The overflow and underflow thresholds themselves.
        around(s, 0x1.62e42fefa39efp9, 8);
        around(s, -0x1.74910d52d3051p9, 8);
        // Where the scaled result turns subnormal (k = -1021 and below).
        for (int k = -1024; k <= -1018; k++) {
            around(s, k * Math.log(2), 4);
            around(s, (k + 0.5) * Math.log(2), 4);
        }
        // Dense sweep over the whole finite range of results.
        for (double x = -746.0; x <= 710.0; x += 0.5) {
            s.add(x);
        }
        for (double x = -2.0; x <= 2.0; x += 1.0 / 256) {
            s.add(x);
        }
        // Tiny arguments.
        for (int e = -1074; e <= -20; e += 3) {
            s.add(Math.scalb(1.0, e));
            s.add(-Math.scalb(1.0, e));
        }
        // Random bit patterns and random values in the range of results.
        Random r = new Random(257);
        for (int i = 0; i < 500; i++) {
            s.add(Double.longBitsToDouble(r.nextLong()));
        }
        for (int i = 0; i < 1500; i++) {
            s.add(r.nextDouble() * 1460.0 - 748.0);
        }
        for (int i = 0; i < 500; i++) {
            s.add(r.nextGaussian());
        }
        return new ArrayList<>(s);
    }

    static List<Double> logInputs() {
        Set<Double> s = new LinkedHashSet<>();
        specials(s);
        // Values near 1, where log takes its |f| < 2^-20 branch.
        around(s, 1.0, 64);
        for (int e = -60; e <= -1; e++) {
            for (double m : new double[] {1.0, 1.5, 0.75}) {
                s.add(1.0 + Math.scalb(m, e));
                s.add(1.0 - Math.scalb(m, e));
            }
        }
        // The thresholds on the fraction's high 20 bits in e_log.c: the
        // normalization at sqrt(2) (hx + 0x95f64 carrying into bit 20), the
        // polynomial's two forms (hx - 0x6147a and 0x6b851 - hx) and the
        // |f| < 2^-20 test ((2 + hx) & 0xfffff < 3), at several exponents.
        int[] fractions = {0x6a09d, 0x6a09e, 0x6a09f, 0x6147a, 0x6147b, 0x61479, 0x6b851, 0x6b850, 0x6b852,
            0xffffc, 0xffffd, 0xffffe, 0xfffff, 0x00000, 0x00001, 0x00002};
        for (int exp : new int[] {1, 2, 1000, 1021, 1022, 1023, 1024, 1025, 1100, 2046}) {
            for (int f : fractions) {
                for (int lo : new int[] {0, 1, 0xffff_ffff}) {
                    s.add(words((exp << 20) | f, lo));
                }
            }
        }
        // Powers of two and their neighbours, over every exponent.
        for (int e = -1074; e <= 1023; e++) {
            double p = Math.scalb(1.0, e);
            s.add(p);
            s.add(Math.nextUp(p));
            s.add(Math.nextDown(p));
        }
        // Subnormals.
        Random r = new Random(1257);
        for (int i = 0; i < 300; i++) {
            s.add(Double.longBitsToDouble(r.nextLong() & 0x000f_ffff_ffff_ffffL));
        }
        // A logarithmic sweep, probabilities in (0, 1] as maxent sees them,
        // and random bit patterns (of both signs).
        for (double x = 1e-12; x < 1e12; x *= 1.03) {
            s.add(x);
        }
        for (int i = 0; i < 1000; i++) {
            s.add(r.nextDouble());
        }
        for (int i = 0; i < 1000; i++) {
            s.add(Double.longBitsToDouble(r.nextLong() & Long.MAX_VALUE));
        }
        for (int i = 0; i < 100; i++) {
            s.add(Double.longBitsToDouble(r.nextLong() | Long.MIN_VALUE));
        }
        return new ArrayList<>(s);
    }

    static final long[] SEEDS = {0L, 1L, -1L, 42L, Long.MIN_VALUE, Long.MAX_VALUE, 0x5DEECE66DL, 123456789L,
        -987654321L};
    static final int[] BOUNDS = {1, 2, 3, 7, 10, 16, 100, 1000, 1 << 30, (1 << 30) + 1, 0x5555_5556, Integer.MAX_VALUE};
    static final int[] SHUFFLE_SIZES = {0, 1, 2, 4, 5, 20, 21, 100};

    static void random(PrintWriter w) {
        for (long seed : SEEDS) {
            String p = seed + "\t";
            Random r = new Random(seed);
            for (int i = 0; i < 64; i++) {
                w.println(p + "nextInt\t-\t" + i + "\t" + r.nextInt());
            }
            for (int bound : BOUNDS) {
                r = new Random(seed);
                for (int i = 0; i < 64; i++) {
                    w.println(p + "nextIntBounded\t" + bound + "\t" + i + "\t" + r.nextInt(bound));
                }
            }
            r = new Random(seed);
            for (int i = 0; i < 64; i++) {
                w.println(p + "nextLong\t-\t" + i + "\t" + r.nextLong());
            }
            r = new Random(seed);
            for (int i = 0; i < 64; i++) {
                w.println(p + "nextBoolean\t-\t" + i + "\t" + (r.nextBoolean() ? 1 : 0));
            }
            r = new Random(seed);
            for (int i = 0; i < 64; i++) {
                w.println(p + "nextFloat\t-\t" + i + "\t" + hex(r.nextFloat()));
            }
            r = new Random(seed);
            for (int i = 0; i < 64; i++) {
                w.println(p + "nextDouble\t-\t" + i + "\t" + hex(r.nextDouble()));
            }
            r = new Random(seed);
            for (int i = 0; i < 64; i++) {
                w.println(p + "nextGaussian\t-\t" + i + "\t" + hex(r.nextGaussian()));
            }
            // Calls of different methods on one generator, so a cached
            // Gaussian is returned after other draws.
            r = new Random(seed);
            for (int i = 0; i < 64; i++) {
                String v = switch (i % 4) {
                    case 0 -> hex(r.nextGaussian());
                    case 1 -> Integer.toString(r.nextInt(1000));
                    case 2 -> hex(r.nextGaussian());
                    default -> Long.toString(r.nextLong());
                };
                w.println(p + "mixed\t-\t" + i + "\t" + v);
            }
            for (int size : SHUFFLE_SIZES) {
                List<Integer> list = new ArrayList<>();
                for (int i = 0; i < size; i++) {
                    list.add(i);
                }
                Collections.shuffle(list, new Random(seed));
                StringBuilder b = new StringBuilder();
                for (int i = 0; i < size; i++) {
                    if (i > 0) {
                        b.append(',');
                    }
                    b.append(list.get(i));
                }
                w.println(p + "shuffle\t" + size + "\t0\t" + b);
            }
        }
    }

    static void compare(PrintWriter w) {
        String[] parts = {"", "a", "b", "z", "A", "é", "ÿ", "Ā", "中", "퟿", "",
            "Ａ", "�", "￿", "𐀀", "😀", "😁", "🤔",
            "􏿿"};
        List<String> strings = new ArrayList<>();
        for (String x : parts) {
            strings.add(x);
            strings.add(x + "a");
            strings.add("ab" + x);
            strings.add(x + x);
        }
        for (String a : strings) {
            for (String b : strings) {
                w.println(escape(a) + "\t" + escape(b) + "\t" + a.compareTo(b));
            }
        }
    }

    /**
     * Double.toString and Float.toString of values on both sides of every rule of their
     * specification (the plain and the scientific layout, the shortest decimal, the two digit
     * decimal that replaces a one digit one), and parseDouble, parseFloat, parseInt and
     * parseBoolean of texts on both sides of their grammars. A toString input is the value's
     * bit pattern and its result the string; a parse input is the string and its result the
     * value's bit pattern, the integer, the boolean, or NumberFormatException.
     */
    static void text(PrintWriter w) {
        Set<Double> doubles = new LinkedHashSet<>();
        specials(doubles);
        for (double x : new double[] {0.001, 1.0e7, 1.0, 10.0, 0.1, 0.5, 2.0 / 3.0, 100.0, 1234567.0, 12345678.0,
            Double.MIN_NORMAL, Double.MAX_VALUE, 1.0e23, 2.0e23, 9.007199254740992e15, 4503599627370496.5}) {
            around(doubles, x, 3);
        }
        for (int i = 1; i <= 300; i++) {
            doubles.add(i * Double.MIN_VALUE);
        }
        for (int n = -324; n <= 308; n++) {
            for (int k = 1; k <= 9; k++) {
                doubles.add(Double.parseDouble(k + "e" + n));
            }
        }
        Random r = new Random(20261014);
        for (int i = 0; i < 3000; i++) {
            double x = Double.longBitsToDouble(r.nextLong());
            if (!Double.isNaN(x)) {
                doubles.add(x);
            }
        }
        for (int i = 0; i < 1000; i++) {
            doubles.add(r.nextDouble() * Math.pow(10, r.nextInt(30) - 15));
            doubles.add((double) r.nextInt(100_000_000));
        }
        for (double x : doubles) {
            w.println("Double.toString\t" + hex(x) + "\t" + escape(Double.toString(x)));
        }
        Set<Float> floats = new LinkedHashSet<>();
        for (float x : new float[] {0.0f, -0.0f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
            Float.MIN_VALUE, Float.MIN_NORMAL, Float.MAX_VALUE, 0.001f, 1.0e7f, 1.0f, 0.1f, 16777216.0f,
            1234567.0f, 12345678.0f, 1.0f / 3.0f}) {
            floats.add(x);
            floats.add(Math.nextUp(x));
            floats.add(Math.nextDown(x));
            floats.add(-x);
        }
        for (int i = 1; i <= 300; i++) {
            floats.add(i * Float.MIN_VALUE);
        }
        for (int n = -45; n <= 38; n++) {
            for (int k = 1; k <= 9; k++) {
                floats.add(Float.parseFloat(k + "e" + n));
            }
        }
        for (int i = 0; i < 3000; i++) {
            float x = Float.intBitsToFloat(r.nextInt());
            if (!Float.isNaN(x)) {
                floats.add(x);
            }
        }
        for (int i = 0; i < 1000; i++) {
            floats.add((float) (r.nextFloat() * Math.pow(10, r.nextInt(16) - 8)));
            floats.add((float) r.nextInt(100_000_000));
        }
        for (float x : floats) {
            w.println("Float.toString\t" + hex(x) + "\t" + escape(Float.toString(x)));
        }
        String[] numbers = {"1.5", " 1.5 ", "\t1.5\n", "\u00a01.5", "1.5f", "1.5F", "1.5d", "1.5D", "1.5e", "1.5e+",
            "1.5e3", "1.5E-3", "+1.5", "-1.5", "--1.5", "+-1.5", ".5", "5.", ".", "-.5", "+.5e-3", "1e400", "-1e400",
            "1e-400", "0.1", "1.00000001", "3.4028236e38", "3.4028235e38", "1.7976931348623157e308",
            "1.7976931348623159e308", "4.9e-324", "2.4e-324", "2.5e-324", "123456789012345678901234567890",
            "0.30000000000000004", "Infinity", "-Infinity", "+Infinity", "infinity", "Inf", "NaN", "-NaN", "nan",
            "NaNf", "Infinityd", "1,5", "1_000", "1e5f", "1x", "", " ", "0x1p3", "0X1.8p1", "0x10", "1f5", "1.5.2",
            "1e5.0", "00012", "-0", "-0.0", "0e0", "1E", "e5", "9007199254740993", "1.5\u0000", "1.5 f",
            // Hexadecimal literals: the grammar, rounding to a double and to a float (ties, carries, the
            // ends of both ranges, more digits than either holds), zero significands and exponents over an int.
            "0x.8p1", "0x1.p0", "0xp1", "0x.p1", "0x1p", "0x1p+", "0x1", "0x1.8", "-0x1p-2", "+0x1P+2", "0x1p3f",
            "0x1p3D", "0x1p3x", "0x1p3 f", "0x1e5", "0x1p3.5", "0x1p0x3", "0x+1p3", "0x1p-0", "0x1p+0", "0x1p00003",
            " 0x1p3 ", "0x1p3\u0000", "0X1P3", "0x0001.0000p0", "0x.0001p16", "0xABCDEFp0", "0xabcdefp0",
            "0x1.1234567890abcdef1234567890p0", "0x1.00000000000008p0", "0x1.00000000000008000000000000000001p0",
            "0x1.00000000000018p0", "0x1.000001p0", "0x1.000003p0", "0x1.0000010000001p0", "0x1.fffffffp0",
            "0x1.fffffffffffffp1023", "0x1.fffffffffffff8p1023", "0x1p1024", "-0x1p1024", "0x1p-1022",
            "0x1.fffffffffffffp-1023", "0x1.fffffffffffff8p-1023", "0x1p-1074", "0x1p-1075", "0x1.8p-1075",
            "0x1.0000000000000000000000000001p-1075", "0x1p-1076", "-0x1p-1076", "0x1.fffffep127", "0x1.ffffffp127",
            "0x1p-126", "0x1.ffffffp-127", "0x1p-149", "0x1p-150", "0x1.8p-150", "0x1p-151", "0x0p99999999999",
            "-0x0.000p-99999999999", "0x1p2147483647", "0x1p2147483648", "0x1p-2147483648", "0x1p-2147483647",
            "0x1p99999999999", "0x1p\u0663", "0xg1p0", "0x1,8p1", "0x.000000000000000000001p0"};
        for (String s : numbers) {
            String d;
            try {
                d = hex(Double.parseDouble(s));
            } catch (NumberFormatException e) {
                d = "NumberFormatException";
            }
            w.println("Double.parseDouble\t" + escape(s) + "\t" + d);
            String f;
            try {
                f = hex(Float.parseFloat(s));
            } catch (NumberFormatException e) {
                f = "NumberFormatException";
            }
            w.println("Float.parseFloat\t" + escape(s) + "\t" + f);
        }
        String[] ints = {"5", "-5", "+5", "2147483647", "-2147483648", "2147483648", "-2147483649", "", "-", "+",
            " 5", "5 ", "1.0", "0x10", "00012", "1e3", "\u0661\u0662", "99999999999", "-0", "+0"};
        for (String s : ints) {
            String i;
            try {
                i = Integer.toString(Integer.parseInt(s));
            } catch (NumberFormatException e) {
                i = "NumberFormatException";
            }
            w.println("Integer.parseInt\t" + escape(s) + "\t" + i);
        }
        for (String s : new String[] {"true", "TRUE", "tRuE", "false", "yes", " true", "true ", "", "1", "t"}) {
            w.println("Boolean.parseBoolean\t" + escape(s) + "\t" + Boolean.parseBoolean(s));
        }
    }

    static String escape(String s) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= 0x20 && c < 0x7f && c != '\\') {
                b.append(c);
            } else {
                b.append(String.format("\\u%04x", (int) c));
            }
        }
        return b.toString();
    }
}
