//! Edge cases for the helpers, checked against values a JVM printed
//! (JDK 21, `StrictMath`, `java.util.Random`, `Collections.shuffle`). Each
//! literal has a comment naming the Java expression that produced it.

use std::cmp::Ordering;
use std::path::PathBuf;

use turbo_jvm::chars::{self, Half, Utf16Unit, Utf16Units};
use turbo_jvm::random::JavaRandom;
use turbo_jvm::strict_math;

/// A small linear congruential generator for test inputs.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// One character from each UTF-8 length, both sides of the surrogate gap
/// and the extremes.
const ALPHABET: [char; 16] = [
    'a',
    ' ',
    '\u{0}',
    '\u{7F}',
    '\u{80}',
    '\u{E9}',
    '\u{7FF}',
    '\u{800}',
    '\u{D7FF}',
    '\u{E000}',
    '\u{FFFF}',
    '\u{10000}',
    '\u{1F600}',
    '\u{1F601}',
    '\u{10FFFF}',
    'z',
];

fn random_string(rng: &mut Lcg, max_len: u32) -> String {
    (0..rng.below(max_len + 1)).map(|_| ALPHABET[rng.below(ALPHABET.len() as u32) as usize]).collect()
}

fn expected_units(text: &str) -> Vec<Utf16Unit> {
    let mut want = Vec::new();
    for (byte, c) in text.char_indices() {
        let mut buf = [0u16; 2];
        let units = c.encode_utf16(&mut buf);
        if units.len() == 1 {
            want.push(Utf16Unit { unit: units[0], byte: byte as u32, half: Half::Whole });
        } else {
            want.push(Utf16Unit { unit: units[0], byte: byte as u32, half: Half::High });
            want.push(Utf16Unit { unit: units[1], byte: byte as u32, half: Half::Low });
        }
    }
    want
}

fn check_units(text: &str) {
    let got: Vec<Utf16Unit> = Utf16Units::new(text).collect();
    assert_eq!(got, expected_units(text), "{text:?}");
    let std: Vec<u16> = text.encode_utf16().collect();
    assert_eq!(got.iter().map(|u| u.unit).collect::<Vec<_>>(), std, "{text:?}");
    assert_eq!(chars::utf16_len(text), std.len(), "{text:?}");
}

#[test]
fn units_of_single_characters() {
    for c in ['\u{0}', '\u{7F}', '\u{80}', '\u{7FF}', '\u{800}', '\u{FFFF}', '\u{10000}', '\u{10FFFF}'] {
        let s = c.to_string();
        check_units(&s);
        let units: Vec<Utf16Unit> = Utf16Units::new(&s).collect();
        if c.len_utf16() == 2 {
            assert_eq!(units.len(), 2);
            assert_eq!((units[0].half, units[1].half), (Half::High, Half::Low));
            assert!((0xD800..0xDC00).contains(&units[0].unit) && (0xDC00..0xE000).contains(&units[1].unit));
            assert_eq!((units[0].byte, units[1].byte), (0, 0));
        } else {
            assert_eq!(units.len(), 1);
            assert_eq!(units[0].half, Half::Whole);
        }
    }
}

#[test]
fn units_at_the_ends_of_a_string() {
    for s in [
        "\u{1F600}",
        "ab\u{1F600}",
        "\u{1F600}ab",
        "\u{1F600}\u{10000}",
        "a\u{1F600}\u{10FFFF}b",
        "\u{10FFFF}\u{E9}\u{10000}",
        "\u{800}\u{1F600}\u{7FF}\u{1F600}",
        "",
    ] {
        check_units(s);
    }
    let units: Vec<Utf16Unit> = Utf16Units::new("\u{1F600}\u{10000}").collect();
    let bytes: Vec<u32> = units.iter().map(|u| u.byte).collect();
    assert_eq!(bytes, [0, 0, 4, 4]);
    let halves: Vec<Half> = units.iter().map(|u| u.half).collect();
    assert_eq!(halves, [Half::High, Half::Low, Half::High, Half::Low]);
}

#[test]
fn utf16_len_matches_the_standard_encoder_for_random_strings() {
    let mut rng = Lcg(1);
    for _ in 0..500 {
        let s = random_string(&mut rng, 40);
        assert_eq!(chars::utf16_len(&s), s.encode_utf16().count(), "{s:?}");
        check_units(&s);
    }
}

fn naive_compare_to(a: &str, b: &str) -> i32 {
    let (x, y): (Vec<u16>, Vec<u16>) = (a.encode_utf16().collect(), b.encode_utf16().collect());
    for (p, q) in x.iter().zip(&y) {
        if p != q {
            return *p as i32 - *q as i32;
        }
    }
    x.len() as i32 - y.len() as i32
}

fn check_pair(a: &str, b: &str) {
    let got = chars::compare_to(a, b);
    assert_eq!(got, naive_compare_to(a, b), "{a:?} vs {b:?}");
    assert_eq!(chars::compare(a, b), got.cmp(&0), "{a:?} vs {b:?}");
    assert_eq!(chars::compare_to(b, a), -got, "{a:?} vs {b:?}");
    assert_eq!(chars::compare_to(a, a), 0, "{a:?}");
    assert_eq!(chars::compare(a, a), Ordering::Equal);
}

#[test]
fn compare_to_matches_a_unit_comparison_for_random_pairs() {
    let mut rng = Lcg(2);
    for _ in 0..2000 {
        let a = random_string(&mut rng, 12);
        let b = match rng.below(4) {
            // A prefix of a (possibly empty or cut mid pair by characters).
            0 => a.chars().take(rng.below(13) as usize).collect(),
            // Equal up to a point, then different.
            1 => format!("{}{}", a.chars().take(3).collect::<String>(), random_string(&mut rng, 6)),
            // Equal up to a surrogate pair, then different tails.
            2 => format!("{a}\u{1F600}{}", random_string(&mut rng, 3)),
            _ => random_string(&mut rng, 12),
        };
        check_pair(&a, &b);
        check_pair(&format!("{a}\u{1F600}x"), &format!("{a}\u{1F600}"));
        check_pair(&format!("{a}\u{1F600}"), &format!("{a}\u{1F601}"));
    }
    check_pair("", "");
    check_pair("", "\u{1F600}");
    check_pair("\u{1F600}", "\u{1F600}\u{1F600}");
    assert_eq!(chars::compare_to("", "\u{1F600}"), -2);
    assert_eq!(chars::compare_to("ab", "a"), 1);
}

#[test]
fn whitespace_agrees_with_the_jvm_for_every_code_point() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/whitespace.tsv");
    let text = std::fs::read_to_string(&path).unwrap();
    let mut set = std::collections::HashSet::new();
    for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
        set.insert(u32::from_str_radix(line.trim(), 16).unwrap());
    }
    assert_eq!(set.len(), 25);
    let mut checked = 0u32;
    for cp in 0..=0x10FFFFu32 {
        if let Some(c) = char::from_u32(cp) {
            checked += 1;
            assert_eq!(chars::is_java_whitespace(c), set.contains(&cp), "U+{cp:04X}");
        }
    }
    assert_eq!(checked, 0x10FFFF + 1 - 0x800);
}

// java.util.Random

#[test]
fn random_matches_values_a_jvm_printed() {
    // Random r = new Random(123456789L); then the calls below in order.
    let mut r = JavaRandom::new(123456789);
    let ints = [-1442945365, -1016548095, 1962592967, 1094656688, 1677212580]; // 5 x r.nextInt()
    for want in ints {
        assert_eq!(r.next_int(), want);
    }
    assert_eq!(r.next_int_bounded(7), 0); // r.nextInt(7)
    assert_eq!(r.next_int_bounded(1 << 20), 936736); // r.nextInt(1 << 20)
    assert_eq!(r.next_long(), 7848905049914246339); // r.nextLong()
    assert_eq!(r.next_double().to_bits(), 0x3fdb958f85c0710a); // Double.doubleToRawLongBits(r.nextDouble())
    assert_eq!(r.next_float().to_bits(), 0x3efb0d96); // Float.floatToRawIntBits(r.nextFloat())
    assert!(!r.next_boolean()); // r.nextBoolean()
    assert_eq!(r.next_gaussian().to_bits(), 0xbfd6b0bceccdc565); // Double.doubleToRawLongBits(r.nextGaussian())
    assert_eq!(r.next_gaussian().to_bits(), 0xbf904d162011e749); // the following r.nextGaussian(), the cached value
}

#[test]
fn shuffle_matches_collections_shuffle() {
    // Collections.shuffle(list of Integer 0..n, new Random(2024)).
    let want10 = [3, 7, 2, 9, 6, 8, 4, 1, 5, 0];
    let mut items: Vec<u32> = (0..10).collect();
    JavaRandom::new(2024).shuffle(&mut items);
    assert_eq!(items, want10);

    let want1000: [u32; 1000] = [
        612, 961, 599, 129, 640, 701, 420, 830, 551, 805, 711, 196, 405, 192, 34, 203, 104, 429, 323, 598, 940, 421,
        90, 356, 494, 495, 227, 607, 317, 295, 401, 720, 68, 653, 877, 302, 818, 81, 63, 658, 216, 270, 643, 135, 788,
        277, 23, 722, 284, 11, 916, 536, 207, 735, 838, 816, 387, 484, 679, 74, 808, 708, 965, 584, 700, 903, 833, 343,
        458, 614, 152, 760, 437, 923, 162, 473, 38, 282, 368, 9, 341, 930, 412, 417, 371, 188, 326, 362, 197, 432, 222,
        949, 304, 96, 610, 414, 839, 926, 19, 758, 902, 899, 882, 693, 699, 717, 209, 841, 65, 441, 355, 907, 836, 301,
        714, 454, 120, 564, 869, 943, 743, 113, 359, 119, 957, 116, 419, 276, 272, 618, 286, 298, 849, 250, 825, 496,
        790, 732, 698, 978, 408, 948, 339, 713, 776, 886, 49, 774, 620, 793, 360, 307, 906, 97, 248, 182, 500, 13, 45,
        327, 533, 667, 459, 826, 813, 366, 876, 16, 363, 139, 491, 623, 231, 644, 252, 477, 938, 316, 186, 331, 595,
        358, 866, 147, 115, 374, 328, 430, 134, 908, 385, 900, 233, 649, 608, 543, 445, 194, 798, 195, 504, 223, 874,
        590, 180, 185, 922, 72, 47, 204, 518, 591, 690, 247, 253, 41, 381, 220, 110, 111, 606, 567, 425, 987, 278, 726,
        303, 880, 433, 782, 319, 493, 637, 489, 350, 42, 898, 375, 266, 884, 211, 488, 824, 390, 126, 954, 457, 460,
        670, 850, 109, 792, 404, 799, 707, 172, 512, 118, 687, 664, 605, 281, 661, 153, 648, 547, 684, 449, 107, 148,
        804, 750, 244, 384, 873, 557, 7, 357, 29, 821, 15, 734, 365, 274, 389, 190, 391, 609, 575, 576, 958, 962, 617,
        94, 960, 77, 511, 562, 578, 254, 461, 755, 12, 887, 834, 351, 934, 85, 910, 594, 981, 666, 239, 299, 964, 499,
        946, 476, 165, 20, 771, 914, 544, 465, 508, 260, 440, 580, 847, 187, 673, 785, 565, 559, 628, 971, 613, 480,
        589, 988, 105, 492, 173, 287, 347, 692, 501, 718, 58, 603, 626, 308, 893, 322, 210, 447, 611, 292, 28, 553,
        522, 426, 574, 791, 579, 521, 345, 434, 998, 855, 456, 744, 787, 349, 483, 428, 170, 320, 80, 651, 364, 262,
        535, 678, 240, 531, 532, 273, 478, 545, 177, 245, 616, 702, 62, 448, 214, 592, 39, 763, 392, 520, 573, 558, 78,
        1, 781, 650, 17, 888, 146, 638, 863, 742, 695, 99, 797, 230, 294, 586, 140, 383, 136, 809, 184, 335, 348, 853,
        694, 563, 996, 8, 234, 795, 36, 864, 660, 883, 178, 149, 894, 997, 924, 840, 523, 811, 269, 427, 912, 800, 596,
        474, 905, 33, 513, 976, 587, 232, 868, 481, 89, 550, 921, 947, 901, 142, 647, 444, 112, 73, 485, 600, 243, 928,
        646, 731, 615, 572, 602, 509, 741, 966, 974, 424, 463, 342, 372, 221, 780, 891, 951, 468, 31, 892, 168, 860,
        851, 93, 352, 740, 329, 141, 4, 619, 175, 98, 950, 927, 479, 925, 854, 506, 145, 929, 817, 861, 959, 704, 138,
        321, 757, 665, 896, 583, 215, 258, 517, 325, 857, 51, 992, 66, 858, 314, 629, 917, 290, 159, 125, 867, 991,
        904, 659, 309, 784, 24, 515, 827, 106, 539, 193, 878, 963, 3, 53, 61, 803, 942, 979, 123, 283, 875, 402, 438,
        226, 601, 367, 370, 117, 416, 464, 455, 497, 32, 102, 984, 777, 577, 859, 524, 986, 675, 154, 114, 955, 241,
        529, 75, 842, 91, 828, 280, 729, 378, 881, 872, 772, 60, 259, 769, 174, 835, 634, 296, 14, 71, 18, 54, 334, 43,
        516, 333, 581, 338, 510, 312, 953, 980, 332, 410, 37, 418, 973, 361, 982, 631, 340, 820, 346, 829, 300, 452,
        354, 765, 879, 680, 723, 212, 297, 388, 748, 801, 318, 143, 730, 636, 862, 865, 642, 749, 525, 228, 475, 55,
        137, 490, 257, 768, 393, 588, 852, 467, 395, 889, 549, 271, 144, 528, 242, 219, 218, 158, 443, 762, 990, 845,
        970, 752, 462, 256, 95, 555, 890, 435, 167, 585, 100, 502, 952, 235, 721, 672, 669, 213, 911, 128, 486, 127,
        48, 403, 236, 249, 396, 778, 291, 677, 306, 919, 844, 406, 202, 27, 751, 871, 310, 373, 569, 466, 189, 377,
        413, 819, 198, 995, 738, 279, 380, 759, 597, 206, 681, 733, 968, 753, 344, 936, 10, 122, 225, 630, 446, 67,
        794, 275, 469, 330, 703, 64, 635, 386, 133, 913, 668, 985, 709, 689, 26, 87, 983, 22, 627, 200, 255, 663, 754,
        848, 832, 88, 263, 46, 761, 696, 69, 84, 739, 715, 92, 822, 163, 534, 568, 56, 685, 103, 199, 25, 482, 622,
        527, 691, 337, 725, 837, 571, 409, 376, 641, 654, 267, 289, 767, 44, 933, 519, 471, 727, 171, 439, 944, 530,
        686, 969, 710, 570, 548, 505, 909, 21, 705, 2, 451, 398, 130, 937, 807, 656, 176, 264, 764, 369, 967, 131, 542,
        151, 76, 379, 161, 662, 311, 989, 224, 975, 682, 52, 191, 786, 472, 706, 561, 918, 815, 353, 728, 994, 560,
        931, 251, 470, 57, 541, 737, 397, 415, 59, 205, 779, 164, 773, 941, 746, 552, 814, 237, 169, 856, 422, 582,
        712, 724, 977, 697, 30, 633, 831, 183, 538, 870, 537, 604, 313, 514, 268, 82, 382, 261, 132, 160, 775, 999,
        399, 86, 932, 812, 208, 121, 566, 394, 745, 810, 40, 155, 556, 766, 915, 411, 688, 796, 945, 719, 593, 156,
        802, 639, 897, 645, 217, 683, 315, 625, 101, 956, 806, 843, 229, 150, 431, 716, 246, 305, 632, 783, 436, 5,
        453, 624, 885, 181, 336, 487, 789, 6, 285, 108, 288, 920, 657, 442, 179, 676, 124, 674, 770, 846, 166, 823,
        157, 450, 526, 70, 265, 507, 747, 201, 0, 671, 324, 736, 400, 972, 407, 498, 993, 756, 423, 652, 895, 939, 238,
        35, 621, 655, 546, 83, 293, 79, 935, 503, 540, 554, 50,
    ];
    let mut items: Vec<u32> = (0..1000).collect();
    JavaRandom::new(2024).shuffle(&mut items);
    assert_eq!(items, want1000);
}

#[test]
fn bound_i32_max_matches_the_jvm() {
    // Random r = new Random(5); three r.nextInt(Integer.MAX_VALUE), then r.nextInt().
    let mut r = JavaRandom::new(5);
    assert_eq!(r.next_int_bounded(i32::MAX), 1568779487);
    assert_eq!(r.next_int_bounded(i32::MAX), 379250092);
    assert_eq!(r.next_int_bounded(i32::MAX), 189533474);
    assert_eq!(r.next_int(), -1667228448);
}

#[test]
fn bound_one_returns_zero_and_consumes_one_draw() {
    // new Random(5): r.nextInt(1) is 0, and r.nextInt() after it is 758500184.
    let mut r = JavaRandom::new(5);
    assert_eq!(r.next_int_bounded(1), 0);
    assert_eq!(r.next_int(), 758500184);
    let mut a = JavaRandom::new(77);
    let mut b = a.clone();
    for _ in 0..100 {
        assert_eq!(a.next_int_bounded(1), 0);
        b.next(31);
        assert_eq!(a.next_long(), b.next_long());
    }
}

#[test]
#[should_panic]
fn bound_zero_panics() {
    JavaRandom::new(1).next_int_bounded(0);
}

#[test]
#[should_panic]
fn negative_bound_panics() {
    JavaRandom::new(1).next_int_bounded(-5);
}

// strict_math

fn same(want: u64, got: f64) -> bool {
    let want = f64::from_bits(want);
    if want.is_nan() { got.is_nan() } else { want.to_bits() == got.to_bits() }
}

/// (input bits, StrictMath.exp(input) bits)
const EXP: &[(u64, u64)] = &[
    (0xc034000000000000, 0x3e21b48655f37267), // StrictMath.exp(-20.0)
    (0xc033000000000000, 0x3e381056ff2c5772), // StrictMath.exp(-19.0)
    (0xc032000000000000, 0x3e505a628c699fa1), // StrictMath.exp(-18.0)
    (0xc031000000000000, 0x3e6639e3175a689d), // StrictMath.exp(-17.0)
    (0xc030000000000000, 0x3e7e355bbaee85cb), // StrictMath.exp(-16.0)
    (0xc02e000000000000, 0x3e94875ca227ec38), // StrictMath.exp(-15.0)
    (0xc02c000000000000, 0x3eabe6c6fdb01612), // StrictMath.exp(-14.0)
    (0xc02a000000000000, 0x3ec2f6053b981d98), // StrictMath.exp(-13.0)
    (0xc028000000000000, 0x3ed9c54c3b43bc8b), // StrictMath.exp(-12.0)
    (0xc026000000000000, 0x3ef18354238f6764), // StrictMath.exp(-11.0)
    (0xc024000000000000, 0x3f07cd79b5647c9a), // StrictMath.exp(-10.0)
    (0xc022000000000000, 0x3f202cf22526545a), // StrictMath.exp(-9.0)
    (0xc020000000000000, 0x3f35fc21041027ad), // StrictMath.exp(-8.0)
    (0xc01c000000000000, 0x3f4de16b9c24a98f), // StrictMath.exp(-7.0)
    (0xc018000000000000, 0x3f644e51f113d4d6), // StrictMath.exp(-6.0)
    (0xc014000000000000, 0x3f7b993fe00d5376), // StrictMath.exp(-5.0)
    (0xc010000000000000, 0x3f92c155b8213cf4), // StrictMath.exp(-4.0)
    (0xc008000000000000, 0x3fa97db0ccceb0af), // StrictMath.exp(-3.0)
    (0xc000000000000000, 0x3fc152aaa3bf81cc), // StrictMath.exp(-2.0)
    (0xbff0000000000000, 0x3fd78b56362cef38), // StrictMath.exp(-1.0)
    (0x0000000000000000, 0x3ff0000000000000), // StrictMath.exp(0.0)
    (0x3ff0000000000000, 0x4005bf0a8b14576a), // StrictMath.exp(1.0)
    (0x4000000000000000, 0x401d8e64b8d4ddae), // StrictMath.exp(2.0)
    (0x4008000000000000, 0x403415e5bf6fb106), // StrictMath.exp(3.0)
    (0x4010000000000000, 0x404b4c902e273a58), // StrictMath.exp(4.0)
    (0x4014000000000000, 0x40628d389970338f), // StrictMath.exp(5.0)
    (0x4018000000000000, 0x407936dc5690c08f), // StrictMath.exp(6.0)
    (0x401c000000000000, 0x409122885aaeddaa), // StrictMath.exp(7.0)
    (0x4020000000000000, 0x40a749ea7d470c6e), // StrictMath.exp(8.0)
    (0x4022000000000000, 0x40bfa7157c470f82), // StrictMath.exp(9.0)
    (0x4024000000000000, 0x40d5829dcf950560), // StrictMath.exp(10.0)
    (0x4026000000000000, 0x40ed3c4488ee4f7f), // StrictMath.exp(11.0)
    (0x4028000000000000, 0x4103de1654d37c9a), // StrictMath.exp(12.0)
    (0x402a000000000000, 0x411b00b5916ac955), // StrictMath.exp(13.0)
    (0x402c000000000000, 0x413259ac48bf05d7), // StrictMath.exp(14.0)
    (0x402e000000000000, 0x4148f0ccafad2a87), // StrictMath.exp(15.0)
    (0x4030000000000000, 0x4160f2ebd0a80020), // StrictMath.exp(16.0)
    (0x4031000000000000, 0x417709348c0ea4f9), // StrictMath.exp(17.0)
    (0x4032000000000000, 0x418f4f22091940bd), // StrictMath.exp(18.0)
    (0x4033000000000000, 0x41a546d8f9ed26e2), // StrictMath.exp(19.0)
    (0x4034000000000000, 0x41bceb088b68e804), // StrictMath.exp(20.0)
    (0x3ddb7cdfd9d7bdbb, 0x3ff000000006df38), // StrictMath.exp(1.0E-10)
    (0x01a56e1fc2f8f359, 0x3ff0000000000000), // StrictMath.exp(1.0E-300)
    (0x0000000000000001, 0x3ff0000000000000), // StrictMath.exp(4.9E-324)
    (0x7fefffffffffffff, 0x7ff0000000000000), // StrictMath.exp(1.7976931348623157E308)
    (0x0010000000000000, 0x3ff0000000000000), // StrictMath.exp(2.2250738585072014E-308)
    (0x40862e42fefa39ef, 0x7fefffffffffff2a), // StrictMath.exp(709.782712893384)
    (0x40862e42fefa39f0, 0x7ff0000000000000), // StrictMath.exp(709.7827128933841)
    (0xc0874910d52d3051, 0x0000000000000001), // StrictMath.exp(-745.1332191019411)
    (0xc0874910d52d3052, 0x0000000000000000), // StrictMath.exp(-745.1332191019412)
    (0x3fe0000000000000, 0x3ffa61298e1e069c), // StrictMath.exp(0.5)
    (0xbfe0000000000000, 0x3fe368b2fc6f960a), // StrictMath.exp(-0.5)
    (0x3fd62e42fefa39ee, 0x3ff6a09e667f3bcc), // StrictMath.exp(0.3465735902799726)
    (0xbfd62e42fefa39f0, 0x3fe6a09e667f3bcc), // StrictMath.exp(-0.3465735902799727)
    (0x3ff0a2b23f3bab73, 0x4006a09e667f3bcc), // StrictMath.exp(1.0397207708399179)
    (0x3ee4f8b588e368f1, 0x3ff0000a7c5e340e), // StrictMath.exp(1.0E-5)
    (0xbee4f8b588e368f1, 0x3fefffeb07515653), // StrictMath.exp(-1.0E-5)
    (0x4059000000000000, 0x48f3494a9b171bf5), // StrictMath.exp(100.0)
    (0xc059000000000000, 0x36ea8c1f14e2af5d), // StrictMath.exp(-100.0)
    (0x4085e00000000000, 0x7f0d945df4f8ec8e), // StrictMath.exp(700.0)
    (0xc085e00000000000, 0x00d14f2b0fb9307f), // StrictMath.exp(-700.0)
    (0xc086232bdd7abcd2, 0x001000000000007c), // StrictMath.exp(-708.3964185322641)
    (0xc086233333333333, 0x000ff15b469edf89), // StrictMath.exp(-708.4)
    (0xc087200000000000, 0x0000000000000055), // StrictMath.exp(-740.0)
    (0x400921fb54442d18, 0x403724046eb09339), // StrictMath.exp(3.141592653589793)
    (0xc005bf0a8b145769, 0x3fb0e49787e20548), // StrictMath.exp(-2.718281828459045)
    (0x3fb999999999999a, 0x3ff1aec7b35a00d4), // StrictMath.exp(0.1)
    (0x3fc999999999999a, 0x3ff38add9e58ac8d), // StrictMath.exp(0.2)
    (0x3fd3333333333334, 0x3ff599058c8c1a96), // StrictMath.exp(0.30000000000000004)
];

/// (input bits, StrictMath.log(input) bits)
const LOG: &[(u64, u64)] = &[
    (0x4034000000000000, 0x4007f7427b73e391), // StrictMath.log(20.0)
    (0x4033000000000000, 0x40078e360604b32c), // StrictMath.log(19.0)
    (0x4032000000000000, 0x40071f7b3a6b9186), // StrictMath.log(18.0)
    (0x4031000000000000, 0x4006aa6bc1fa7f7a), // StrictMath.log(17.0)
    (0x4030000000000000, 0x40062e42fefa39ef), // StrictMath.log(16.0)
    (0x402e000000000000, 0x4005aa16394d481f), // StrictMath.log(15.0)
    (0x402c000000000000, 0x40051cca16d7bba7), // StrictMath.log(14.0)
    (0x402a000000000000, 0x400485042b318c51), // StrictMath.log(13.0)
    (0x4028000000000000, 0x4003e116bcd39e7d), // StrictMath.log(12.0)
    (0x4026000000000000, 0x40032ee3b77f374c), // StrictMath.log(11.0)
    (0x4024000000000000, 0x40026bb1bbb55516), // StrictMath.log(10.0)
    (0x4022000000000000, 0x400193ea7aad030b), // StrictMath.log(9.0)
    (0x4020000000000000, 0x4000a2b23f3bab73), // StrictMath.log(8.0)
    (0x401c000000000000, 0x3fff2272ae325a57), // StrictMath.log(7.0)
    (0x4018000000000000, 0x3ffcab0bfa2a2002), // StrictMath.log(6.0)
    (0x4014000000000000, 0x3ff9c041f7ed8d33), // StrictMath.log(5.0)
    (0x4010000000000000, 0x3ff62e42fefa39ef), // StrictMath.log(4.0)
    (0x4008000000000000, 0x3ff193ea7aad030a), // StrictMath.log(3.0)
    (0x4000000000000000, 0x3fe62e42fefa39ef), // StrictMath.log(2.0)
    (0x3ff0000000000000, 0x0000000000000000), // StrictMath.log(1.0)
    (0x3ff0000000000000, 0x0000000000000000), // StrictMath.log(1.0)
    (0x4000000000000000, 0x3fe62e42fefa39ef), // StrictMath.log(2.0)
    (0x4008000000000000, 0x3ff193ea7aad030a), // StrictMath.log(3.0)
    (0x4010000000000000, 0x3ff62e42fefa39ef), // StrictMath.log(4.0)
    (0x4014000000000000, 0x3ff9c041f7ed8d33), // StrictMath.log(5.0)
    (0x4018000000000000, 0x3ffcab0bfa2a2002), // StrictMath.log(6.0)
    (0x401c000000000000, 0x3fff2272ae325a57), // StrictMath.log(7.0)
    (0x4020000000000000, 0x4000a2b23f3bab73), // StrictMath.log(8.0)
    (0x4022000000000000, 0x400193ea7aad030b), // StrictMath.log(9.0)
    (0x4024000000000000, 0x40026bb1bbb55516), // StrictMath.log(10.0)
    (0x4026000000000000, 0x40032ee3b77f374c), // StrictMath.log(11.0)
    (0x4028000000000000, 0x4003e116bcd39e7d), // StrictMath.log(12.0)
    (0x402a000000000000, 0x400485042b318c51), // StrictMath.log(13.0)
    (0x402c000000000000, 0x40051cca16d7bba7), // StrictMath.log(14.0)
    (0x402e000000000000, 0x4005aa16394d481f), // StrictMath.log(15.0)
    (0x4030000000000000, 0x40062e42fefa39ef), // StrictMath.log(16.0)
    (0x4031000000000000, 0x4006aa6bc1fa7f7a), // StrictMath.log(17.0)
    (0x4032000000000000, 0x40071f7b3a6b9186), // StrictMath.log(18.0)
    (0x4033000000000000, 0x40078e360604b32c), // StrictMath.log(19.0)
    (0x4034000000000000, 0x4007f7427b73e391), // StrictMath.log(20.0)
    (0x3fefffffca501acb, 0xbe7ad7f2b1049b9f), // StrictMath.log(0.9999999)
    (0x3ff000001ad7f29b, 0x3e7ad7f2847b6492), // StrictMath.log(1.0000001)
    (0x4000000000000000, 0x3fe62e42fefa39ef), // StrictMath.log(2.0)
    (0x3fe0000000000000, 0xbfe62e42fefa39ef), // StrictMath.log(0.5)
    (0x000012688b70e62b, 0xc0864e69394d9508), // StrictMath.log(1.0E-310)
    (0x7fefffffffffffff, 0x40862e42fefa39ef), // StrictMath.log(1.7976931348623157E308)
    (0x0000000000000001, 0xc0874385446d71c3), // StrictMath.log(4.9E-324)
    (0x0010000000000000, 0xc086232bdd7abcd2), // StrictMath.log(2.2250738585072014E-308)
    (0x3ddb7cdfd9d7bdbb, 0xc037069e2aa2aa5b), // StrictMath.log(1.0E-10)
    (0x01a56e1fc2f8f359, 0xc085963447f87fb5), // StrictMath.log(1.0E-300)
    (0x3ff0000000000000, 0x0000000000000000), // StrictMath.log(1.0)
    (0x4024000000000000, 0x40026bb1bbb55516), // StrictMath.log(10.0)
    (0x3fb999999999999a, 0xc0026bb1bbb55515), // StrictMath.log(0.1)
    (0x4005bf0a8b145769, 0x3ff0000000000000), // StrictMath.log(2.718281828459045)
    (0x0000000000000000, 0xfff0000000000000), // StrictMath.log(0.0)
    (0x8000000000000000, 0xfff0000000000000), // StrictMath.log(-0.0)
    (0xbff0000000000000, 0xfff8000000000000), // StrictMath.log(-1.0)
];

#[test]
fn exp_matches_strict_math() {
    for &(x, want) in EXP {
        let got = strict_math::exp(f64::from_bits(x));
        assert!(same(want, got), "exp({:e}): want {want:016x}, got {:016x}", f64::from_bits(x), got.to_bits());
    }
}

#[test]
fn log_matches_strict_math() {
    for &(x, want) in LOG {
        let got = strict_math::log(f64::from_bits(x));
        assert!(same(want, got), "log({:e}): want {want:016x}, got {:016x}", f64::from_bits(x), got.to_bits());
    }
}

/// The next larger finite double.
fn next_up(x: f64) -> f64 {
    if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

#[test]
fn exp_is_monotone() {
    let mut x = -40.0f64;
    let mut prev = strict_math::exp(x);
    for _ in 0..4000 {
        x += 0.02;
        let y = strict_math::exp(x);
        assert!(prev <= y, "exp({}) = {prev:e} > exp({x}) = {y:e}", x - 0.02);
        let tiny = strict_math::exp(next_up(x));
        assert!(y <= tiny, "exp({x}) = {y:e} > exp(next float) = {tiny:e}");
        prev = y;
    }
}

#[test]
fn log_is_monotone() {
    let mut x = 1e-6f64;
    let mut prev = strict_math::log(x);
    for _ in 0..4000 {
        x *= 1.005;
        let y = strict_math::log(x);
        assert!(prev <= y, "log({}) = {prev} > log({x}) = {y}", x / 1.005);
        let tiny = strict_math::log(next_up(x));
        assert!(y <= tiny, "log({x}) = {y} > log(next float) = {tiny}");
        prev = y;
    }
}
