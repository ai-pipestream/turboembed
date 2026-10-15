//! `CryptoPiiExtractorTest` and `CryptoAddressTest`. The legacy Bitcoin
//! addresses are made up: version bytes over hashes of fixed strings.

mod support;

use support::*;
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

const LEGACY: &str = "1Pjk7c5ipz7ZpbVT3VKwcFNGeqgBFpbQdw";
const LEGACY_2: &str = "1PjX57Cc5aVJ8SFEt3xzf8d46UQnfk59LC";
const LEGACY_3: &str = "1MT8G2xiEVHzykbM3aEDfoSnQgUuSuM27D";
const P2SH: &str = "3APADkvHwo6aV4ZRCY7H7RZTcdvMxbYyNR";
const P2SH_2: &str = "3D1xjDadfryB2aRssyDoeQY27F422rjZMN";
const TESTNET: &str = "mxHzmMbgSqnUMGQFjeTGww1GgSHRcUb1Y4";
/// All ones but the checksum: a zero hash.
const ZERO_LEGACY: &str = "11111111111111111111BZbvjr";
const SEGWIT: &str = "bc1pqqqqqcnjqs";
const ETH: &str = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed";

fn crypto() -> PiiDetector {
    pack(Pack::Crypto)
}

fn btc(text: &str) -> bool {
    kinds(&crypto(), text).contains(&"btc-address")
}

fn eth(text: &str) -> bool {
    kinds(&crypto(), text).contains(&"eth-address")
}

#[test]
fn legacy_addresses() {
    for text in [LEGACY, LEGACY_2, LEGACY_3, P2SH, P2SH_2, ZERO_LEGACY, "16L5yRNPTuciSgXGHqYwn9N6NeoKqopAu"] {
        assert_found(&crypto(), text, &[(0, text.len() as u32, "btc-address", text)]);
    }
}

#[test]
fn legacy_addresses_with_a_broken_checksum() {
    for text in [&LEGACY[..33], &LEGACY_2[..33], &P2SH[..33], &LEGACY_3[..33]] {
        for last in ['A', 'b', 'z'] {
            let mut s = text.to_owned();
            s.push(last);
            if s != LEGACY && s != LEGACY_2 && s != P2SH && s != LEGACY_3 {
                assert_none(&crypto(), &s);
            }
        }
    }
}

#[test]
fn legacy_near_misses() {
    for text in [
        LEGACY[..30].to_owned(),
        format!("{LEGACY}XX"),
        format!("2{}", &LEGACY[1..]),
        format!("{}0{}", &LEGACY[..10], &LEGACY[11..]),
        format!("{}I{}", &LEGACY[..10], &LEGACY[11..]),
        format!("x{LEGACY}"),
        TESTNET.to_owned(),
    ] {
        assert!(!btc(&text), "{text}");
    }
}

#[test]
fn segwit_addresses() {
    for text in [
        "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
        "bc1qrp33g0q5c5txsp9arysrx4k6zdkfs4nce4xj0gdcccefvpysxf3qccfmv3",
        "bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0",
    ] {
        assert_eq!(normalized(&crypto(), text), [text]);
    }
    let upper = "BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4";
    assert_eq!(normalized(&crypto(), upper), [upper.to_ascii_lowercase()]);
}

#[test]
fn segwit_near_misses() {
    for text in [
        "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5",
        "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t",
        "bc1Qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
        "bc1qqqqsyydq4q",
        "bc1pqqq3g00lg3",
        "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx",
        "bc1q",
        "bc1",
        "bcqw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
        "xbc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
    ] {
        assert!(!btc(text), "{text}");
    }
}

#[test]
fn witness_versions() {
    for text in [
        "bc1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq9e75rs",
        "bc1pqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqmmente",
        "bc1zqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqxsfwuy",
        "bc1rqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqdw7930",
        "bc1yqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq4xqamh",
        "bc19qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq7chkku",
        "bc1xqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqrn8tpp",
        "bc18qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqgdsqv2",
        "bc1gqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq6rjj4c",
        "bc1fqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3a9ecn",
        "bc12qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqvk4y0w",
        "bc1tqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq8gz0z9",
        "bc1vqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqlquhga",
        "bc1dqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq57tu9k",
        "bc1wqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqf4mpjt",
        "bc10qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqztv2lq",
        "bc1sqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqyflvfx",
    ] {
        assert_eq!(normalized(&crypto(), text), [text]);
        assert_eq!(normalized(&crypto(), &text.to_ascii_uppercase()), [text]);
    }
}

#[test]
fn payload_limits() {
    for text in [
        ZERO_LEGACY,
        "1111111111111111111114oLvT2",
        SEGWIT,
        "bc1sqqqqkfw08p",
        "bc1pqqqqqkzsz3c",
        "bc1pqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq2f6xj9",
        "bc1sqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqrue65v",
        "bc1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqthqst8",
    ] {
        assert_found(&crypto(), text, &[(0, text.len() as u32, "btc-address", text)]);
    }
}

#[test]
fn invalid_witness_encodings() {
    for text in [
        "bc1qqqqq399cqn",
        "bc1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqmql8k8",
        "bc1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqj9pecr",
        "bc1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqpqy20t",
        "bc1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq88p3kr",
        "bc1pqqlppvpg",
        "bc1pqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqepcyyg",
        "bc13qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq0hg8yd",
        "bc1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqs9wcxj",
        "bc1pqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqw8flwm",
        "bc1pqqqpaw88az",
        "bc1pqqqqqq90twsu",
    ] {
        assert_none(&crypto(), text);
    }
}

#[test]
fn ethereum_addresses() {
    for text in [
        ETH,
        "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
        "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB",
        "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
        "0x52908400098527886E0F7030069857D2E4169EE7",
        "0x8617E340B3D01FA5F11F306F4090FD50E238070D",
        "0xde709f2102306220921060314715629080e2fb77",
        "0x27b1fdb04752bbc536007a920d24acb045561c26",
    ] {
        let r = rows(&crypto(), text);
        assert_eq!((r.len(), r[0].0, r[0].1, r[0].2), (1, 0, text.len() as u32, "eth-address"), "{text}");
    }
}

#[test]
fn ethereum_normalizes_to_eip_55() {
    for (text, want) in [
        ("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed", ETH),
        ("0x5AAEB6053F3E94C9B9A09F33669435E7EF1BEAED", ETH),
        (ETH, ETH),
        ("0X5aaeb6053f3e94c9b9a09f33669435e7ef1beaed", ETH),
        ("0xfb6916095ca1df60bb79ce92ce3ea74c37c5d359", "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359"),
    ] {
        assert_eq!(normalized(&crypto(), text), [want]);
    }
}

#[test]
fn confidence_follows_the_case() {
    let lower = layer(&crypto(), "0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed");
    let mixed = layer(&crypto(), ETH);
    assert_eq!(lower.mentions().next().unwrap().confidence.name(), "format");
    assert_eq!(mixed.mentions().next().unwrap().confidence.name(), "checksum");
}

#[test]
fn ethereum_broken_checksums_and_near_misses() {
    for text in [
        "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAeD",
        "0x5AAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
        "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5D359",
        "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6Fb",
    ] {
        assert_none(&crypto(), text);
    }
    for text in [
        "0x5aaeb6053f3e94c9b9a09f33669435e7ef1beae",
        "0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaedd",
        "0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaeg",
        "5aaeb6053f3e94c9b9a09f33669435e7ef1beaed",
        "0x0000000000000000000000000000000000000000",
        "x0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed",
        "0y5aaeb6053f3e94c9b9a09f33669435e7ef1beaed",
    ] {
        assert!(!eth(text), "{text}");
    }
}

#[test]
fn addresses_in_a_sentence() {
    let text = format!("Send to {LEGACY} or {ETH}, thanks.");
    assert_eq!(covered(&crypto(), &text), [LEGACY, ETH]);
    assert_eq!(kinds(&crypto(), &format!("{LEGACY} {P2SH}")), ["btc-address", "btc-address"]);
    let both = format!("{LEGACY} and {ETH}");
    assert_eq!(kinds(&only(Extractor::Crypto, &[PiiKind::BtcAddress]), &both), ["btc-address"]);
    assert_eq!(kinds(&only(Extractor::Crypto, &[PiiKind::EthAddress]), &both), ["eth-address"]);
}

#[test]
fn text_without_an_address() {
    for text in [
        "no address here",
        "commit 8f14e45fceea167a5a36dedd4bea2543",
        "0x1234",
        "1234567890",
        "",
        "13 items in 3 boxes",
        "commit da39a3ee5e6b4b0d3255bfef95601890afd80709",
    ] {
        assert_none(&crypto(), text);
    }
}

fn addresses() -> Vec<(String, &'static str, String)> {
    vec![
        (ZERO_LEGACY.to_owned(), "btc-address", ZERO_LEGACY.to_owned()),
        (SEGWIT.to_owned(), "btc-address", SEGWIT.to_owned()),
        (SEGWIT.to_ascii_uppercase(), "btc-address", SEGWIT.to_owned()),
        (ETH.to_owned(), "eth-address", ETH.to_owned()),
        (ETH.to_ascii_lowercase(), "eth-address", ETH.to_owned()),
        (ETH.to_ascii_uppercase(), "eth-address", ETH.to_owned()),
    ]
}

#[test]
fn delimiters_around_an_address() {
    for (address, kind, norm) in addresses() {
        for d in
            ["", " ", "\t", "\n", "\u{a0}", "😀", "\"", "'", ",", ";", ":", "(", ")", "[", "]", "_", "-", ".", "\0"]
        {
            let text = format!("{d}{address}{d}");
            let start = d.len() as u32;
            assert_found(&crypto(), &text, &[(start, start + address.len() as u32, kind, &norm)]);
        }
    }
}

#[test]
fn word_characters_around_an_address() {
    for (address, _, _) in addresses() {
        for c in ["a", "Z", "0", "é", "中", "𐐀", "١", "𝟙"] {
            assert_none(&crypto(), &format!("{c}{address}"));
            assert_none(&crypto(), &format!("{address}{c}"));
        }
    }
}

#[test]
fn upper_case_ethereum_is_masked_whole() {
    let address = "0X5AAEB6053F3E94C9B9A09F33669435E7EF1BEAED";
    let text = format!("Address: {address}.");
    let pii = layer(&crypto(), &text);
    assert_eq!(pii.mentions().next().unwrap().normalized, ETH);
    assert_eq!(
        turbo_pii::rewrite::mask_all(&text, &pii, '*').unwrap(),
        format!("Address: {}.", "*".repeat(address.len()))
    );
}

#[test]
fn detection_is_shared_across_threads() {
    let d = crypto();
    let text = format!("{ZERO_LEGACY} {} {}", SEGWIT.to_ascii_uppercase(), ETH.to_ascii_lowercase());
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                for _ in 0..32 {
                    assert_eq!(normalized(&d, &text), [ZERO_LEGACY, SEGWIT, ETH]);
                }
            });
        }
    });
}
