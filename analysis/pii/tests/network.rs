//! `NetworkPiiExtractorTest` and `NetworkAddressFormsTest`: IPv4, IPv6
//! and MAC addresses. Addresses come from the documentation ranges, the
//! private ranges, or are made up.

mod support;

use support::*;
use turbo_pii::rewrite::mask_all;
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

const IPV4: &str = "192.0.2.1";
const IPV6: &str = "2001:db8::";
const MAC: &str = "02:1b:44:11:3a:b7";

fn network() -> PiiDetector {
    pack(Pack::Network)
}

fn just(kind: PiiKind) -> PiiDetector {
    only(Extractor::Network, &[kind])
}

fn has(d: &PiiDetector, text: &str, kind: &str) -> bool {
    kinds(d, text).contains(&kind)
}

#[test]
fn ipv4_addresses() {
    for text in [
        "192.168.1.1",
        "10.0.0.1",
        "5.6.7.8",
        "1.2.3.4",
        "203.0.113.195",
        "255.255.255.0",
        "0.0.0.1",
        "127.0.0.1",
        "100.64.0.1",
        "172.16.254.1",
    ] {
        assert_found(&network(), text, &[(0, text.len() as u32, "ipv4", text)]);
    }
}

#[test]
fn ipv4_near_misses() {
    for text in [
        "256.1.1.1",
        "1.256.1.1",
        "1.1.1.256",
        "300.300.300.300",
        "1.2.3",
        "1.2.3.4.5",
        "1.2.3.4.5.6",
        "01.2.3.4",
        "1.02.3.4",
        "1.2.3.04",
        "1.2.3.4444",
        "1234.1.1.1",
        "0.0.0.0",
        "255.255.255.255",
        "1.2.3.-4",
        "1..2.3",
        "a1.2.3.4",
        "1.2.3.4a",
        "v1.2.3.4",
        "10.0.0.1.example.com",
    ] {
        assert!(!has(&network(), text, "ipv4"), "{text}");
    }
}

#[test]
fn ipv4_in_context() {
    assert_eq!(covered(&network(), "The host at 192.0.2.44, port 8080, timed out."), ["192.0.2.44"]);
    assert_eq!(normalized(&network(), "route 10.1.2.3 10.1.2.4"), ["10.1.2.3", "10.1.2.4"]);
    assert_eq!(normalized(&network(), "block 198.51.100.14/24 now"), ["198.51.100.14"]);
    assert_eq!(covered(&network(), "😀 host 192.0.2.1 😀"), ["192.0.2.1"]);
}

#[test]
fn ipv6_addresses() {
    for (text, norm) in [
        ("2001:0db8:0000:0000:0000:ff00:0042:8329", "2001:db8::ff00:42:8329"),
        ("2001:db8:0:0:0:ff00:42:8329", "2001:db8::ff00:42:8329"),
        ("2001:db8::ff00:42:8329", "2001:db8::ff00:42:8329"),
        ("fe80::1", "fe80::1"),
        ("2001:db8::", "2001:db8::"),
        ("fd00:1234:5678:9abc:def0:1234:5678:9abc", "fd00:1234:5678:9abc:def0:1234:5678:9abc"),
        ("1:2:3:4:5:6:7:8", "1:2:3:4:5:6:7:8"),
        ("2001:DB8::1", "2001:db8::1"),
        ("::ffff:192.0.2.128", "::ffff:c000:280"),
        ("::192.0.2.128", "::c000:280"),
        ("2001:db8::192.0.2.128", "2001:db8::c000:280"),
        ("2001:db8:4700::1111", "2001:db8:4700::1111"),
        ("fe80::0202:b3ff:fe1e:8329", "fe80::202:b3ff:fe1e:8329"),
    ] {
        let end = text.len() as u32;
        match text.find("192.0.2.128") {
            Some(at) => {
                assert_found(&network(), text, &[(0, end, "ipv6", norm), (at as u32, end, "ipv4", "192.0.2.128")])
            }
            None => assert_found(&network(), text, &[(0, end, "ipv6", norm)]),
        }
    }
    for form in [
        "2001:0db8:0000:0000:0000:0000:0000:0001",
        "2001:db8:0:0:0:0:0:1",
        "2001:db8::0:1",
        "2001:db8::1",
        "2001:0DB8::1",
    ] {
        assert_eq!(normalized(&network(), form), ["2001:db8::1"]);
    }
}

#[test]
fn ipv6_near_misses() {
    for text in [
        "12:34:56",
        "10:30",
        "23:59:59",
        "a::b",
        "aa::bb",
        "::1",
        "::",
        "Foo::bar",
        "std::vector",
        "1:2:3:4:5:6:7:8:9",
        "2001:db8:::1",
        "2001:db8::1::2",
        "12345::1",
        "gggg::1",
        "2001:db8:0:0:0:0:0:0:1",
        "x2001:db8::1",
    ] {
        assert!(!has(&network(), text, "ipv6"), "{text}");
    }
    assert_eq!(covered(&network(), "Peer 2001:db8::dead:beef went away."), ["2001:db8::dead:beef"]);
}

#[test]
fn mac_addresses() {
    for (text, norm) in [
        ("00:1B:44:11:3A:B7", "00:1b:44:11:3a:b7"),
        ("00:1b:44:11:3a:b7", "00:1b:44:11:3a:b7"),
        ("00-1B-44-11-3A-B7", "00:1b:44:11:3a:b7"),
        ("001b.4411.3ab7", "00:1b:44:11:3a:b7"),
        ("3C:5A:B4:00:00:01", "3c:5a:b4:00:00:01"),
        ("de:ad:be:ef:00:01", "de:ad:be:ef:00:01"),
        ("DE-AD-BE-EF-00-01", "de:ad:be:ef:00:01"),
        ("0800.2b01.0203", "08:00:2b:01:02:03"),
    ] {
        assert_found(&network(), text, &[(0, text.len() as u32, "mac", norm)]);
    }
    for text in [
        "00:1b:44:11:3a",
        "00:1b:44:11:3a:b7:c9",
        "00:1b:44:11:3a:bg",
        "00:1b-44:11:3a:b7",
        "001:b44:113:ab7:00:11",
        "00:00:00:00:00:00",
        "ff:ff:ff:ff:ff:ff",
        "FF-FF-FF-FF-FF-FF",
        "001b.4411.3ab",
        "001b.4411.3ab7.1234",
        "z00:1b:44:11:3a:b7",
        "00:1b:44:11:3a:b77",
    ] {
        assert!(!has(&network(), text, "mac"), "{text}");
    }
    assert_none(&network(), "aa:00:1b:44:11:3a:b7");
}

#[test]
fn three_kinds_in_one_line() {
    let text = "host 10.1.2.3 mac 00:1b:44:11:3a:b7 peer 2001:db8::1 done";
    assert_eq!(kinds(&network(), text), ["ipv4", "mac", "ipv6"]);
    assert_eq!(kinds(&just(PiiKind::Ipv4), text), ["ipv4"]);
    assert_eq!(kinds(&just(PiiKind::Mac), text), ["mac"]);
    assert_eq!(kinds(&just(PiiKind::Ipv6), text), ["ipv6"]);
    for text in ["no address here", "version 3.9 released", "call (555) 123-4567", "jane@example.com", "", "1,234.56"] {
        assert_none(&network(), text);
    }
}

#[test]
fn trailing_compression() {
    for suffix in [
        "", " ", "\n", "\t", ".", ",", ";", ")", "]", "}", "!", "?", "\"", "'", "/64", "%eth0", "😀", "\u{a0}", "...",
        ".. ",
    ] {
        let prefix = "😀 [";
        let start = prefix.len() as u32;
        assert_found(
            &just(PiiKind::Ipv6),
            &format!("{prefix}{IPV6}{suffix}"),
            &[(start, start + IPV6.len() as u32, "ipv6", IPV6)],
        );
    }
}

#[test]
fn invalid_ipv6_continuations() {
    for text in [
        "2001:db8::192.0.2.1.5",
        "2001:db8::192.0.2.256",
        "2001:db8::192.0.2.01",
        "2001:db8::192.0.2",
        "2001:db8::192..2.1",
        "2001:db8::1.example",
        "2001:db8::1.2",
        "2001:db8::1.é",
        "2001:db8::1.𐐀",
        "::ffff:192.0.2.1.5",
        "::ffff:192.0.2.999",
        "2001:db8::192.0.2.1:5",
        "2001:db8:::1",
        "2001:db8:::",
        "2001:db8::x",
        "2001:db8::é",
        "2001:db8::𐐀",
    ] {
        assert_none(&just(PiiKind::Ipv6), text);
    }
}

#[test]
fn structured_endings() {
    let ipv6 = format!("{IPV6}1");
    for next in ["a", "1", "é", "中", "𐐀", "١", "𝟙"] {
        for (kind, address, separator) in [
            (PiiKind::Ipv4, IPV4, "."),
            (PiiKind::Ipv6, ipv6.as_str(), "."),
            (PiiKind::Mac, MAC, ":"),
            (PiiKind::Mac, "02-1b-44-11-3a-b7", "-"),
            (PiiKind::Mac, "021b.4411.3ab7", "."),
            (PiiKind::Ipv4, IPV4, ".."),
            (PiiKind::Ipv6, ipv6.as_str(), ".."),
            (PiiKind::Mac, MAC, "::"),
            (PiiKind::Mac, "02-1b-44-11-3a-b7", "--"),
            (PiiKind::Mac, "021b.4411.3ab7", ".."),
        ] {
            assert_none(&just(kind), &format!("{address}{separator}{next}"));
        }
    }
}

#[test]
fn dotted_prefixes() {
    let ipv6 = format!("{IPV6}1");
    for prefix in ["host.", "a.", "1.", "é.", "中.", "𐐀.", "١.", "𝟙.", "host..", "𐐀..", "1.."] {
        for (kind, address) in [(PiiKind::Ipv4, IPV4), (PiiKind::Ipv6, ipv6.as_str()), (PiiKind::Mac, "021b.4411.3ab7")]
        {
            assert_none(&just(kind), &format!("{prefix}{address}"));
        }
    }
}

#[test]
fn unspecified_and_short_ipv6() {
    for text in [
        "::",
        "0:0:0:0:0:0:0:0",
        "0000:0000:0000:0000:0000:0000:0000:0000",
        "0:0:0:0:0:0:0.0.0.0",
        "0:0:0:0:0:0.0.0.0",
        "::0.0.0.0",
        "000:000::0",
        "::1",
        "a::b",
        "aa::bb",
        "2001::",
        "::abcd",
    ] {
        assert_none(&just(PiiKind::Ipv6), text);
    }
    for text in ["::ffff:0.0.0.0", "2001:db8::0.0.0.0", "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"] {
        let r = rows(&just(PiiKind::Ipv6), text);
        assert_eq!((r.len(), r[0].0, r[0].1), (1, 0, text.len() as u32), "{text}");
    }
    for count in [8, 9, 64, 4096] {
        let text = vec!["abcd"; count].join(":");
        assert_eq!(rows(&just(PiiKind::Ipv6), &text).len(), usize::from(count == 8));
    }
}

#[test]
fn ipv6_zero_runs() {
    for mask in 0..256u32 {
        let zero = |i: usize| mask & (1 << i) != 0;
        let groups: Vec<String> = (0..8).map(|i| if zero(i) { "0".to_owned() } else { format!("ab{i}") }).collect();
        let source =
            (0..8).map(|i| if zero(i) { "0000".to_owned() } else { format!("0AB{i}") }).collect::<Vec<_>>().join(":");
        let pattern: String = (0..8).map(|i| if zero(i) { '0' } else { 'x' }).collect();
        let mut norm = groups.join(":");
        for length in (2..=8).rev() {
            if let Some(start) = pattern.find(&"0".repeat(length)) {
                norm = format!("{}::{}", groups[..start].join(":"), groups[start + length..].join(":"));
                break;
            }
        }
        if mask == 255 {
            assert_none(&just(PiiKind::Ipv6), &source);
        } else {
            assert_found(&just(PiiKind::Ipv6), &source, &[(0, source.len() as u32, "ipv6", &norm)]);
        }
    }
}

#[test]
fn ipv4_octets() {
    for position in 0..4 {
        for (value, accepted) in [
            ("0", true),
            ("1", true),
            ("9", true),
            ("10", true),
            ("99", true),
            ("100", true),
            ("254", true),
            ("255", true),
            ("00", false),
            ("01", false),
            ("001", false),
            ("256", false),
            ("999", false),
            ("1000", false),
            ("١", false),
            ("𝟙", false),
        ] {
            let mut groups = ["192", "0", "2", "1"];
            groups[position] = value;
            let source = groups.join(".");
            if accepted {
                assert_found(&just(PiiKind::Ipv4), &source, &[(0, source.len() as u32, "ipv4", &source)]);
            } else {
                assert_none(&just(PiiKind::Ipv4), &source);
            }
        }
    }
}

#[test]
fn network_pack_masks_every_address() {
    let text = "Peer [2001:db8::], host 192.0.2.1, MAC 02-1B-44-11-3A-B7.";
    assert_eq!(normalized(&network(), text), [IPV6, IPV4, MAC]);
    assert_eq!(
        mask_all(text, &layer(&network(), text), '*').unwrap(),
        "Peer [**********], host *********, MAC *****************."
    );
}

#[test]
fn addresses_on_every_thread() {
    let d = network();
    std::thread::scope(|s| {
        for worker in 0..8 {
            let d = &d;
            s.spawn(move || {
                let prefix = format!("{} [", "😀".repeat(worker));
                let text = format!("{prefix}{IPV6}] {IPV4} {MAC}");
                for _ in 0..32 {
                    let r = rows(d, &text);
                    assert_eq!(r.iter().map(|r| r.3.as_str()).collect::<Vec<_>>(), [IPV6, IPV4, MAC]);
                    assert_eq!((r[0].0, r[0].1), (prefix.len() as u32, (prefix.len() + IPV6.len()) as u32));
                }
            });
        }
    });
}
