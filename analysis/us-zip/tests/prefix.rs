//! Which states a three-digit ZIP prefix belongs to.

use turbo_us_zip::{UsZipTable, zip3_in_state};

/// The prefix ranges the Postal Service assigns to each state, territory
/// and armed forces designation; every one is still admitted.
const ASSIGNED: &[(&str, &[(u16, u16)])] = &[
    ("AL", &[(350, 369)]),
    ("AK", &[(995, 999)]),
    ("AZ", &[(850, 865)]),
    ("AR", &[(716, 729)]),
    ("CA", &[(900, 961)]),
    ("CO", &[(800, 816)]),
    ("CT", &[(60, 69)]),
    ("DE", &[(197, 199)]),
    ("FL", &[(320, 339), (341, 349)]),
    ("GA", &[(300, 319), (398, 399)]),
    ("HI", &[(967, 968)]),
    ("ID", &[(832, 838)]),
    ("IL", &[(600, 629)]),
    ("IN", &[(460, 479)]),
    ("IA", &[(500, 528)]),
    ("KS", &[(660, 679)]),
    ("KY", &[(400, 427)]),
    ("LA", &[(700, 714)]),
    ("ME", &[(39, 49)]),
    ("MD", &[(206, 219)]),
    ("MA", &[(10, 27), (55, 55)]),
    ("MI", &[(480, 499)]),
    ("MN", &[(550, 567)]),
    ("MS", &[(386, 397)]),
    ("MO", &[(630, 658)]),
    ("MT", &[(590, 599)]),
    ("NE", &[(680, 693)]),
    ("NV", &[(889, 898)]),
    ("NH", &[(30, 38)]),
    ("NJ", &[(70, 89)]),
    ("NM", &[(870, 884)]),
    ("NY", &[(5, 5), (63, 63), (100, 149)]),
    ("NC", &[(270, 289)]),
    ("ND", &[(580, 588)]),
    ("OH", &[(430, 459)]),
    ("OK", &[(730, 732), (734, 749)]),
    ("OR", &[(970, 979)]),
    ("PA", &[(150, 196)]),
    ("RI", &[(28, 29)]),
    ("SC", &[(290, 299)]),
    ("SD", &[(570, 577)]),
    ("TN", &[(370, 385)]),
    ("TX", &[(733, 733), (750, 799), (885, 885)]),
    ("UT", &[(840, 847)]),
    ("VT", &[(50, 54), (56, 59)]),
    ("VA", &[(201, 201), (220, 246)]),
    ("WA", &[(980, 994)]),
    ("WV", &[(247, 268)]),
    ("WI", &[(530, 549)]),
    ("WY", &[(820, 831)]),
    ("DC", &[(200, 200), (202, 205), (569, 569)]),
    ("AS", &[(967, 967)]),
    ("GU", &[(969, 969)]),
    ("MP", &[(969, 969)]),
    ("FM", &[(969, 969)]),
    ("MH", &[(969, 969)]),
    ("PW", &[(969, 969)]),
    ("PR", &[(6, 7), (9, 9)]),
    ("VI", &[(8, 8)]),
    ("UM", &[(968, 968)]),
    ("AA", &[(340, 340)]),
    ("AE", &[(90, 98)]),
    ("AP", &[(962, 966)]),
];

#[test]
fn every_assigned_prefix_belongs_to_its_state() {
    for (state, ranges) in ASSIGNED {
        for &(lo, hi) in *ranges {
            for p in lo..=hi {
                let zip3 = format!("{p:03}");
                assert!(zip3_in_state(zip3.as_bytes(), state.as_bytes()), "{state} {zip3}");
            }
        }
    }
}

#[test]
fn every_prefix_with_zctas_belongs_to_their_states() {
    let t = UsZipTable::bundled();
    for p in 0..1000 {
        let zip3 = format!("{p:03}");
        for (state, _) in t.prefix_states(&zip3) {
            assert!(zip3_in_state(zip3.as_bytes(), state.as_bytes()), "{state} {zip3}");
        }
    }
}

#[test]
fn a_prefix_of_another_state_is_refused() {
    assert!(zip3_in_state(b"627", b"IL"));
    assert!(!zip3_in_state(b"123", b"IN"));
    assert!(!zip3_in_state(b"902", b"OR"));
    assert!(!zip3_in_state(b"000", b"NY"));
    // A ZCTA of prefix 422 lies mostly in Tennessee.
    assert!(zip3_in_state(b"422", b"TN"));
}

#[test]
fn codes_and_prefixes_that_are_not_ascii_are_refused() {
    assert!(!zip3_in_state(b"627", b"il"));
    assert!(!zip3_in_state(b"627", b"ILL"));
    assert!(!zip3_in_state(b"62", b"IL"));
    assert!(!zip3_in_state("\u{ff16}2".as_bytes(), b"IL"));
    assert!(!zip3_in_state(b"6a7", b"IL"));
}
