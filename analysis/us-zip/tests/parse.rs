//! Tables in the bundled format that break it are refused with the table,
//! line and reason.

use turbo_us_zip::UsZipTable;

const COUNTIES: &str = "# geoid\tstate\tname\n17167\tIL\tSangamon County\n";
const PLACES: &str = "1772000\tIL\tSpringfield\tcity\t39.7800\t-89.6400\n";
const PREFIXES: &str = "627\tIL:1\n";
const ROW: &str = "62704\tIL\t53.17\t39.7712\t-89.6820\t-89.750\t39.723\t-89.620\t39.812\t17167:1000\t1772000:940";

fn parse(zctas: &str) -> Result<UsZipTable, String> {
    UsZipTable::parse(zctas, PLACES, COUNTIES, PREFIXES).map_err(|e| e.to_string())
}

#[test]
fn a_small_table_parses() {
    let t = parse(&format!("# header\n{ROW}\n")).unwrap();
    let z = t.zcta("62704").unwrap();
    assert_eq!(z.places().next().unwrap().1, 940);
    assert_eq!(z.point().latitude(), 39.7712);
    assert_eq!(z.bounding_box().west(), -89.75);
}

#[test]
fn broken_rows_are_refused_with_their_line() {
    let cases = [
        (ROW.replace("62704", "6270"), "us-zcta line 2: zcta is not 5 ASCII digits"),
        (ROW.replace("\tIL\t", "\til\t"), "us-zcta line 2: state is not two capital letters"),
        (ROW.replace("39.7712", "39.77"), "us-zcta line 2: lat is not a coordinate with 4 decimal places"),
        (ROW.replace("39.7712", "99.7712"), "us-zcta line 2: latitude must be in"),
        (ROW.replace("17167:1000", ""), "us-zcta line 2: no county"),
        (ROW.replace("17167:1000", "17167:0"), "us-zcta line 2: counties part is not GEOID:share"),
        (ROW.replace("17167:1000", "17001:1000"), "names county 17001 that its table lacks"),
        (format!("{ROW}\textra"), "us-zcta line 2: too many fields"),
        (ROW.replace("\t1772000:940", ""), "us-zcta line 2: no places field"),
    ];
    for (row, want) in cases {
        let err = parse(&format!("# header\n{row}\n")).unwrap_err();
        assert!(err.contains(want), "{err:?} lacks {want:?}");
    }
}

#[test]
fn rows_out_of_order_are_refused() {
    let err = parse(&format!("{ROW}\n{ROW}\n")).unwrap_err();
    assert!(err.contains("out of order"), "{err}");
}

#[test]
fn a_coordinate_that_would_wrap_is_refused() {
    for (row, want) in [
        (ROW.replace("\t-89.750\t", "\t4294877.546\t"), "west is not a coordinate"),
        (ROW.replace("39.7712", "429536.5096"), "lat is not a coordinate"),
        (ROW.replace("-89.6820", "-180.0001"), "lon is not a coordinate"),
    ] {
        let err = parse(&format!("{row}\n")).unwrap_err();
        assert!(err.contains(want), "{err:?} lacks {want:?}");
    }
    let place = PLACES.replace("39.7800", "429536.5096");
    let err = UsZipTable::parse(&format!("{ROW}\n"), &place, COUNTIES, PREFIXES).unwrap_err().to_string();
    assert!(err.contains("us-places line 1: lat is not a decimal"), "{err}");
}

#[test]
fn a_point_outside_its_box_or_disordered_shares_are_refused() {
    for (row, want) in [
        (ROW.replace("39.7712", "40.7712"), "the point is outside the box"),
        (ROW.replace("1772000:940", "1772000:40,1799999:60"), "places shares are not largest first"),
        (ROW.replace("1772000:940", "1772000:500,1772000:440"), "places names 1772000 twice"),
    ] {
        let err = parse(&format!("{row}\n")).unwrap_err();
        assert!(err.contains(want), "{err:?} lacks {want:?}");
    }
}
