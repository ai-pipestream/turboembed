//! Fixed fingerprints recorded before adopting the shared hasher.
use turbo_analysis::Annotator;
use turbo_numeric::money::{Currency, Date, DatedMoneyConverter, EcbRates, MoneyConverter, MoneyExtractor};
use turbo_numeric::{NumberNotation, TemporalExtractor};

#[test]
fn hash_compat_numeric_reference_symbols_and_flags() {
    let mut hashes = Vec::new();
    let usd = Currency::parse("USD").unwrap();
    let eur = Currency::parse("EUR").unwrap();
    for parity in [false, true] {
        hashes.push(TemporalExtractor::new().with_opennlp_parity(parity).config_hash());
        for date in ["1969-12-31", "1970-01-01", "2026-07-14"] {
            hashes.push(
                TemporalExtractor::with_reference(Date::parse(date).unwrap()).with_opennlp_parity(parity).config_hash(),
            );
        }
        for notation in [NumberNotation::LatinUs, NumberNotation::LatinEu] {
            hashes.push(MoneyExtractor::new(notation).with_opennlp_parity(parity).config_hash());
            hashes.push(
                MoneyExtractor::with_symbols([('€', eur), ('$', usd)], notation)
                    .unwrap()
                    .with_opennlp_parity(parity)
                    .config_hash(),
            );
        }
        for target in [usd, eur] {
            let rates = || EcbRates::parse("Date,USD,\n2026-07-10,1.2,\n").unwrap();
            hashes.push(
                MoneyConverter::new(rates(), target, Date::parse("2026-07-10").unwrap())
                    .with_minor_unit_rounding(parity)
                    .config_hash(),
            );
            hashes.push(DatedMoneyConverter::new(rates(), target).with_minor_unit_rounding(parity).config_hash());
        }
    }
    // Fixed values from the previous implementation, before migration.
    assert_eq!(
        hashes,
        [
            590684067820433389,
            17857015481493844361,
            589727492704079044,
            603335048614818380,
            3932204394527099820,
            8868782043000568424,
            13111831815224636505,
            6758972686239557501,
            14620031065552304354,
            14598359691360076319,
            4492553279402714018,
            4479889104468790495,
            15658190276026904068,
            3411760286923090456,
            15034027211670106815,
            8425856777961928919,
            13736002451435771639,
            10151180069813026923,
            5995554905529250184,
            4844096047218153396,
            3349011632369641913,
            2733602875863570682,
            4716471891601526265,
            11937326488270249274
        ]
    );
}
