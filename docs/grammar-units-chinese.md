# Chinese unit conventions

Chinese unit spellings use modern mainland quantities unless an explicit unit
name says otherwise. The shared unit model retains their original names and
normalizes them as follows:

| Unit | Base value | Convention |
| --- | --- | --- |
| jin (斤) | 0.5 kg | Modern mainland market jin |
| liang (两, 兩) | 0.05 kg | Ten liang per modern mainland jin |
| mu (亩, 畝) | 2000/3 square metres | Fifteen mu per hectare |
| kilocalorie (千卡) | 4184 joules | Thermochemical calorie |

These are explicit policies, not inferred historic or regional conversions.
The older sixteen-liang jin and regional weight/area systems must not silently
reuse these variants. Conversion of mu multiplies before dividing: three mu
normalize exactly to 2000 square metres. A standalone decimal conversion factor
for one mu is necessarily rounded because its decimal expansion repeats.

Primary references:

- [Fujian statistics unit conversion table](https://tjj.fujian.gov.cn/xxgk/zdff/202410/P020241024592998124217.pdf)
  gives the modern market jin as 500 grams.
- [Prince Gong's Palace Museum on the 1959 weight reform](https://www.pgm.org.cn/pgm/xxdt/202112/45faa74de446499c98c1acddeef83ca7.shtml)
  distinguishes ten-liang and historic sixteen-liang systems.
- [Suzhou archives land-unit explanation](https://www.sipac.gov.cn/szdaglzx/yqfzktzz/202102/f15b9b3167fa4f17980a78e42e9fb066.shtml)
  gives fifteen mu per hectare.
- [NIST conversion factors](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b9)
  distinguish the thermochemical kilocalorie from other calorie conventions.


## Grammar and source evidence

`CHINESE_UNITS` and `GrammarAnnotator::new(&CHINESE_UNITS)` recognize Chinese
ages, physical measures, temperatures and currencies. The shared CJK reader
works directly on borrowed source text, including fullwidth decimal digits,
Arabic/ideographic mixtures, traditional variants, negatives and large integer
amounts. Chinese uses only its numeral vocabulary, not Japanese kana spellings.
Longest complete unit spellings take precedence over numeric prefixes:
`12千米` means twelve kilometres, while `三百米` means three hundred metres.

Currency composition preserves identities and ordered minor places:
`50元6角三分` is CNY 50.63, and `两美元又三美分` is USD 2.03. Explicit historical
Japanese sen are accepted in `10日元5日本銭`; this does not infer hundredths for
unqualified yen amounts. Fullwidth Chinese commas separate clauses, so
`$20，300美圆` remains two prices. The person classifier `位` is not a bit unless
its immediate explanatory context identifies storage/bits; `五角大楼` is not
an amount of money. These are bounded grammar disambiguations, not general
semantic analysis.

All 91 unchanged Chinese NumberWithUnit cases at Recognizers-Text revision
`da7edcff59f669b2a460ab9d400e36298f0d658e` pass: 10 ages, 35 currencies,
36 dimensions and 10 temperatures. No unsupported case is omitted. The test
checks original UTF-16 boundaries against source byte offsets, unit names,
values, result types, and explicit ISO currency or subtype fields. Chinese
fixtures omit dimension subtypes; absent optional fields are unspecified,
not fabricated expectations. Numeric comparison uses the established relative
1e-12 tolerance for upstream serialized floating-point values. A separate real
Analyzer regression proves the 24-digit integer remains exact in the Money
layer rather than adopting the upstream rounded scientific notation.

Native tests also cover Unicode source-offset round trips, fullwidth source
preservation, currency/place refusals, person classifiers and Latin identifiers.
The CJK boundary hook leaves the existing word-boundary policy unchanged for
other grammars. Existing core and shared-layer invariants still validate output.

This is one language slice of #929; its other language reviews remain separate.
