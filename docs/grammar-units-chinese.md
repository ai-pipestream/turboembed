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
