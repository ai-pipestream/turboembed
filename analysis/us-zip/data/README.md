# Bundled ZIP tables

Four tab separated tables, built from US Census Bureau files, which are
in the public domain as works of the US federal government. Each table
starts with `#` lines naming the input files and their SHA-256 digests,
then its columns.

| Table | Rows | Columns |
|---|---|---|
| `us-zcta.txt` | 33,791 ZCTAs | zcta, state, land_km2, lat, lon, west, south, east, north, counties, places |
| `us-places.txt` | 32,188 places | geoid, state, name, kind, lat, lon |
| `us-counties.txt` | 3,235 counties | geoid, state, name |
| `zip3-states.txt` | 896 prefixes | prefix, states |

- **state** is the USPS code of the state whose counties hold most of
  the ZCTA's land.
- **lat, lon** is the Census internal point, a point inside the area near
  its centre, to four decimal places. The places of the island areas
  (American Samoa, Guam, the Northern Mariana Islands, the US Virgin
  Islands) have none, since the place Gazetteer leaves them out; their
  names come from the ZCTA to place relationship file.
- **west, south, east, north** is the box of the ZCTA's 1:500,000
  cartographic boundary shape, rounded outward to three decimal places so
  it still holds the shape, and widened where needed to hold the internal
  point (a 2024 point can sit just outside the generalized 2020 shape of a
  small coastal ZCTA). The shapes themselves are not bundled.
- **counties, places** list the county and place GEOIDs a ZCTA overlaps,
  each with its share of the ZCTA's land in thousandths, largest first.
  Parts with no land are left out, so a ZCTA wholly outside every place
  has no places.
- **name, kind** split a Census place name from its description: the
  trailing words that are `CDP`, in brackets or start with a lower case
  letter (`Springfield city` is `Springfield` and `city`).
- **states** of a prefix are the states of its ZCTAs, each with how many.

## Inputs

| File | Source |
|---|---|
| `2024_Gaz_zcta_national.txt` | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_zcta_national.zip |
| `2020_Gaz_place_national.txt` | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_place_national.zip |
| `tab20_zcta520_county20_natl.txt` | https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt |
| `tab20_zcta520_place20_natl.txt` | https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_place20_natl.txt |
| `national_county2020.txt` | https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt |
| `cb_2020_us_zcta520_500k.shp`, `.dbf` | https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip |

## Rebuilding

Download and unpack the files above into one folder, then

```text
cargo run --release -p turbo-us-zip --example census_tables -- <folder> analysis/us-zip/data
```

The builder reads the shapefile's record boxes and the dBASE key column
itself, so it needs no library beyond the workspace's own. Its output is
the same for the same inputs.

## What a ZCTA is not

A ZCTA approximates the area of the ZIP Code most addresses in it use.
ZIP Codes of post office boxes only, of one organization and military
ZIP Codes have no ZCTA. A Census place is a city, town, village or
census designated place, which is not always the city name the Postal
Service prints on mail for that ZIP Code.
