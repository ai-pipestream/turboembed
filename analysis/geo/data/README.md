# Bundled gazetteer table

`naturalearth-populated-places.txt` is the table OpenNLP's
`BundledGazetteer` loads, copied byte for byte from
`geo/src/main/resources/opennlp/geo/naturalearth-populated-places.txt` on the
`ready/geo` branch of opennlp-addons (commit
`abaada5e21c3177d911233149f5c7095a6266fb4`). `BundledGazetteer` embeds it
with `include_str!`.

| | |
|---|---|
| Rows | 7,342 places (7,400 lines with the comment header) |
| Encoding | ASCII |
| SHA-256 | `8a5db2b986fdbe7daf6c6992f98a5bad7ee4af01e9243cfdf2d384418140a074` |
| SHA-512 | `58630d4e7ea774d33dfaa4f29d0b0d5ff0f21f785b246e3b3a2ad9c5c5c6b75c71f31599fdced78148548b24046165d993a61cc39c73fbcfcac4854132ffda05` |

The OpenNLP project derived it from Natural Earth's 1:10m "Populated Places"
layer (`ne_10m_populated_places`, natural-earth-vector 5.2.0-pre). Natural
Earth is in the public domain. The file's header records the derivation:
where each field comes from, how names were folded to ASCII, and how damaged
upstream values were left out or repaired. The JVM goldens check the digest
above against the resource in the OpenNLP jar (`analysis/geo/README.md`).

No GeoNames or Overture data is bundled. GeoNames is under CC BY 4.0 and
Overture's divisions under ODbL, so callers supply those tables and their
attribution themselves.
