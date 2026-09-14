# Vendored native sources

These sources are vendored to keep the native tokenizer self-contained. They
introduce no shared-library runtime requirement. Their upstream license files
are retained verbatim beside the corresponding source.

## utf8proc 2.11.3

Source tag: [`v2.11.3`](https://github.com/JuliaStrings/utf8proc/tree/v2.11.3).

| Local file | Upstream URL | SHA-256 |
|---|---|---|
| `utf8proc/utf8proc.c` | <https://raw.githubusercontent.com/JuliaStrings/utf8proc/v2.11.3/utf8proc.c> | `edf80118fd34796bcabef4567cc684adebc7058f2afe8cfdc11b7a78e383ed84` |
| `utf8proc/utf8proc.h` | <https://raw.githubusercontent.com/JuliaStrings/utf8proc/v2.11.3/utf8proc.h> | `a4e498b7392c383cf3b22e662da21e0b48f1264806235d87b5d8bd166232f658` |
| `utf8proc/utf8proc_data.c` | <https://raw.githubusercontent.com/JuliaStrings/utf8proc/v2.11.3/utf8proc_data.c> | `950e549dbfc853c4304425f3af1875e72fa9fc9697c273c763400c2da4e380a7` |
| `utf8proc/LICENSE.md` | <https://raw.githubusercontent.com/JuliaStrings/utf8proc/v2.11.3/LICENSE.md> | `3b510150d34f248a221bb88e1d811238d6c6c18b51231822c42974c39bb07256` |

## nlohmann/json 3.12.0

Source tag: [`v3.12.0`](https://github.com/nlohmann/json/tree/v3.12.0).

| Local file | Upstream URL | SHA-256 |
|---|---|---|
| `nlohmann/json.hpp` | <https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp> | `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` |
| `nlohmann/LICENSE.MIT` | <https://raw.githubusercontent.com/nlohmann/json/v3.12.0/LICENSE.MIT> | `46a65cffd1ea955132d95a8dd921640714a8d6b537d2e4e482d31145ae95b603` |

The published `json.hpp` checksum matches the verified upstream v3.12.0 file.

## unicode_categories 0.1.1

Source crate: [`unicode_categories` 0.1.1](https://crates.io/crates/unicode_categories/0.1.1),
the version and crate checksum pinned in `Cargo.lock`. The generated native
header uses its exact category tables and reproduces the crate's algorithmic
private-use ranges. Both upstream license alternatives are retained.

| Local file | Upstream path | SHA-256 |
|---|---|---|
| `unicode_categories/tables.rs` | `unicode_categories-0.1.1/src/tables.rs` | `162393c222be7f49425f232abef890da4ccf7a45e2c43df34830d32c979f5bd8` |
| `unicode_categories/LICENSE-APACHE` | `unicode_categories-0.1.1/LICENSE-APACHE` | `c6596eb7be8581c18be736c846fb9173b69eccf6ef94c5135893ec56bd92ba08` |
| `unicode_categories/LICENSE-MIT` | `unicode_categories-0.1.1/LICENSE-MIT` | `98a817e7b85e5fe4e79d165c656a4a139793b8b643d0f3d9cbcc3538ef82ec82` |
