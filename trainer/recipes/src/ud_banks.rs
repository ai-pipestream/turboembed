//! The UD r2.16 treebanks beside English EWT (#1522, #1651), pinned as
//! [`crate::ud::EN_EWT`] is: release tag, commit, and the size and SHA-256
//! of each split. The CC BY 4.0 ones (AnCora Spanish and Catalan, FTB
//! Finnish, MarkIT Italian, Porttinari Portuguese) can go in a default
//! bundle; Porttinari's text rights are not settled, so no head trained on
//! it ships. The CC BY-SA 4.0 ones (GSD German, French, Russian and
//! Portuguese, Alpino Dutch) are share-alike: download after acceptance.
//! FTB is dual-licensed, LGPL 3.0 or later or CC BY 4.0; heads take it
//! under CC BY 4.0, keeping its copyright notice in the attribution.

use crate::corpus::Pin;
use crate::ud::Treebank;

/// UD Spanish AnCora r2.16.
pub const ES_ANCORA: Treebank = Treebank {
    id: "ud-es-ancora",
    name: "Universal Dependencies Spanish AnCora r2.16",
    tag: "r2.16",
    commit: "b35db4b8727856c00052b7380fdb61790c8d427f",
    licence: "CC BY 4.0",
    spdx: "CC-BY-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Spanish-AnCora/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Spanish AnCora r2.16, published at \
         https://github.com/UniversalDependencies/UD_Spanish-AnCora; used under the \
         Creative Commons Attribution 4.0 International licence (https://creativecommons.org/licenses/by/4.0/).",
    train: Pin {
        path: "es_ancora-ud-train.conllu",
        size: 42_958_918,
        sha256: "79e7463700e75bdf921c85b6e04495af118462a2b032bbff590fb82aa6ad8d5d",
    },
    dev: Pin {
        path: "es_ancora-ud-dev.conllu",
        size: 5_104_844,
        sha256: "e257745f4c339be53d5158867a28a2bfbcb98d890db86faa324e5f45f0284afc",
    },
    test: Pin {
        path: "es_ancora-ud-test.conllu",
        size: 5_106_509,
        sha256: "2a24316503a56e171f07b38cf2ec13c22e3b62a337c6dbf2d9d1a54ada2b012b",
    },
    recipe_only: false,
};

/// UD Catalan AnCora r2.16.
pub const CA_ANCORA: Treebank = Treebank {
    id: "ud-ca-ancora",
    name: "Universal Dependencies Catalan AnCora r2.16",
    tag: "r2.16",
    commit: "624910328d67168a42cbc85ed88304ecee1ca26c",
    licence: "CC BY 4.0",
    spdx: "CC-BY-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Catalan AnCora r2.16, published at \
         https://github.com/UniversalDependencies/UD_Catalan-AnCora; used under the \
         Creative Commons Attribution 4.0 International licence (https://creativecommons.org/licenses/by/4.0/).",
    train: Pin {
        path: "ca_ancora-ud-train.conllu",
        size: 39_714_229,
        sha256: "bb5d06fb14336bae2c3aa64aa7e65afe183295a4669ce4f45bf0ed57c18ccccd",
    },
    dev: Pin {
        path: "ca_ancora-ud-dev.conllu",
        size: 5_421_739,
        sha256: "bfcb95de30b6ea0b250619386a281add3e2bbd153804a89da4dbe2938ad28199",
    },
    test: Pin {
        path: "ca_ancora-ud-test.conllu",
        size: 5_515_247,
        sha256: "f68ff102b9bf8bc197be3262a1caf25682683c4db4c648663c7a791bfbfa115f",
    },
    recipe_only: false,
};

/// UD Finnish FTB r2.16.
pub const FI_FTB: Treebank = Treebank {
    id: "ud-fi-ftb",
    name: "Universal Dependencies Finnish FTB r2.16",
    tag: "r2.16",
    commit: "d34cfa9f9d1dffea9d5092b6a3189602ecdc8ac5",
    licence: "CC BY 4.0",
    spdx: "CC-BY-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Finnish-FTB/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Finnish FTB r2.16, published at \
         https://github.com/UniversalDependencies/UD_Finnish-FTB; FTB1U is Copyright 2014 FIN-CLARIN. Used under the \
         Creative Commons Attribution 4.0 International licence (https://creativecommons.org/licenses/by/4.0/).",
    train: Pin {
        path: "fi_ftb-ud-train.conllu",
        size: 10_505_459,
        sha256: "0cc04932f18613c4491ace06f597b7b8caed36e557168b9b6ef154da7ae2d71b",
    },
    dev: Pin {
        path: "fi_ftb-ud-dev.conllu",
        size: 1_296_909,
        sha256: "1e46f7f88b6f1fa4f39b010b4ac8b8664b1be2c899f531f21b7bdc3fcf2e9142",
    },
    test: Pin {
        path: "fi_ftb-ud-test.conllu",
        size: 1_342_602,
        sha256: "3aca7fbd727846c0955e9db581571a0fca89d46f825ee7f2a32fab1ea0bc31ba",
    },
    recipe_only: false,
};

/// UD Italian MarkIT r2.16.
pub const IT_MARKIT: Treebank = Treebank {
    id: "ud-it-markit",
    name: "Universal Dependencies Italian MarkIT r2.16",
    tag: "r2.16",
    commit: "fa54dc64d616cd9fc42bd2c4a0d8003f8ef47b4d",
    licence: "CC BY 4.0",
    spdx: "CC-BY-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Italian-MarkIT/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Italian MarkIT r2.16, published at \
         https://github.com/UniversalDependencies/UD_Italian-MarkIT; used under the \
         Creative Commons Attribution 4.0 International licence (https://creativecommons.org/licenses/by/4.0/).",
    train: Pin {
        path: "it_markit-ud-train.conllu",
        size: 1_266_080,
        sha256: "4f6fc3423acb00d49e1e167bd21b11ebbc23e22275fd6f5eeb4d75cfbbccc5c6",
    },
    dev: Pin {
        path: "it_markit-ud-dev.conllu",
        size: 665_430,
        sha256: "633b6186f47ecfebfcff4ff9c7e19a0cbffd1f655f14fc1e9265b12bda597f97",
    },
    test: Pin {
        path: "it_markit-ud-test.conllu",
        size: 658_412,
        sha256: "9048ed0064e3989e2c37377755a28c25c7e462b1ad916e5808dc0a7ce95fc3fc",
    },
    recipe_only: false,
};

/// UD Portuguese Porttinari r2.16.
pub const PT_PORTTINARI: Treebank = Treebank {
    id: "ud-pt-porttinari",
    name: "Universal Dependencies Portuguese Porttinari r2.16",
    tag: "r2.16",
    commit: "a86b99a6cfa333360cdedd9468bd064b31885a87",
    licence: "CC BY 4.0",
    spdx: "CC-BY-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Portuguese-Porttinari/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Portuguese Porttinari r2.16, published at \
         https://github.com/UniversalDependencies/UD_Portuguese-Porttinari; used under the \
         Creative Commons Attribution 4.0 International licence (https://creativecommons.org/licenses/by/4.0/).",
    train: Pin {
        path: "pt_porttinari-ud-train.conllu",
        size: 8_194_835,
        sha256: "d9d3d76ac9ac8cb569c42a698cc60721e088db7c7e90401582aab68aa9ca85d3",
    },
    dev: Pin {
        path: "pt_porttinari-ud-dev.conllu",
        size: 1_151_582,
        sha256: "c5efbb16458c94d8803ebf8bb9f53f9a692c0a06c77d76326642ad4139a6cf09",
    },
    test: Pin {
        path: "pt_porttinari-ud-test.conllu",
        size: 2_333_603,
        sha256: "08e51c3687f2800ea45445cb48c7030832984c316b1c39757606499b8e7577b9",
    },
    recipe_only: true,
};

/// UD German GSD r2.16.
pub const DE_GSD: Treebank = Treebank {
    id: "ud-de-gsd",
    name: "Universal Dependencies German GSD r2.16",
    tag: "r2.16",
    commit: "0726bb0c7e7328d5ef1a5d65c70d184882eacd95",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_German-GSD/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies German GSD r2.16, published at \
         https://github.com/UniversalDependencies/UD_German-GSD; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "de_gsd-ud-train.conllu",
        size: 19_627_297,
        sha256: "b01561c37fafa1b505630f69d173ca0c015a91e444a2cfe7aeca67288b1c77f2",
    },
    dev: Pin {
        path: "de_gsd-ud-dev.conllu",
        size: 904_804,
        sha256: "00431f2624cbbf66796c097a9303bbc8f8e8ca88ef35c4c5664aa396b2d674c9",
    },
    test: Pin {
        path: "de_gsd-ud-test.conllu",
        size: 1_207_783,
        sha256: "bceb8e867ba3ee7f1449d44fb31538d63a514f07e6763e7d9a5ffe794407bed4",
    },
    recipe_only: false,
};

/// UD French GSD r2.16.
pub const FR_GSD: Treebank = Treebank {
    id: "ud-fr-gsd",
    name: "Universal Dependencies French GSD r2.16",
    tag: "r2.16",
    commit: "42fdb4fc842d060c9f8af663c6ba5db229354e8c",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_French-GSD/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies French GSD r2.16, published at \
         https://github.com/UniversalDependencies/UD_French-GSD; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "fr_gsd-ud-train.conllu",
        size: 23_020_440,
        sha256: "3682d4276e12f422816f86ee10fa0a0525eca750a8e502fd914e18511dd7695d",
    },
    dev: Pin {
        path: "fr_gsd-ud-dev.conllu",
        size: 2_316_421,
        sha256: "3e7531b645e5bd6fe37bb619cdd8292e87f5937e64cd8627b4094324eb830e6b",
    },
    test: Pin {
        path: "fr_gsd-ud-test.conllu",
        size: 652_793,
        sha256: "5d1743c7a9ce2908943d4a430ed9a77755e2a8d7e32d42ee0f1d6a0b528f0be8",
    },
    recipe_only: false,
};

/// UD Dutch Alpino r2.16.
pub const NL_ALPINO: Treebank = Treebank {
    id: "ud-nl-alpino",
    name: "Universal Dependencies Dutch Alpino r2.16",
    tag: "r2.16",
    commit: "b323b32cdd66ef2a9c7c86ed078620e103e078dc",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Dutch-Alpino/blob/r2.16/README.txt",
    attribution: "Trained on Universal Dependencies Dutch Alpino r2.16, published at \
         https://github.com/UniversalDependencies/UD_Dutch-Alpino; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "nl_alpino-ud-train.conllu",
        size: 15_428_342,
        sha256: "a3c88336fd735d67ce53eafd6e61450a40fc6cf4eee02c60df8acd75c7b26067",
    },
    dev: Pin {
        path: "nl_alpino-ud-dev.conllu",
        size: 997_933,
        sha256: "3d6ad878d8b7b362fd756da22ae230951233be1e5596743af772150c89ba346d",
    },
    test: Pin {
        path: "nl_alpino-ud-test.conllu",
        size: 951_318,
        sha256: "f5b5000fcbb6745d42f7cccf3986bb2550f9b56a171b0d4808fb45d2df7972a2",
    },
    recipe_only: false,
};

/// UD Russian GSD r2.16.
pub const RU_GSD: Treebank = Treebank {
    id: "ud-ru-gsd",
    name: "Universal Dependencies Russian GSD r2.16",
    tag: "r2.16",
    commit: "27975702bf9507f0a244007f47345239a2d39c2f",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Russian-GSD/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Russian GSD r2.16, published at \
         https://github.com/UniversalDependencies/UD_Russian-GSD; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "ru_gsd-ud-train.conllu",
        size: 6_751_583,
        sha256: "0ae33a4bbda12697e2eb0a32af4971bda73af5852c8521744a974a316c02bab5",
    },
    dev: Pin {
        path: "ru_gsd-ud-dev.conllu",
        size: 1_052_011,
        sha256: "e3e3d4e8d1d7544e531b2b81ae64257efacc97f93276ce6d2ce52fd1de052fa2",
    },
    test: Pin {
        path: "ru_gsd-ud-test.conllu",
        size: 1_027_273,
        sha256: "f26e022329162a1c6306f76644d06f770f1572501755421165387137fe63138d",
    },
    recipe_only: false,
};

/// UD Portuguese GSD r2.16.
pub const PT_GSD: Treebank = Treebank {
    id: "ud-pt-gsd",
    name: "Universal Dependencies Portuguese GSD r2.16",
    tag: "r2.16",
    commit: "5edd2ef9dceb88058f64b823b6980bd856d7f892",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Portuguese-GSD/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Portuguese GSD r2.16, published at \
         https://github.com/UniversalDependencies/UD_Portuguese-GSD; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "pt_gsd-ud-train.conllu",
        size: 13_521_544,
        sha256: "c28bdc7ff8ddf03cba586062be0b041308da36bd82c321b437564d2b6c8664bb",
    },
    dev: Pin {
        path: "pt_gsd-ud-dev.conllu",
        size: 1_671_373,
        sha256: "0d3b6c8518ad727a26476eb682f58d5dc2bbda0307270d2a7691eb6463a074e6",
    },
    test: Pin {
        path: "pt_gsd-ud-test.conllu",
        size: 1_658_886,
        sha256: "0572d50de07593d662f7a1ff49490c7c3722612acac1d5c50916deb3d7b02a31",
    },
    recipe_only: false,
};
