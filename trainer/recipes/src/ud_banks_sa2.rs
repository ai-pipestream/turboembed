//! The CC BY-SA 4.0 UD r2.16 treebanks of the languages from Latvian to
//! Turkish (#1651), pinned and classed as in [`crate::ud_banks_sa`].

use crate::corpus::Pin;
use crate::ud::Treebank;

/// UD Latvian LVTB r2.16.
pub const LV_LVTB: Treebank = Treebank {
    id: "ud-lv-lvtb",
    name: "Universal Dependencies Latvian LVTB r2.16",
    tag: "r2.16",
    commit: "f2d46c53458aeede94d08302c1b8c8b03a02ef0b",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Latvian-LVTB/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Latvian LVTB r2.16, published at \
         https://github.com/UniversalDependencies/UD_Latvian-LVTB; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "lv_lvtb-ud-train.conllu",
        size: 29_688_519,
        sha256: "329f2de69468448c600677300c98b04b5f32ec811654d90a1619ce4e209bca1c",
    },
    dev: Pin {
        path: "lv_lvtb-ud-dev.conllu",
        size: 3_943_332,
        sha256: "c4c746386a6fa678ef1c40702f5ae2ef28870f2ef2d68025f6738d5e129ba1ac",
    },
    test: Pin {
        path: "lv_lvtb-ud-test.conllu",
        size: 4_262_981,
        sha256: "3e7dc6125eb58f11d9c284b21d484ccf3904f841a22f17b8f70d9df325118261",
    },
    recipe_only: false,
};

/// UD Norwegian Bokmaal r2.16.
pub const NO_BOKMAAL: Treebank = Treebank {
    id: "ud-no-bokmaal",
    name: "Universal Dependencies Norwegian Bokmaal r2.16",
    tag: "r2.16",
    commit: "9cac023138b2fc1ee8a6e28a25b2f7d308fb3aea",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Norwegian Bokmaal r2.16, published at \
         https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "no_bokmaal-ud-train.conllu",
        size: 15_882_804,
        sha256: "caa990b444d11030df842d39f2b4bda271beb90a93deda0a56406fa061535b03",
    },
    dev: Pin {
        path: "no_bokmaal-ud-dev.conllu",
        size: 2_371_164,
        sha256: "1c803c3f90e510ce995c5f9108e240f283c34025333a61281cb90fdf2e82c83a",
    },
    test: Pin {
        path: "no_bokmaal-ud-test.conllu",
        size: 1_949_600,
        sha256: "b45e4809d38ea560f0b164023b0f4c6d8f0d7731f66c4df129917bb06d1676f8",
    },
    recipe_only: false,
};

/// UD Persian PerDT r2.16.
pub const FA_PERDT: Treebank = Treebank {
    id: "ud-fa-perdt",
    name: "Universal Dependencies Persian PerDT r2.16",
    tag: "r2.16",
    commit: "1184f0765a335d9fcb06ba6e34f85e7c8f09cf0b",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Persian-PerDT/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Persian PerDT r2.16, published at \
         https://github.com/UniversalDependencies/UD_Persian-PerDT; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "fa_perdt-ud-train.conllu",
        size: 31_417_277,
        sha256: "f7a32f29bb7a3305cae8556132605420c7f13b919c165f42fe77555539aaaef6",
    },
    dev: Pin {
        path: "fa_perdt-ud-dev.conllu",
        size: 1_730_754,
        sha256: "a8394fe73e9f78ef57e41c32483609557d9aee175fefdca6455efd5bda7ce005",
    },
    test: Pin {
        path: "fa_perdt-ud-test.conllu",
        size: 1_680_380,
        sha256: "c8a7738b9cdf1e9b5529b1039a754de3e8d4c4e52cbb316c0c8f15b45971d468",
    },
    recipe_only: false,
};

/// UD Romanian RRT r2.16.
pub const RO_RRT: Treebank = Treebank {
    id: "ud-ro-rrt",
    name: "Universal Dependencies Romanian RRT r2.16",
    tag: "r2.16",
    commit: "f4cf72215ebf330d20ab73dbcb9effc114adcc80",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Romanian-RRT/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Romanian RRT r2.16, published at \
         https://github.com/UniversalDependencies/UD_Romanian-RRT; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "ro_rrt-ud-train.conllu",
        size: 14_555_874,
        sha256: "a4d1b848cc813ebb1a75b8c51b6b85f41e08e8ca179785802028f311bb4d0a4b",
    },
    dev: Pin {
        path: "ro_rrt-ud-dev.conllu",
        size: 1_347_209,
        sha256: "6e940a2dfd3c1ded9859f2fa5c9b1edb687cd0b4b9d99c6496f7cc729b8742d9",
    },
    test: Pin {
        path: "ro_rrt-ud-test.conllu",
        size: 1_285_762,
        sha256: "9084ce9ae5f43d25ff39e66f42f3d2e41e85030891beccb1eb70fed444fbe3f4",
    },
    recipe_only: false,
};

/// UD Serbian SET r2.16.
pub const SR_SET: Treebank = Treebank {
    id: "ud-sr-set",
    name: "Universal Dependencies Serbian SET r2.16",
    tag: "r2.16",
    commit: "ce79bffd4ff94cd910099b2820f7b069af6c9d69",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Serbian-SET/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Serbian SET r2.16, published at \
         https://github.com/UniversalDependencies/UD_Serbian-SET; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "sr_set-ud-train.conllu",
        size: 5_637_628,
        sha256: "bd4d41f7e807a46a92d1c0e61531809356582dd96d47eb38189f88ca1943cd1d",
    },
    dev: Pin {
        path: "sr_set-ud-dev.conllu",
        size: 910_076,
        sha256: "a9ce898b53e5f7fa3d314fd703f08398cb2cf4045e1110b6542970c6a46508bd",
    },
    test: Pin {
        path: "sr_set-ud-test.conllu",
        size: 867_534,
        sha256: "83c62b07a9375fcc4d3d22f67446f40f0c25994c2322b6aba904e88df8f7d1d1",
    },
    recipe_only: false,
};

/// UD Slovak SNK r2.16.
pub const SK_SNK: Treebank = Treebank {
    id: "ud-sk-snk",
    name: "Universal Dependencies Slovak SNK r2.16",
    tag: "r2.16",
    commit: "98dc19ff85e2aeff28468f2ec6397399c6156242",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Slovak-SNK/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Slovak SNK r2.16, published at \
         https://github.com/UniversalDependencies/UD_Slovak-SNK; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "sk_snk-ud-train.conllu",
        size: 7_608_348,
        sha256: "e1a92cc7e39e7fd2791fa5a3f30e024278e94640b631bbc1d966ca22acd49844",
    },
    dev: Pin {
        path: "sk_snk-ud-dev.conllu",
        size: 1_201_051,
        sha256: "2c10cd4b9b6e64d1ddb2c35d09994ac99da29340ebafbf6598c497a74cc270c7",
    },
    test: Pin {
        path: "sk_snk-ud-test.conllu",
        size: 1_214_078,
        sha256: "5e0f9155c62422778586aa819e0dea13e4f4cf8ed1894019f87ab751ac2de4d0",
    },
    recipe_only: false,
};

/// UD Slovenian SSJ r2.16.
pub const SL_SSJ: Treebank = Treebank {
    id: "ud-sl-ssj",
    name: "Universal Dependencies Slovenian SSJ r2.16",
    tag: "r2.16",
    commit: "2071ccb723670950196203f35588602ef3315b32",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Slovenian-SSJ/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Slovenian SSJ r2.16, published at \
         https://github.com/UniversalDependencies/UD_Slovenian-SSJ; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "sl_ssj-ud-train.conllu",
        size: 16_743_285,
        sha256: "033c3c049b118765b995aab9464abcae10edae56fcbc3614fb32abcc7124bbff",
    },
    dev: Pin {
        path: "sl_ssj-ud-dev.conllu",
        size: 2_054_820,
        sha256: "ddca0bed60dae3b6fb82ee2518760ae34235fd92d187f241153645a96d18d423",
    },
    test: Pin {
        path: "sl_ssj-ud-test.conllu",
        size: 1_989_725,
        sha256: "acfaff87b7d9f21c34240817ce187a515ac04ae6a6a81f896d68ea168572722b",
    },
    recipe_only: false,
};

/// UD Swedish Talbanken r2.16.
pub const SV_TALBANKEN: Treebank = Treebank {
    id: "ud-sv-talbanken",
    name: "Universal Dependencies Swedish Talbanken r2.16",
    tag: "r2.16",
    commit: "3fd9c4123f155560ebb68c324b346fa9fa33991c",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Swedish-Talbanken/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Swedish Talbanken r2.16, published at \
         https://github.com/UniversalDependencies/UD_Swedish-Talbanken; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "sv_talbanken-ud-train.conllu",
        size: 5_907_221,
        sha256: "fcc4bed5885e3d95f34a0e1bf813783135d38ae76d5822c3530b371fe78fc35f",
    },
    dev: Pin {
        path: "sv_talbanken-ud-dev.conllu",
        size: 863_260,
        sha256: "2e30e94a73a6db89b09fe80767df0c8c1194a48525879126a793015e868f8265",
    },
    test: Pin {
        path: "sv_talbanken-ud-test.conllu",
        size: 1_802_238,
        sha256: "eb3950dfb820d13634bed35c1d0f69872851754e0b19af1a9705ce6978580434",
    },
    recipe_only: false,
};

/// UD Turkish BOUN r2.16.
pub const TR_BOUN: Treebank = Treebank {
    id: "ud-tr-boun",
    name: "Universal Dependencies Turkish BOUN r2.16",
    tag: "r2.16",
    commit: "cfa62ac55a426d596700eceac436944e5cf0a967",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Turkish-BOUN/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Turkish BOUN r2.16, published at \
         https://github.com/UniversalDependencies/UD_Turkish-BOUN; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "tr_boun-ud-train.conllu",
        size: 7_919_910,
        sha256: "373fe39e3acdba3219521f54fa5ae768851fbb4bf9a31079ad1ca6590410650c",
    },
    dev: Pin {
        path: "tr_boun-ud-dev.conllu",
        size: 967_078,
        sha256: "f6d1ed6e3b57b4e658177eb16089e9fb616bc68fb110eb0894c2b409e41b5c76",
    },
    test: Pin {
        path: "tr_boun-ud-test.conllu",
        size: 955_648,
        sha256: "4f74fc0f9efdd1285d781f3efd1e98cbf39cf4433df9eb4dc14162e8ed36773b",
    },
    recipe_only: false,
};
