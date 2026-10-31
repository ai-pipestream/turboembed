//! The CC BY-SA UD r2.16 treebanks of the languages from Afrikaans to
//! Korean (#1651), pinned as [`crate::ud::EN_EWT`] is: release tag,
//! commit, and the size and SHA-256 of each split. All are share-alike,
//! so a head trained on one is offered for download once a user accepts
//! the treebank's licence, never in a default bundle. Irish IDT is CC
//! BY-SA 3.0, the rest 4.0; [`crate::ud_banks_sa2`] holds the rest.

use crate::corpus::Pin;
use crate::ud::Treebank;

/// UD Afrikaans AfriBooms r2.16.
pub const AF_AFRIBOOMS: Treebank = Treebank {
    id: "ud-af-afribooms",
    name: "Universal Dependencies Afrikaans AfriBooms r2.16",
    tag: "r2.16",
    commit: "b9e55dcee1db036af13cae975a555471dc1139b8",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Afrikaans-AfriBooms/blob/r2.16/README.txt",
    attribution: "Trained on Universal Dependencies Afrikaans AfriBooms r2.16, published at \
         https://github.com/UniversalDependencies/UD_Afrikaans-AfriBooms; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "af_afribooms-ud-train.conllu",
        size: 2_125_517,
        sha256: "d8a9dccd8f78a2f6e2aaa0cf804b6d03c04ed5ae3301e23189b1f42fad8449d1",
    },
    dev: Pin {
        path: "af_afribooms-ud-dev.conllu",
        size: 329_117,
        sha256: "85db66a40f9316799e444346e16c844b19e6558674036397a1a23f019ef047c8",
    },
    test: Pin {
        path: "af_afribooms-ud-test.conllu",
        size: 633_142,
        sha256: "e3043f5721b99ea98497c2970155486b0ded178071a0dc8e4201b5547bd10fe3",
    },
    recipe_only: false,
};

/// UD Armenian BSUT r2.16.
pub const HY_BSUT: Treebank = Treebank {
    id: "ud-hy-bsut",
    name: "Universal Dependencies Armenian BSUT r2.16",
    tag: "r2.16",
    commit: "05013586bd6e7d45a86d3d9cd13f5956f465d460",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Armenian-BSUT/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Armenian BSUT r2.16, published at \
         https://github.com/UniversalDependencies/UD_Armenian-BSUT; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "hy_bsut-ud-train.conllu",
        size: 2_782_681,
        sha256: "12b89363338f2524125c9403ffce1492391ac3fe771fd41359d6c56bc39feab5",
    },
    dev: Pin {
        path: "hy_bsut-ud-dev.conllu",
        size: 1_399_249,
        sha256: "b77b0dba2cc7be98c7a20112c30c0b7abccc901f6dcae6dd88270da4f446566d",
    },
    test: Pin {
        path: "hy_bsut-ud-test.conllu",
        size: 1_365_203,
        sha256: "ec8f65ec06a1c6de3d74857c0daf3dd1607c39dd942453caaad3df4064f77132",
    },
    recipe_only: false,
};

/// UD Croatian SET r2.16.
pub const HR_SET: Treebank = Treebank {
    id: "ud-hr-set",
    name: "Universal Dependencies Croatian SET r2.16",
    tag: "r2.16",
    commit: "4a72e430b7b0d28668cc04d495e4cdb38b19f768",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Croatian-SET/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Croatian SET r2.16, published at \
         https://github.com/UniversalDependencies/UD_Croatian-SET; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "hr_set-ud-train.conllu",
        size: 11_585_888,
        sha256: "af0843a774c7d604789684a16934b0383f3e2950f343992d8ab16c9e08a1a44e",
    },
    dev: Pin {
        path: "hr_set-ud-dev.conllu",
        size: 1_690_740,
        sha256: "c0d3fef546efbfaa34bf5c34377a15f08b4b2f74bd3e0afcb54a9f463d9d1a12",
    },
    test: Pin {
        path: "hr_set-ud-test.conllu",
        size: 1_840_562,
        sha256: "78357fc28a5fe99f3c87eb90a014e6b5dd3770ee2d3530e70c3a78aea99384c1",
    },
    recipe_only: false,
};

/// UD Danish DDT r2.16.
pub const DA_DDT: Treebank = Treebank {
    id: "ud-da-ddt",
    name: "Universal Dependencies Danish DDT r2.16",
    tag: "r2.16",
    commit: "427ed59a0a3c7af15f7a7687db9355051ff93b70",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Danish-DDT/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Danish DDT r2.16, published at \
         https://github.com/UniversalDependencies/UD_Danish-DDT; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "da_ddt-ud-train.conllu",
        size: 5_137_462,
        sha256: "65730e35b07d1e944ee9ae1d344b5c84746cf0aeadf5faf0550059b4ded86fb6",
    },
    dev: Pin {
        path: "da_ddt-ud-dev.conllu",
        size: 660_002,
        sha256: "d714b22776ad60fbdb6e4ffc15a5c3ffc162705617567d3cf2b09cc6b51e800a",
    },
    test: Pin {
        path: "da_ddt-ud-test.conllu",
        size: 639_330,
        sha256: "2232ee43dcd35dba32b476534d230edf1aa26d4bd830813f452fd829c276f737",
    },
    recipe_only: false,
};

/// UD Georgian GLC r2.16.
pub const KA_GLC: Treebank = Treebank {
    id: "ud-ka-glc",
    name: "Universal Dependencies Georgian GLC r2.16",
    tag: "r2.16",
    commit: "30df1e56030073d2a1760692b8f0a24b4303e027",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Georgian-GLC/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Georgian GLC r2.16, published at \
         https://github.com/UniversalDependencies/UD_Georgian-GLC; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "ka_glc-ud-train.conllu",
        size: 5_741_477,
        sha256: "883bc81efdf955c0519dd97d1d4c8695ba7193fa6d37b6be61d60ad537ddd27e",
    },
    dev: Pin {
        path: "ka_glc-ud-dev.conllu",
        size: 1_361_686,
        sha256: "3c4de8864ec5b54da6a46ff34b1739da8697d8e74c571aef54a6153d097467ef",
    },
    test: Pin {
        path: "ka_glc-ud-test.conllu",
        size: 1_422_174,
        sha256: "04084d74d6b72f86b4dc276c8c32dce6a3671ef0ed1510260f51e0c3ac0a9ab1",
    },
    recipe_only: false,
};

/// UD Icelandic IcePaHC r2.16.
pub const IS_ICEPAHC: Treebank = Treebank {
    id: "ud-is-icepahc",
    name: "Universal Dependencies Icelandic IcePaHC r2.16",
    tag: "r2.16",
    commit: "f6e7a4f56562d3570d783e37a83f133950dabd99",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Icelandic-IcePaHC/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Icelandic IcePaHC r2.16, published at \
         https://github.com/UniversalDependencies/UD_Icelandic-IcePaHC; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "is_icepahc-ud-train.conllu",
        size: 61_400_654,
        sha256: "9da7493d832d1e6a84218fd6f9ce57316a000631a1d6d318b30362ca7a8bbbce",
    },
    dev: Pin {
        path: "is_icepahc-ud-dev.conllu",
        size: 11_860_634,
        sha256: "073b73ffe209068098a7fb5425cfc216eebcc189f43e7717d9b0e5ee626fa0ee",
    },
    test: Pin {
        path: "is_icepahc-ud-test.conllu",
        size: 11_900_331,
        sha256: "be4b5e52cdcd1c8ff4ac0c622306416ef5a47a8a9352902cd9e33626d2317d57",
    },
    recipe_only: false,
};

/// UD Indonesian GSD r2.16.
pub const ID_GSD: Treebank = Treebank {
    id: "ud-id-gsd",
    name: "Universal Dependencies Indonesian GSD r2.16",
    tag: "r2.16",
    commit: "a1222f691d5437be4c7e8db11a2c31e68f1a7d01",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Indonesian-GSD/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Indonesian GSD r2.16, published at \
         https://github.com/UniversalDependencies/UD_Indonesian-GSD; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "id_gsd-ud-train.conllu",
        size: 7_675_206,
        sha256: "dfd2c852a4f04b2e66c5617d5d3d478437713ec1c621cc64995e084a74f76623",
    },
    dev: Pin {
        path: "id_gsd-ud-dev.conllu",
        size: 993_314,
        sha256: "14bd353e18870dbb11fe61da558c91a5200977873b252698cd8e9fecca3eecf0",
    },
    test: Pin {
        path: "id_gsd-ud-test.conllu",
        size: 925_288,
        sha256: "6434c1d96cbf0637232e3250b586434603ec7afbd1da846160bf187f6332f136",
    },
    recipe_only: false,
};

/// UD Irish IDT r2.16.
pub const GA_IDT: Treebank = Treebank {
    id: "ud-ga-idt",
    name: "Universal Dependencies Irish IDT r2.16",
    tag: "r2.16",
    commit: "f5092f976ab3836676f1685b48597f267f738aa5",
    licence: "CC BY-SA 3.0",
    spdx: "CC-BY-SA-3.0",
    terms: "https://github.com/UniversalDependencies/UD_Irish-IDT/blob/r2.16/README.txt",
    attribution: "Trained on Universal Dependencies Irish IDT r2.16, published at \
         https://github.com/UniversalDependencies/UD_Irish-IDT; used under the \
         Creative Commons Attribution-ShareAlike 3.0 Unported licence (https://creativecommons.org/licenses/by-sa/3.0/).",
    train: Pin {
        path: "ga_idt-ud-train.conllu",
        size: 6_631_088,
        sha256: "105d5a054adea2371313ae3448f06381eb4ba250a2493a242918069384fb6a71",
    },
    dev: Pin {
        path: "ga_idt-ud-dev.conllu",
        size: 682_411,
        sha256: "153da60e0283f426b9334e654a23651272c1c7ace1d9a64c9d255554442ee5bd",
    },
    test: Pin {
        path: "ga_idt-ud-test.conllu",
        size: 687_169,
        sha256: "cb9aa443f1a9a227a4e398e4542c60e06a8faa465b0979213786953f3eee426d",
    },
    recipe_only: false,
};

/// UD Korean Kaist r2.16.
pub const KO_KAIST: Treebank = Treebank {
    id: "ud-ko-kaist",
    name: "Universal Dependencies Korean Kaist r2.16",
    tag: "r2.16",
    commit: "888f855c7bcf1e39291246744f4e98f6bda6742e",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_Korean-Kaist/blob/r2.16/README.md",
    attribution: "Trained on Universal Dependencies Korean Kaist r2.16, published at \
         https://github.com/UniversalDependencies/UD_Korean-Kaist; used under the \
         Creative Commons Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "ko_kaist-ud-train.conllu",
        size: 34_505_277,
        sha256: "6f99de7e21b8119022831afda52643dd8ff60ea386f2679c938dd7f652725d94",
    },
    dev: Pin {
        path: "ko_kaist-ud-dev.conllu",
        size: 3_010_966,
        sha256: "a1ce2dceee65683c2df3b0cce96c83d01e5ff67756060b998fe01d8bc8ca4faa",
    },
    test: Pin {
        path: "ko_kaist-ud-test.conllu",
        size: 3_324_246,
        sha256: "fd94dc89afb01d1f7f340a46d567e0f27ae6903a70d7b6f650b49f7427f83b97",
    },
    recipe_only: false,
};
