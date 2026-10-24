/// A checked in file the generator loads from the fixtures directory,
/// `analysis/goldens/data/fixtures`: a model trained for the tests or
/// written by the Rust trainer, a
/// dictionary, a WordPiece vocabulary, a frequency lexicon or a file of a
/// MeCab dictionary directory, and the SHA-512 of its bytes. `file` is the
/// path below the fixtures directory, with `/` between directory and file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fixture {
    pub file: &'static str,
    pub sha512: &'static str,
}

/// The fixtures, which are all the files of the fixtures directory. The
/// file names are the ones the generator's `Runs` class opens there; a
/// fixture in a subdirectory is a file of a MeCab dictionary directory, of
/// the feature generator descriptors and their resources (`featuregen`) or
/// of the name finder's dictionaries and patterns (`namefind`) or a
/// hand-written Hunspell dictionary (`hunspell`), and `Runs` opens the
/// directory.
pub const FIXTURES: [Fixture; 110] = [
    Fixture {
        file: "en-abbrev.bin",
        sha512: "759ab1defc61351f53cdd82a26e708789a225feb6e1ec0cfd6e2e54f35f67c14511f44d0eb73216f9d4c89c0fb26ea28fd6931a7c4f851d09ca487d0a39700fd",
    },
    Fixture {
        file: "en-plain.bin",
        sha512: "8da69d821956999ff898cf23963145cc134beb2bb59aa1ccc5e93d90b04895057057d80baf0e8c726bf28dea47021e51eab43689c8dc5fb78c99f19949c93ad8",
    },
    Fixture {
        file: "en-plain-rust.bin",
        sha512: "ef8684db6573d7096b72db1d520fe7181c8ee058d2c9c1abe140556d36da18d1693e63c2f84faf8f427fdd80b0e0f4bd47ce6a92a7aa53233b0b74c403cff736",
    },
    Fixture {
        file: "en-plain-qn-rust.bin",
        sha512: "23f36eea3d663b3791fd6ee4c5f90e51f7298f4488c815f27b440a2edcac760cdcb6fbe4da87e1762c09c7f6f5db981cfb4657d0f0a6171a4f6cfaebdb9eda0f",
    },
    Fixture {
        file: "en-plain-perceptron-rust.bin",
        sha512: "5882f58f6cea311ff8b2043509992a31a2870e75fe7fc3e08a96ca7f441e4ef58ee8d851ace176942987fa860c7ca37cab6bbcd8c38b5b67abfa57918789b915",
    },
    Fixture {
        file: "tokenizer-abbreviations.bin",
        sha512: "361484f0a67e129fe6f19908704d23e83c8a2bf4551a5f7eec3b40b9082a66b7026d429233c11639b35bccf042d347d79222cfed62685563f2a926cb8a2460d4",
    },
    Fixture {
        file: "pos-fixture.bin",
        sha512: "70bd00e5c7baba5552e311ad002aa382fd57aa2bc5248ffe6008476546df3799a4e8578c3e6e90aa51d4a766857d07fa7003d395151d292ede3804fc308d5d32",
    },
    Fixture {
        file: "pos-fixture-rust.bin",
        sha512: "7ce6a0d76f18242966640050d44a94f0f8ff496b2a518a72c0e191a964d321588bda11782be6ae3d09f62fec387f01db32bdeb87eb64a741f07bb10359e57328",
    },
    Fixture {
        file: "lemma-fixture-rust.bin",
        sha512: "4fa4c03da55e729af919a2d76a67db49ac0269d073ac05a5af3da9fd8d67ea8261765d97a3c5e4ca0d22b251e4dc01401f432d6a394d6a6d3fd4b06fecb41402",
    },
    Fixture {
        file: "ner-fixture-rust.bin",
        sha512: "c9e30b48571cd106396a9f84c0692216b5934cd7cae4ec277ceb867b93a5b9ed4285349d48c2e3d7c508bb824697a2e68055b54817b0f9fbc06e011c510ef070",
    },
    Fixture {
        file: "langdetect-fixture-rust.bin",
        sha512: "631c804500a2a2cbd3c1b88febdf21332487760abc40efcee217159e3e694a01dbbc82ee9d5102dccc9f5a889775aad4c93efedf3621df9d4622448a1d58ae99",
    },
    Fixture {
        file: "tags.tagdict",
        sha512: "cbd3dee98987711b48b42967df8e6e3e8400dcba70336fa2c83bfdba03a8c42b3bd7d5f6f0c80c2f35dc01173795792e117489f11fd23f03f9fcba3fa453417d",
    },
    Fixture {
        file: "lemmatizer-dictionary.txt",
        sha512: "e642dfeb8d0b9f45b1853808878afa8a0a3cbe9f738a0646b9e4cafdaf8bb8c915aa548d72833274e877a2b57118906557512d9e6fe22b320c7be87fc964e707",
    },
    Fixture {
        file: "chunker-fixture.bin",
        sha512: "be8e0bfd341644705a652b85b978e4138056922ddb9caea1ac9a2395f0648e55ad1f1b035d90f6febc0f057f832cecac377f397552fdbdfefadaa27cfa232d6a",
    },
    Fixture {
        file: "chunker-fixture-rust.bin",
        sha512: "71caec6c87065f0872d0ef255ed95443b0c675acf273c2d19a1aefd6c006f1ab99846bbb28b56d2e78abcb3f78074d00b1135a6dffc4e14ed3957101cf80ee44",
    },
    Fixture {
        file: "english-dictionary.xml",
        sha512: "e824b14f66036b2703cbbd3af6a893f7f07cf282c9e8021ae7be38f10da305a87aef48b129923f3973c78f495c2ed073adcad68c90c2f7d94bc6d32870839164",
    },
    Fixture {
        file: "en-head-rules.txt",
        sha512: "4924379ef7ca9376c6953e166512a556006e069e9ba319dc941a74342e3b46aa64313014d5a801030d5867dc6ea4be9208b75706579840bbcc12fa00aab0cca4",
    },
    Fixture {
        file: "es-head-rules.txt",
        sha512: "b01d1e26e3a298064a86d0222459e1b00fcd5ba4e1bb3859245d86ecd458feb793bc208505100e8150f5d1f317762dfe0b01842a7d718245463e77a6a310e900",
    },
    Fixture {
        file: "es-regex-head-rules.txt",
        sha512: "d75363b131a71c1b6fd243e9f339ee97393fe36a9ed89d857c49d89da793dc9b1e48b92ca0e67fb65c04305fba81d17754b896b2cba90a166348c00feff19a6b",
    },
    Fixture {
        file: "gazetteer-geonames.txt",
        sha512: "69e2ef02c7ae86751635496c5b46895854cd95977fb4e6a6756df9aa69dc6bbf8487983af8803650627067167e3f5212ef0c9adb6a9b225e3a99837cfeb9a95d",
    },
    Fixture {
        file: "gazetteer-overture.txt",
        sha512: "851543d2634344f6764f3210f18a8da8f1e5400a637b312960e4c0eb1e3e6419af3c22f4b2917753f4b141158d15a1500e5cf0491b1f7211f13ecc352171db57",
    },
    Fixture {
        file: "gazetteer-user.txt",
        sha512: "68f44c6be8bbc9c3d3d697681eb121f0593087066eaf3dba7b5e6a44780f0c05bcfe79894c8cc26d609459c8362ab1f65d8540ba4799468d69356746e72aac49",
    },
    Fixture {
        file: "gazetteer-suppressions.txt",
        sha512: "50db213abd6c0dab3409a1e8ad77aee22bd05920f2fb23c393fccec01f342d128a336761acb8bb3df401549ad319cd6551ba3b06558e4d4cf0604025d1ef39d6",
    },
    Fixture {
        file: "glossary-terms.tsv",
        sha512: "9356e730339664c7fcbf3cd6c0aef6287ac07f0db2e13d03927a9c5f057305965cb13a1b0df0d5f5e5072a9dded43c07f406cfad9fcad5416d07fef80fb0623a",
    },
    Fixture {
        file: "en-detokenizer.xml",
        sha512: "d071b56f77ae5b6aff0a591c1cf7e601a4007b5fdbcc0bd3fc31e8c84e95ec232df9765a70bfef786f75dbefcc4cad7d053b34b00ecc805507192a5fddc81260",
    },
    Fixture {
        file: "fr-detokenizer.xml",
        sha512: "9c6c9165060e7ff2801b9cb5e6c778e1e93bc68fc4c3fcd4d44b23dbb3254f048e7c9f4d036e1d1bacac6f337f105105990956baf3775f41bc0d97045a25346d",
    },
    Fixture {
        file: "latin-detokenizer.xml",
        sha512: "8de094b90fd1856620ddb7c0d361833055c58d945b13f8263d027797de9510f6e1cd4b73fffaa6693dab807a11b618ffe8d6ec060d2d4fc40cdc83a91afaa32f",
    },
    Fixture {
        file: "pt-detokenizer.xml",
        sha512: "36be57752d28262efa8fdb0b0d4d997a28f00eb32890419210911303459a1b3a0ccecc29281f5a415a80b20fb75ac973e52237605960bd7669e2409738f19555",
    },
    Fixture {
        file: "quotes-detokenizer.xml",
        sha512: "d8444302db26d336e30283fa9caf62250b55c671e775629137499b16615e0d2b6686379cff4d1baa34157deebe61170ebbf9a28d9e0f09622f1aefd7bcf285f0",
    },
    Fixture {
        file: "all-minilm-l6-v2-vocab.txt",
        sha512: "f51f06aae7f580a88998f6f7f24b52495c8d3d289fdbdc21231c05f3a8965783074d95c17b819186f9a63b622280e8a051105a2161cd0d153fa57db7a0aba9f9",
    },
    Fixture {
        file: "zh-unigram-fixture.txt",
        sha512: "75fe2ae460f0a048e0978dfcf54474752aad6ecde6b64ad86bd59729f3dbce829043469e17ad929fed559bccdc1db72a358e706527316e8faf733c63ff144011",
    },
    Fixture {
        file: "jieba-dict-subset.txt",
        sha512: "365d5a67c098b362ec12c166a9b438031fd9ffa5947113bc04e2fad1d623918a4303cb0f9657e28f4e619fae502f3bf4c412285cfe5d6a128439a614ea60e80c",
    },
    Fixture {
        file: "zh-unigram-modes.txt",
        sha512: "d07069e2b828406873f37f056ca71e2175d8ecbd922d2a34c356f6695c0975e03830b7fa8000ba8476312514dca8992b24d848a82abd3c4ce677b0fdf1fe6f9a",
    },
    Fixture {
        file: "mecab-fixture-ja/A-nouns.csv",
        sha512: "3514d1ce9d28d24eafd14406987352f004a85a5980e86971475401e2c87610fbf383e207498a9067bb37f260dffb3fb03abbedd353dbc08b50d60d24a0aa8847",
    },
    Fixture {
        file: "mecab-fixture-ja/b-others.csv",
        sha512: "1ee8fee5fbed1177dfea712201212a0b6732220f9047a537f8536f31906af2d75591b5d4533f21112999142230a44594654eedc85071418dca1df2d1cd777b21",
    },
    Fixture {
        file: "mecab-fixture-ja/char.def",
        sha512: "a1827cdf76d1ad05a1a82d4960a37c5bb2251ac7fadad8b20627fac45de2d758b4d6ecb2a3cc6313e73df4c929ff9d09a35d1617346a0fcfc438d95d620089c4",
    },
    Fixture {
        file: "mecab-fixture-ja/matrix.def",
        sha512: "5561b2fa77b12841439fa9f81f247749ebe21634e7bc17ed47c1f94698b4b15e0b9ee433a276edbef3cdf3fd7b19b84f74f478728f8e69551ceb2a93ab4b6286",
    },
    Fixture {
        file: "mecab-fixture-ja/unk.def",
        sha512: "b480bc0b9a28f0876ea756bacecf4f9970d72c33f170fd09279d3f9c92c43b831686d01ed98bd541fb84266f752776cba30061607497e59b27d23884efa49832",
    },
    Fixture {
        file: "mecab-fixture-ko/char.def",
        sha512: "771c2403dfb33b08d5f2494463bad93465585d254b532ffadde7515703d01ea444deb05a57371dd459d8d4e9eef25b050e2be8a6c7470c052b86e8f1d7b09900",
    },
    Fixture {
        file: "mecab-fixture-ko/matrix.def",
        sha512: "c11d28694008c74e84f9fe2f3c42d25106fb496851bbe92b166af575922ff4223456cfe2ca3692fee1863a8a3539859ab772e9b9868bea04df41e43cc1ca2575",
    },
    Fixture {
        file: "mecab-fixture-ko/unk.def",
        sha512: "260314c1de3d187ac3fa8ef5a35e354d318af678de099f7a54d3021694cc374b336e1f223bfbde0dd2969aba9ea2445fc5f549c265c394977aad942cec396374",
    },
    Fixture {
        file: "mecab-fixture-ko/words.csv",
        sha512: "969a6bcd215a61dd8bc7799f69a7b1a6aeb5975d8202bba4c982e39271e208df8fe1b893dac9c73f5d508901ad8959c66e3c874abc8219db2afa21dc6d0471ce",
    },
    Fixture {
        file: "mecab-fixture-partial/char.def",
        sha512: "45ae8dfcf2398f038d40f0910410dccdb74a2a1ba969bc718c976e675371c72bd0efb3666b041b201c3a38a27855cbd50a7bfd16685b0764ea87e2104c7b0db4",
    },
    Fixture {
        file: "mecab-fixture-partial/matrix.def",
        sha512: "d7d297def9b3076ff213fba877138bef306126f77e2a3d962ab3be0625e1b4bbd018207610edde710251e1e86e0d6aae5fd21d1b844edcaa509a809b7c957832",
    },
    Fixture {
        file: "mecab-fixture-partial/unk.def",
        sha512: "677ede432bd5d3d939221a8cc56f36a0dcbad64705b9b02661ef57d10c7ec6686e44201dc34f5cd028b87e9ae1c1a50ec6e552d0461f13df8ec35adcd609e013",
    },
    Fixture {
        file: "mecab-fixture-partial/words.csv",
        sha512: "7a0d6be9beaf23fec933e7142dc79c20508e9ef7414181996203be9f3ab5b1cd5cca26be772b523e0ff4d0bf0e895f2c33ec9c615957b9d85cfd271257614d20",
    },
    Fixture {
        file: "ner-bio-fixture.bin",
        sha512: "429f451debd80c724c96cc08e7fc83ecc56047a9ca604f3ae98e9a61a5d55c972b786448dc6c5650130117144488adc61edc31994b9d8b8f84072ce6f01ebe89",
    },
    Fixture {
        file: "ner-bilou-fixture.bin",
        sha512: "b19000b06acefcfd6a1293f91353412f679d3499ef12ce4c909e92d15179dc1b51931796a4b6e50ac8eff7c45bc8a2c3cb838516ae0acaca289e0e8dc8e221a4",
    },
    Fixture {
        file: "ner-qn-fixture.bin",
        sha512: "26740433be9cb394e721cda1422e25b4daa25f13b0a2b9bfe66c2aacd0b352910c46f99e26377d2c0b46c3a14253b6ea8f34fa7bcc4c00c9711fcc7bfe469b85",
    },
    Fixture {
        file: "tags-cases.tagdict",
        sha512: "5ea821ccd9d28e5d10fbf3c05a56847f61c59c50f5dd52b9bf037e5612ab72c079d5b988180ee64ad95dc0a0dce0750457aa020b74a6287240bb69e1673cd64d",
    },
    Fixture {
        file: "tags-exact.tagdict",
        sha512: "6f0d506cb466377bc9e583c25365ad11a9dfd14dafc5cc0d3c6ac18d3442374e697c313cc0fdec7df62a7306671baac2e6da35c4ee969c04e795f167be109eba",
    },
    Fixture {
        file: "featuregen/pos-default.xml",
        sha512: "dc3dfbfaefc8ef5ddc096a0b64605a5ab3af873fb9f5fc8fca94e9a781e35153732fb47558a8cf276bf198ba008103a8981a770870ca701e2284322e5e0ed314",
    },
    Fixture {
        file: "featuregen/ner-default.xml",
        sha512: "ebf4d1eb4746ee7644c58ef01730facbe4354cfdf52708c6acfb431d238a553d153839f53a5f1f40fdfbe659fb01b025a2c08be3784af63ef09edff27e354105",
    },
    Fixture {
        file: "featuregen/everything.xml",
        sha512: "383fa7feb02a3d443f50f046358fcb520dd909271c681ed601ef3d0e845df94cb128304ae0855460ba521739c48164ba5270fae3f7e7bc47f0c29bc81dd955b8",
    },
    Fixture {
        file: "featuregen/nesting.xml",
        sha512: "ca01053da8a8ca8f27b52dc7d7199b9813f4987baa294032ae983fb5b1212250b61268162902c3bb8d4974f6cf496557642f91afdf8e90575a766779d3a21f0d",
    },
    Fixture {
        file: "featuregen/brown.cluster",
        sha512: "0e250a649cfd9444563e76085d999081ea97d359d2e861d3f8d4f955a555338ecf3d9a9fca91d88eecb0366a237207d484e27ab5898ee814d7c34e1f31202c28",
    },
    Fixture {
        file: "featuregen/word.cluster",
        sha512: "f6d195f9f881b735fad37d3477a11fe3c5eebc6263ec62c2f833ed8b0c0211ed973f03746077085426fe6c7e37079c1bb8b5fa6cab9a356bf40eb8a56c25ff27",
    },
    Fixture {
        file: "featuregen/names.dictionary",
        sha512: "d1d6630c2689154da78e901d4b3b0849ae20859e8948c66e9851576aa5ec3f72ffffbae34518516740a345e6afa31734bac58c44469766f62bc1156c8d78c57d",
    },
    Fixture {
        file: "namefind/places.xml",
        sha512: "8aeff78b3dd35b9dc064a9968b7ab98e99f9b312ce615e23ff1c2424d466c6b66d7bf8a4651c6f8996fa69c0443276f8ca6a68467dc9b848bced0919f9a788b8",
    },
    Fixture {
        file: "namefind/people.xml",
        sha512: "38baf8b0ce7d9d4bf4e27d985e02110823f60b38362a945963cc7dfde74ebd7c871e32260b536338337b8629a04526fee57760bb45bbb9972404f2b8f1e44bde",
    },
    Fixture {
        file: "namefind/overlap.xml",
        sha512: "9257b73ff0e915c78d641174e1ecc75b5f76077ac8a77831745b5fb89b3b47672aefb307f45c954da43490962f3e212687aaef5b970009a92cd2b78eeaad1086",
    },
    Fixture {
        file: "namefind/empty.xml",
        sha512: "a2f1d40b1336f5d68bc719625b38aebf9d4c0264b5b4e605699fcba812e2c5851bc1d54e9a7d87ef3ba1a8875c6d116baa7f0bd714cd7869d58d15b18b706d25",
    },
    Fixture {
        file: "namefind/patterns.txt",
        sha512: "99205861cc0ba0b89776d162b507341b40a916a9959858250473dd09be9e9d12fca01f3618f993f4313dd0f9096a9889b478d879980eeedae85ea5f96e164987",
    },
    Fixture {
        file: "namefind/regex-boundaries.txt",
        sha512: "b80577f1fef83a2020d0e2aa506cb9ee412041b87d26d921543c986356fde51f2dabfe8a1455ca881bc45a6c856ca08a0527895f82137178da895584f1688362",
    },
    Fixture {
        file: "hunspell/affixes.aff",
        sha512: "ccd685453c3cfd7261c575c61c9432eae526e81e471562d503f80464c94cd1072e69dcc0d6e1df71106bbcd5982458f9f8df59afd048787cd6781ec5cb17cb4f",
    },
    Fixture {
        file: "hunspell/affixes.dic",
        sha512: "eb7fbf055253cb690992e15003930143bb244c06ad1b49b4766950cc5ebceb3b2b2d361c5c9533e4486c0246b6959686b30c5c011c3091527da0d74be307aeee",
    },
    Fixture {
        file: "hunspell/bom-crlf.aff",
        sha512: "7adcf18c5802c807f864b3c6f9515a7583c81b2709010fe062f9771a29c85f27c7764f7a18802a03d73d934e9d5d569447f1c0983822f57ea7da8d78186c772d",
    },
    Fixture {
        file: "hunspell/bom-crlf.dic",
        sha512: "33984facc0489af2d8ce1cdee48bf57903a720502b3f8f9e6638044c2cf45637cb4ff8b4cd4a1eb86a30b51a7f131e41016c5fb961534e8b9566df26827753bc",
    },
    Fixture {
        file: "hunspell/complex.aff",
        sha512: "c5f24f865986104940fdef8d9c11a8962aee9ab4f83f6d8283f64075ef46ba65c86836e91b37d1870b0c278a056d68d7f714d62fffafc8c1522ff7be1e4a6ce4",
    },
    Fixture {
        file: "hunspell/complex.dic",
        sha512: "99b6056e7ee47a469c6dab3a48d8fe019669a3e73a38e8cb011c3e606ced373041da636cc8f87b9c4af3ed36af3d8ad1ce3d43502f10fd73394d0b3656185062",
    },
    Fixture {
        file: "hunspell/compound-rule.aff",
        sha512: "af2cd9d6433a08963bb489c9851b88cd9618e88160c7806973c5048babcd979950e108ceae2db6753b1e6c3abcc347fa63d3ae2739026530e90b9152aab4e501",
    },
    Fixture {
        file: "hunspell/compound-rule.dic",
        sha512: "0c6fe9b0aed1b4f88636ff9de1e573bc8b9c39de301d60c5737b6577d489978771d9f364559ee50a9e64a701e8845b36826ef21301f2767640caf466cb7fc6a4",
    },
    Fixture {
        file: "hunspell/compound.aff",
        sha512: "61da3813545fd68df09904a6679e6f5886e9b32ad866e6c61ffe807eb055099aedbab03a734a0ac2956e0fbc0ad203994245633f99c7e87b154446a745d085dc",
    },
    Fixture {
        file: "hunspell/compound.dic",
        sha512: "fbad7ebac56842da3313a8f9db5b52812af69959b9f3ffc9388de43caff8d7a296eb9bba2426b5686c8ef49164b676bf71185be9d513ef6fdb12ee9f649d0d80",
    },
    Fixture {
        file: "hunspell/error-alias.aff",
        sha512: "072a776a420704e2103de13856934aac01072e035a6aad24d72a38c689f04feed6e7bebca996fa8b1e1385c5e446a2d095a5850908245bc4ed2c437ced2babfb",
    },
    Fixture {
        file: "hunspell/error-alias.dic",
        sha512: "d989c50a12d076caa0cc69a2d007262d8f2afabe5bc9bbec045bd95f2cc20e1ea8a3ed5b3f9310eb3dbf66aab008800789cd2d316384fe0caaf9f68580e9895b",
    },
    Fixture {
        file: "hunspell/error-compound-rule.aff",
        sha512: "04c5ba03a3a961064a70ae4f325e71c24c331fbf3590e5e9e9b518e32e0cd3c0d3162cf782b9e9df5c22ce12e775a6969de0e0aee5ac13783873ea06fae6cef6",
    },
    Fixture {
        file: "hunspell/error-compound-rule.dic",
        sha512: "23099098334d7665269632fe6575c3f065a897eddc70de6ecd0ad8d6ea26e3f11b25cb5c49865307eb9accf1980f855e3afecfae3d6c93e5202b1c2a7ed2634c",
    },
    Fixture {
        file: "hunspell/error-condition.aff",
        sha512: "0acafa9721653d526cb1905a8bd55e11246724f75b8c8b63366111c6786973b00fe9ec7431348d4ebe93dad51acc6ea74b7460a56f3ca119120cf784899eaf60",
    },
    Fixture {
        file: "hunspell/error-condition.dic",
        sha512: "23099098334d7665269632fe6575c3f065a897eddc70de6ecd0ad8d6ea26e3f11b25cb5c49865307eb9accf1980f855e3afecfae3d6c93e5202b1c2a7ed2634c",
    },
    Fixture {
        file: "hunspell/error-encoding.aff",
        sha512: "714adf253b070a850519d825b1af0a102c1b8f2ec33465562a7af13966cfdf4ab21dde49107eb7cb549df29994f2e1d74447d697e04653c9b2f8fd90a28e8e76",
    },
    Fixture {
        file: "hunspell/error-encoding.dic",
        sha512: "23099098334d7665269632fe6575c3f065a897eddc70de6ecd0ad8d6ea26e3f11b25cb5c49865307eb9accf1980f855e3afecfae3d6c93e5202b1c2a7ed2634c",
    },
    Fixture {
        file: "hunspell/error-flag-mode.aff",
        sha512: "16bacb0c6fb34e7ac74f62ec95eba0c058d4f4cf2c73a1161c060af64cefbc6999fb019462f9d3cfbb5009f02b841d0ee686e9e081537b3b231a3f12a72ecbed",
    },
    Fixture {
        file: "hunspell/error-flag-mode.dic",
        sha512: "23099098334d7665269632fe6575c3f065a897eddc70de6ecd0ad8d6ea26e3f11b25cb5c49865307eb9accf1980f855e3afecfae3d6c93e5202b1c2a7ed2634c",
    },
    Fixture {
        file: "hunspell/error-long-flag.aff",
        sha512: "20093efc6e53cecba32681e69d269c9a87cfd54c4cf3592539d860a016e54b4851b44382497582da9e9e79e25f361a134cf7737feabb43146ec62bb20f2041c2",
    },
    Fixture {
        file: "hunspell/error-long-flag.dic",
        sha512: "53a755054b0d9c79034b86b44a72bcf5b3ea76ba06da37d705b0349294b05462302f689f6c29d9c0db133b72635a23713e3fa565eade23d26346321d8fd375fd",
    },
    Fixture {
        file: "hunspell/error-morphology.aff",
        sha512: "8f8d8bbc81afa5a00d6d4adf535b6319d35c8753bc3671d09f7408281bd610275783fc6215649b0fa7542c98adad734db1d3ae9936d2cca8df4279072aadcd61",
    },
    Fixture {
        file: "hunspell/error-morphology.dic",
        sha512: "23099098334d7665269632fe6575c3f065a897eddc70de6ecd0ad8d6ea26e3f11b25cb5c49865307eb9accf1980f855e3afecfae3d6c93e5202b1c2a7ed2634c",
    },
    Fixture {
        file: "hunspell/error-truncated.aff",
        sha512: "547ea039f89e0c4976dd4f72aded285e07912c3180033c9501fdfd6879e2cd961a502cdd2faf56764581b9b9605064631c2f6b98932e82d89ced23004c142800",
    },
    Fixture {
        file: "hunspell/error-truncated.dic",
        sha512: "23099098334d7665269632fe6575c3f065a897eddc70de6ecd0ad8d6ea26e3f11b25cb5c49865307eb9accf1980f855e3afecfae3d6c93e5202b1c2a7ed2634c",
    },
    Fixture {
        file: "hunspell/error-utf8.aff",
        sha512: "e15cb93c173f8401c828632bd01bf5e139ac37b3699de575d68c99f3a0f18448518afcab6a22296e877b606bda37784ecdb9bd016962569241c0eb749b005d8f",
    },
    Fixture {
        file: "hunspell/error-utf8.dic",
        sha512: "23099098334d7665269632fe6575c3f065a897eddc70de6ecd0ad8d6ea26e3f11b25cb5c49865307eb9accf1980f855e3afecfae3d6c93e5202b1c2a7ed2634c",
    },
    Fixture {
        file: "hunspell/flag-long.aff",
        sha512: "a1316980dbc434051b6950fe8c0528f2a44a55e005fa20c9234a13db0d9dc76a570fcaac771f018f35c8f5ab87ef04e676fd0dbe05dafb36b909267e56c05677",
    },
    Fixture {
        file: "hunspell/flag-long.dic",
        sha512: "ce93f3dd05c7e28c1d04fdd3b4e55853a361616a3de83737dbeb60c988a779428c43d0ad2f14f244b384c7482b261147b80290d2fc99d0beecd64b3183c37b73",
    },
    Fixture {
        file: "hunspell/flag-num.aff",
        sha512: "b083752a3e70ae5413991074ebf9a2e85299ce238fa6ce2629c437b0a3ef8289c157bcc5ac3d46a10c068e7faadc76426c0bce632d9d89a3518d6aa1344cbe1a",
    },
    Fixture {
        file: "hunspell/flag-num.dic",
        sha512: "71c0c4165f998b7aa00e9a5ac66aaaa0ed9c5aa0aa1d1a32edf91d42e1bca23c73d26ecb90464b6f539bc24f91bb060e11affc038ab8660d3146ef981623f5fd",
    },
    Fixture {
        file: "hunspell/flag-utf8.aff",
        sha512: "2747b89d50e91b939839c2d8bb89188529a7e5e2474477ce7551a904f8e2d92ee0b492f068c0fc4261924b9ee8dbb6d4f06463c224fb9a72d27c3bc00a1700de",
    },
    Fixture {
        file: "hunspell/flag-utf8.dic",
        sha512: "1b38fc7af8bbf2750b479f9d122608a40d0becd141845acd88060274ae48e9b019e8f8561656812d4529871d0900ea694cf016f217511f5c6765b783d8ec0bfe",
    },
    Fixture {
        file: "hunspell/hungarian.aff",
        sha512: "4c5f6a1913406ae7fa6b87fb137fa58ddaac6564ffce42a19fc96925767a1bd2e62b62a306b9e5867871e4a8769f1bef845dd0f08d5299e6b7bad7ad1519f880",
    },
    Fixture {
        file: "hunspell/hungarian.dic",
        sha512: "ceb67b26020bcae7d25c3fc9d165b88b784d89af3b1ea5122bd2198b0da2775aab09c42ce43e4f6560b3bd7339920e95aee2e6b603dcb387bca998f5ae4898c2",
    },
    Fixture {
        file: "hunspell/latin1.aff",
        sha512: "c66022c6e5938d378fb808b2e34a40eea804229e5ab8238674eeb8b0f6eaec2ba0e6e41d95e6bb8e0d6bbabb2f5baed0c943791bb3bc57bdb762eca0569b29c8",
    },
    Fixture {
        file: "hunspell/latin1.dic",
        sha512: "98f7c364ed1ccc66e1ef6d77f0692a791f6a165dc0f5cfb999c68cb74a5abb9294cb0a2962b6a2e1f207022206c33549e7949ba619b9eb079002720f850714d7",
    },
    Fixture {
        file: "hunspell/raw-flags.aff",
        sha512: "4c0b7dac00021063f7f7ec1653e56a3973fad794317f130ce62e2e15110587be18c6e5aba46644120e66fb8941bc6a2b886c1b6b259f8d7cc734c44ed4efade0",
    },
    Fixture {
        file: "hunspell/raw-flags.dic",
        sha512: "c9e6cb2c59506cc6cc191a029a243a946803943fe58187b768ce464a9309b4f82e7751f149b5d6661a1795c1506f91529b8654fa558592109fd8033ff6cabf33",
    },
    Fixture {
        file: "hunspell/triple.aff",
        sha512: "9e7cb401f60d478df635d86196da743cc325019e6115f1a9ada63e8e7b5668aa66bae9cee8e330b22ecef903a8c29fa9cbb2ba976019d2e5a40587744b0628de",
    },
    Fixture {
        file: "hunspell/triple.dic",
        sha512: "063cab07493fdab849751570d4a5c3eb4f67122143e936cf9f30d2571aa7f01d9e8a19c4e69ed0c05686b44499214d151fd2569b5f0125fcf1ce89da6df78411",
    },
    Fixture {
        file: "hunspell/turkic.aff",
        sha512: "a70955a5127e470a5789b8424845569e96f216e4c60645d57abd406a794f4c7456af852af7e7bd8fe2073ad8aa84c0a459c898fca8319db0b37e3012454f9ea7",
    },
    Fixture {
        file: "hunspell/turkic.dic",
        sha512: "7ea5a0446d7a16ed514a57288548d8ee573960f8a0e8345386e2bf5d39ea4b40297d7256d60f674a691e6f40a35cd0472f50f9caad88fc3b71962c8ff587226e",
    },
    Fixture {
        file: "hunspell/unsupported-directives.aff",
        sha512: "4af39d97f627aacb95cb7b1d55c4aea7cdaae411d79cbd36c6fa524b58e66f8fec532bee524d2bbe30917cec2bc7d8a9cd59569a572c8b5e94874e9299ea639d",
    },
    Fixture {
        file: "hunspell/unsupported-directives.dic",
        sha512: "21e28574e31224641376918eac322f33935dcdc06c36ece9a191b18c20eb366d76d0e3e928be63d4c8278952cd6486a4a088d1b7c06d4d8704cfc0b432dd6ea5",
    },
];
