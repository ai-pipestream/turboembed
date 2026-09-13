// SPDX-License-Identifier: Apache-2.0
//
// BERT uncased BasicTokenizer + WordPiece. Matches HuggingFace
// BertTokenizer (do_lower_case, tokenize_chinese_chars, strip_accents
// implied by lowercasing). Load-time uses std::unordered_map; tokenize
// writes into a caller/engine scratch id buffer.

#include "internal.hpp"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace turborerank {
namespace impl {
namespace {

bool is_control(uint32_t cp) {
    if (cp == '\t' || cp == '\n' || cp == '\r') {
        return false;
    }
    return cp < 32 || (cp >= 0x7F && cp <= 0x9F);
}

bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x00A0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_punct_ascii(uint32_t cp) {
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
           (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
}

bool is_punctuation(uint32_t cp) {
    if (is_punct_ascii(cp)) {
        return true;
    }
    // Unicode P* buckets used by BERT BasicTokenizer.
    return (cp >= 0x00A1 && cp <= 0x00BF && cp != 0x00A0) ||
           (cp >= 0x2010 && cp <= 0x2027) || (cp >= 0x2030 && cp <= 0x205E) ||
           (cp >= 0x3001 && cp <= 0x303F) || cp == 0x00AD || cp == 0x058A ||
           cp == 0x05BE || cp == 0x055A || cp == 0x055B;
}

bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

bool is_combining_mark(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x1AB0 && cp <= 0x1AFF) ||
           (cp >= 0x1DC0 && cp <= 0x1DFF) || (cp >= 0x20D0 && cp <= 0x20FF) ||
           (cp >= 0xFE20 && cp <= 0xFE2F);
}

// Precomposed Latin → base letter (NFD then drop Mn). Enough for
// BERT-uncased English + common European test text.
uint32_t strip_latin_accent(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') {
        return cp - 'A' + 'a';
    }
    if (cp >= 'a' && cp <= 'z') {
        return cp;
    }
    switch (cp) {
    case 0x00C0: case 0x00C1: case 0x00C2: case 0x00C3: case 0x00C4:
    case 0x00C5: case 0x0100: case 0x0102: case 0x0104:
    case 0x00E0: case 0x00E1: case 0x00E2: case 0x00E3: case 0x00E4:
    case 0x00E5: case 0x0101: case 0x0103: case 0x0105:
        return 'a';
    case 0x00C7: case 0x0106: case 0x0108: case 0x010A: case 0x010C:
    case 0x00E7: case 0x0107: case 0x0109: case 0x010B: case 0x010D:
        return 'c';
    case 0x00D0: case 0x010E: case 0x0110:
    case 0x00F0: case 0x010F: case 0x0111:
        return 'd';
    case 0x00C8: case 0x00C9: case 0x00CA: case 0x00CB: case 0x0112:
    case 0x0114: case 0x0116: case 0x0118: case 0x011A:
    case 0x00E8: case 0x00E9: case 0x00EA: case 0x00EB: case 0x0113:
    case 0x0115: case 0x0117: case 0x0119: case 0x011B:
        return 'e';
    case 0x011C: case 0x011E: case 0x0120: case 0x0122:
    case 0x011D: case 0x011F: case 0x0121: case 0x0123:
        return 'g';
    case 0x0124: case 0x0126: case 0x0125: case 0x0127:
        return 'h';
    case 0x00CC: case 0x00CD: case 0x00CE: case 0x00CF: case 0x0128:
    case 0x012A: case 0x012C: case 0x012E: case 0x0130:
    case 0x00EC: case 0x00ED: case 0x00EE: case 0x00EF: case 0x0129:
    case 0x012B: case 0x012D: case 0x012F: case 0x0131:
        return 'i';
    case 0x0134: case 0x0135:
        return 'j';
    case 0x0136: case 0x0137:
        return 'k';
    case 0x0139: case 0x013B: case 0x013D: case 0x013F: case 0x0141:
    case 0x013A: case 0x013C: case 0x013E: case 0x0140: case 0x0142:
        return 'l';
    case 0x0143: case 0x0145: case 0x0147: case 0x00D1:
    case 0x0144: case 0x0146: case 0x0148: case 0x00F1:
        return 'n';
    case 0x00D2: case 0x00D3: case 0x00D4: case 0x00D5: case 0x00D6:
    case 0x00D8: case 0x014C: case 0x014E: case 0x0150:
    case 0x00F2: case 0x00F3: case 0x00F4: case 0x00F5: case 0x00F6:
    case 0x00F8: case 0x014D: case 0x014F: case 0x0151:
        return 'o';
    case 0x0154: case 0x0156: case 0x0158:
    case 0x0155: case 0x0157: case 0x0159:
        return 'r';
    case 0x015A: case 0x015C: case 0x015E: case 0x0160:
    case 0x015B: case 0x015D: case 0x015F: case 0x0161:
    case 0x00DF:
        return 's';
    case 0x0162: case 0x0164: case 0x0166:
    case 0x0163: case 0x0165: case 0x0167:
        return 't';
    case 0x00D9: case 0x00DA: case 0x00DB: case 0x00DC: case 0x0168:
    case 0x016A: case 0x016C: case 0x016E: case 0x0170: case 0x0172:
    case 0x00F9: case 0x00FA: case 0x00FB: case 0x00FC: case 0x0169:
    case 0x016B: case 0x016D: case 0x016F: case 0x0171: case 0x0173:
        return 'u';
    case 0x0174: case 0x0175:
        return 'w';
    case 0x00DD: case 0x0176: case 0x0178:
    case 0x00FD: case 0x00FF: case 0x0177:
        return 'y';
    case 0x0179: case 0x017B: case 0x017D:
    case 0x017A: case 0x017C: case 0x017E:
        return 'z';
    case 0x00C6: case 0x00E6:
        return 0; // æ — emit "ae" at the caller
    default:
        if (cp < 128) {
            return static_cast<uint32_t>(std::tolower(static_cast<int>(cp)));
        }
        return cp;
    }
}

size_t utf8_next(const char *s, size_t n, size_t i, uint32_t *cp) {
    if (i >= n) {
        *cp = 0;
        return 0;
    }
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c >> 5) == 0x6 && i + 1 < n) {
        *cp = (static_cast<uint32_t>(c & 0x1F) << 6) |
              (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        return 2;
    }
    if ((c >> 4) == 0xE && i + 2 < n) {
        *cp = (static_cast<uint32_t>(c & 0x0F) << 12) |
              ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
              (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        return 3;
    }
    if ((c >> 3) == 0x1E && i + 3 < n) {
        *cp = (static_cast<uint32_t>(c & 0x07) << 18) |
              ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
              ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
              (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        return 4;
    }
    *cp = 0xFFFD;
    return 1;
}

void append_utf8(std::string *out, uint32_t cp) {
    if (cp < 0x80) {
        out->push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

void clean_and_tokenize_basic(const char *utf8, size_t n, std::vector<std::string> *words) {
    std::string cleaned;
    cleaned.reserve(n + 8);
    size_t i = 0;
    while (i < n) {
        uint32_t cp = 0;
        const size_t adv = utf8_next(utf8, n, i, &cp);
        i += adv;
        if (cp == 0 || is_control(cp)) {
            continue;
        }
        if (is_whitespace(cp)) {
            cleaned.push_back(' ');
            continue;
        }
        if (is_cjk(cp)) {
            cleaned.push_back(' ');
            append_utf8(&cleaned, cp);
            cleaned.push_back(' ');
            continue;
        }
        if (is_combining_mark(cp)) {
            continue;
        }
        if (cp == 0x00C6) {
            cleaned.append("ae");
            continue;
        }
        if (cp == 0x00E6) {
            cleaned.append("ae");
            continue;
        }
        const uint32_t stripped = strip_latin_accent(cp);
        if (stripped == 0) {
            continue;
        }
        append_utf8(&cleaned, stripped);
    }

    std::string cur;
    auto flush_word = [&]() {
        if (cur.empty()) {
            return;
        }
        // Split punctuation: each punct is its own token.
        std::string piece;
        size_t j = 0;
        const size_t m = cur.size();
        while (j < m) {
            uint32_t cp = 0;
            const size_t adv = utf8_next(cur.data(), m, j, &cp);
            if (is_punctuation(cp)) {
                if (!piece.empty()) {
                    words->push_back(piece);
                    piece.clear();
                }
                std::string p;
                append_utf8(&p, cp);
                words->push_back(p);
            } else {
                for (size_t k = 0; k < adv; ++k) {
                    piece.push_back(cur[j + k]);
                }
            }
            j += adv;
        }
        if (!piece.empty()) {
            words->push_back(piece);
        }
        cur.clear();
    };

    size_t j = 0;
    const size_t m = cleaned.size();
    while (j < m) {
        uint32_t cp = 0;
        const size_t adv = utf8_next(cleaned.data(), m, j, &cp);
        if (cp == ' ') {
            flush_word();
        } else {
            for (size_t k = 0; k < adv; ++k) {
                cur.push_back(cleaned[j + k]);
            }
        }
        j += adv;
    }
    flush_word();
}

bool wordpiece_word(
    const Vocab &vocab,
    const std::string &word,
    int32_t *ids,
    size_t ids_cap,
    size_t *n_out
) {
    if (word.empty()) {
        return true;
    }
    // Split into Unicode characters (as UTF-8 slices).
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < word.size()) {
        uint32_t cp = 0;
        const size_t adv = utf8_next(word.data(), word.size(), i, &cp);
        chars.emplace_back(word.substr(i, adv));
        i += adv;
    }

    bool is_bad = false;
    size_t start = 0;
    std::vector<int32_t> sub;
    while (start < chars.size()) {
        size_t end = chars.size();
        int32_t found = -1;
        while (start < end) {
            std::string substr;
            if (start > 0) {
                substr = "##";
            }
            for (size_t k = start; k < end; ++k) {
                substr += chars[k];
            }
            auto it = vocab.token_to_id.find(substr);
            if (it != vocab.token_to_id.end()) {
                found = it->second;
                break;
            }
            --end;
        }
        if (found < 0) {
            is_bad = true;
            break;
        }
        sub.push_back(found);
        start = end;
    }
    if (is_bad) {
        if (*n_out >= ids_cap) {
            return false;
        }
        ids[(*n_out)++] = vocab.unk_id;
        return true;
    }
    if (*n_out + sub.size() > ids_cap) {
        return false;
    }
    for (int32_t id : sub) {
        ids[(*n_out)++] = id;
    }
    return true;
}

} // namespace

bool load_vocab_txt(const char *path, Vocab *vocab, std::string *err) {
    if (path == nullptr || vocab == nullptr) {
        if (err) {
            *err = "load_vocab_txt: null argument";
        }
        return false;
    }
    std::ifstream in(path);
    if (!in) {
        if (err) {
            *err = std::string("vocab.txt not found: ") + path;
        }
        return false;
    }
    vocab->token_to_id.clear();
    std::string line;
    int32_t id = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        vocab->token_to_id.emplace(line, id);
        ++id;
    }
    auto set_special = [&](const char *tok, int32_t fallback) -> int32_t {
        auto it = vocab->token_to_id.find(tok);
        return it == vocab->token_to_id.end() ? fallback : it->second;
    };
    vocab->unk_id = set_special("[UNK]", kUnkId);
    vocab->cls_id = set_special("[CLS]", kClsId);
    vocab->sep_id = set_special("[SEP]", kSepId);
    vocab->pad_id = set_special("[PAD]", kPadId);
    vocab->loaded = !vocab->token_to_id.empty();
    if (!vocab->loaded) {
        if (err) {
            *err = std::string("vocab.txt empty: ") + path;
        }
        return false;
    }
    return true;
}

size_t tokenize_wordpiece(
    const Vocab &vocab,
    const char *utf8,
    size_t utf8_len,
    int32_t *ids,
    size_t ids_cap,
    std::string *err
) {
    if (!vocab.loaded) {
        if (err) {
            *err = "tokenize: vocab not loaded";
        }
        return 0;
    }
    if (ids == nullptr || ids_cap == 0) {
        if (err) {
            *err = "tokenize: null id buffer";
        }
        return 0;
    }
    if (utf8 == nullptr && utf8_len > 0) {
        if (err) {
            *err = "tokenize: null text with len>0";
        }
        return 0;
    }
    std::vector<std::string> words;
    clean_and_tokenize_basic(utf8 == nullptr ? "" : utf8, utf8_len, &words);
    size_t n = 0;
    for (const auto &w : words) {
        if (!wordpiece_word(vocab, w, ids, ids_cap, &n)) {
            if (err) {
                *err = "tokenize: id scratch overflow";
            }
            return 0;
        }
    }
    return n;
}

} // namespace impl
} // namespace turborerank
