//! Clean-room English phonetic onset classification for indefinite articles (`a` vs `an`, `ART-001`).

/// Returns `Some(true)` when `word` has a vowel phonetic onset requiring `"an"`,
/// `Some(false)` when `word` has a consonant phonetic onset requiring `"a"`,
/// or `None` when `word` is not an alphabetic/initialism token that governs an article.
pub(crate) fn requires_an(word: &str, american_silent_h_herb: bool) -> Option<bool> {
    let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
    if trimmed.is_empty() {
        return None;
    }

    let first_segment = trimmed.split(['-', '/']).next().unwrap_or(trimmed);
    if first_segment.is_empty() {
        return None;
    }

    let first_char = first_segment.chars().next()?;
    if !first_char.is_ascii_alphabetic() {
        return match first_segment {
            "8" | "11" | "18" | "80" | "800" => Some(true),
            _ => None,
        };
    }

    // Single-letter token or hyphenated single-letter prefix (`x-ray`, `n-gram`, `u-turn`, `a-frame`).
    // Only apply single-letter pronunciation when the token is a hyphenated letter prefix (`x-ray`)
    // or a standalone single letter other than indefinite article `a`/`A`.
    if first_segment.len() == 1 && (trimmed.contains(['-', '/']) || !first_segment.eq_ignore_ascii_case("a")) {
        return Some(letter_name_has_vowel_onset(first_char));
    }

    // Check explicit initialism overrides first (covers mixed-case like `NVMe`, `OAuth`, `XPath`
    // and acronyms vs letter-by-letter initialisms like `SQL`, `REST`, `URL`, `MRI`, `HTTP`).
    if let Some(onset) = explicit_initialism_vowel_onset(first_segment) {
        return Some(onset);
    }

    // All-uppercase initialisms of 2..=5 characters without vowels (e.g. `LLM`, `NDA`, `SSD`, `RPC`, `HTML`).
    if is_vowelless_initialism(first_segment) {
        return Some(letter_name_has_vowel_onset(first_char));
    }

    let lower = first_segment.to_ascii_lowercase();

    // 1. Silent-`h` stems requiring `an`.
    if is_silent_h(&lower, american_silent_h_herb) {
        return Some(true);
    }

    // 2. `/ju:/` or `/w/` consonant-onset stems spelled with initial vowel letters (`u`, `e`, `o`) requiring `a`.
    if is_consonant_glide_vowel_stem(&lower) {
        return Some(false);
    }

    // 3. Standard orthographic vowel vs consonant onset.
    Some(matches!(first_char.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u'))
}

///Synthesizes the replacement indefinite article (`"a"` <-> `"an"`), preserving original capitalization.
pub(crate) fn replacement_article(surface: &str, want_an: bool) -> Option<&'static str> {
    match (surface, want_an) {
        ("a", true) => Some("an"),
        ("A", true) => Some("An"),
        ("an", false) => Some("a"),
        ("An", false) => Some("A"),
        ("AN", false) => Some("A"),
        _ => None,
    }
}

const fn letter_name_has_vowel_onset(c: char) -> bool {
    matches!(c.to_ascii_uppercase(), 'A' | 'E' | 'F' | 'H' | 'I' | 'L' | 'M' | 'N' | 'O' | 'R' | 'S' | 'X')
}

fn is_vowelless_initialism(segment: &str) -> bool {
    let len = segment.len();
    (2..=6).contains(&len)
        && segment.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        && segment.bytes().any(|b| b.is_ascii_uppercase())
        && !segment.bytes().any(|b| matches!(b, b'A' | b'E' | b'I' | b'O' | b'U' | b'Y'))
}

fn explicit_initialism_vowel_onset(segment: &str) -> Option<bool> {
    // Only treat as an initialism/acronym when it contains at least 2 uppercase letters
    // or uppercase + digits (`S3`, `L1`, `L2`, `MP3`, `MD5`, `X86`).
    let upper_count = segment.bytes().filter(|b| b.is_ascii_uppercase()).count();
    let has_digit = segment.bytes().any(|b| b.is_ascii_digit());
    if upper_count < 2 && !(upper_count == 1 && has_digit) {
        return None;
    }
    match segment {
        // Vowel-sound initialisms (`A`, `E`, `F`, `H`, `I`, `L`, `M`, `N`, `O`, `R`, `S`, `X`).
        "MRI" | "HTTP" | "HTTPS" | "HTML" | "FBI" | "NDA" | "SLA" | "SLO" | "SLI" | "SSD" | "RPC" | "SQL" | "LLM"
        | "LSTM" | "LP" | "L1" | "L2" | "L3" | "MD5" | "MP3" | "MP4" | "MTU" | "MVC" | "MVP" | "MX" | "NPU" | "NLU"
        | "NLP" | "NFT" | "NIC" | "NN" | "NNS" | "NP" | "NTP" | "NVMe" | "RFC" | "RFI" | "RGB" | "RMA" | "RMS"
        | "RNG" | "RNN" | "ROI" | "RPM" | "RSA" | "RSS" | "RTOS" | "RTT" | "SDK" | "SFTP" | "SHA" | "SKU" | "SMB"
        | "SML" | "SMP" | "SMS" | "SMTP" | "SNI" | "SNMP" | "SOA" | "SPI" | "SPDX" | "SRAM" | "SRE" | "SRL" | "SSE"
        | "SSH" | "SSID" | "SSL" | "SSO" | "SSR" | "SST" | "SVG" | "SVM" | "SVN" | "S3" | "FPU" | "FPGA" | "FPS"
        | "FQDN" | "FSM" | "FST" | "FTP" | "FTPS" | "FX" | "HDD" | "HDMI" | "HDFS" | "HMAC" | "HPC" | "HSM" | "HT"
        | "HSTS" | "XHR" | "XML" | "XMPP" | "XOR" | "XPath" | "XSS" | "XSLT" | "X86" | "API" | "AST" | "ACL"
        | "ACM" | "ADC" | "AES" | "AI" | "ALU" | "AMQP" | "ANSI" | "AOT" | "APK" | "ARP" | "ASIC" | "ASN" | "ASR"
        | "AVX" | "AWS" | "EC2" | "ECC" | "ECDSA" | "EDA" | "EDI" | "EFI" | "EKS" | "ELF" | "ELK" | "EMR" | "EOF"
        | "EOL" | "EOS" | "ERP" | "ESB" | "ETL" | "EULA" | "IAM" | "IANA" | "ICMP" | "IDE" | "IDL" | "IDS" | "IEEE"
        | "IETF" | "IGMP" | "IMAP" | "IMEI" | "IOPS" | "IOT" | "IoT" | "IP" | "IPC" | "IPFS" | "IPMI" | "IPO"
        | "IPS" | "IPv4" | "IPv6" | "IPV4" | "IPV6" | "IR" | "IRC" | "IRQ" | "ISA" | "ISO" | "ISP" | "ISR" | "IT"
        | "ITU" | "OAuth" | "ODBC" | "OEM" | "OID" | "OIDC" | "OLAP" | "OLED" | "OLTP" | "ONNX" | "OOM" | "OOP"
        | "OPA" | "ORM" | "OS" | "OSI" | "OSPF" | "OSS" | "OTA" | "OTP" | "OWASP" => Some(true),
        // Consonant-sound initialisms and acronyms (`U*` -> `/ju:/`, plus `JWT`, `CPU`, `GPU`, `REST`, `TLS`, etc.).
        "URL" | "URLs" | "URI" | "URIs" | "URN" | "URNs" | "UDP" | "UUID" | "USB" | "UTF" | "UI" | "UX" | "UAT"
        | "UDF" | "UEFI" | "UID" | "UML" | "UN" | "UPC" | "UPS" | "US" | "USA" | "UTC" | "JWT" | "CPU" | "GPU"
        | "REST" | "TLS" | "TCP" | "DNS" | "BGP" | "CDN" | "CLI" | "CRDT" | "CORS" | "CSRF" | "CSS" | "CSV" | "DAG"
        | "DB" | "DDL" | "DOM" | "DRAM" | "DSL" | "DTO" | "GCN" | "GNN" | "GPT" | "gRPC" | "GRPC" | "GUI" | "GUID"
        | "JSON" | "JVM" | "KMS" | "KV" | "LAN" | "LDAP" | "NAT" | "NAS" | "PCA" | "PCI" | "PDF" | "PII" | "PKI"
        | "PNG" | "POSIX" | "PR" | "QA" | "QPS" | "QR" | "RAID" | "RAM" | "RBAC" | "RDBMS" | "ROM" | "SAN" | "SAS"
        | "SATA" | "SAML" | "SCIM" | "SCP" | "SCSI" | "SIMD" | "SIM" | "SIP" | "SOC" | "SOHO" | "SOP" | "SPA"
        | "TAR" | "TLD" | "TOML" | "TPM" | "TTL" | "TTY" | "VLAN" | "VM" | "VPC" | "VPN" | "WASM" | "WAF" | "WAN"
        | "WARC" | "YAML" | "ZKP" => Some(false),
        _ => None,
    }
}

fn is_silent_h(lower: &str, american_silent_h_herb: bool) -> bool {
    if matches!(
        lower,
        "hour"
            | "hours"
            | "hourly"
            | "hourglass"
            | "honest"
            | "honestly"
            | "honesty"
            | "heir"
            | "heirs"
            | "heiress"
            | "heiresses"
            | "heirloom"
            | "heirlooms"
            | "honor"
            | "honors"
            | "honored"
            | "honoring"
            | "honour"
            | "honours"
            | "honoured"
            | "honouring"
            | "honorable"
            | "honourable"
            | "honorably"
            | "honourably"
            | "honorary"
            | "honorific"
            | "honorifics"
            | "homage"
    ) {
        return true;
    }
    american_silent_h_herb
        && matches!(lower, "herb" | "herbs" | "herbal" | "herbalist" | "herbicide" | "herbivore" | "herbivorous")
}

fn is_consonant_glide_vowel_stem(lower: &str) -> bool {
    // 1. `/w/` onset on `one` / `once` / `ouija`.
    if matches!(lower, "one" | "ones" | "once" | "oneness" | "ouija") {
        return true;
    }
    // 2. `/ju:/` onset on `eu*` / `ew*`.
    if matches!(
        lower,
        "eulogy"
            | "eulogies"
            | "eulogize"
            | "euphemism"
            | "euphemisms"
            | "euphemistic"
            | "euphoria"
            | "euphoric"
            | "europe"
            | "european"
            | "europeans"
            | "euro"
            | "euros"
            | "eurasia"
            | "eurasian"
            | "eureka"
            | "euthanasia"
            | "ewe"
            | "ewes"
            | "ewer"
            | "ewers"
            | "eucalyptus"
            | "euclidean"
            | "eugenics"
            | "eukaryote"
            | "eukaryotic"
            | "euler"
            | "eulerian"
            | "eunuch"
            | "euphony"
    ) {
        return true;
    }
    // 3. `/ju:/` onset on `u*` stems (`uni*`, `use*`, `usu*`, `ute*`, `uti*`, `uto*`, etc.).
    if matches!(
        lower,
        "unit"
            | "units"
            | "united"
            | "uniting"
            | "unity"
            | "union"
            | "unions"
            | "unique"
            | "uniquely"
            | "uniqueness"
            | "unicorn"
            | "unicorns"
            | "uniform"
            | "uniforms"
            | "uniformly"
            | "uniformity"
            | "unify"
            | "unified"
            | "unifies"
            | "unifying"
            | "unification"
            | "unilateral"
            | "unilaterally"
            | "unanimous"
            | "unanimously"
            | "unanimity"
            | "universal"
            | "universally"
            | "universality"
            | "universe"
            | "universes"
            | "university"
            | "universities"
            | "uranium"
            | "usage"
            | "usages"
            | "use"
            | "uses"
            | "used"
            | "useful"
            | "usefully"
            | "usefulness"
            | "useless"
            | "uselessly"
            | "user"
            | "users"
            | "username"
            | "usernames"
            | "userland"
            | "usual"
            | "usually"
            | "usurp"
            | "usurped"
            | "usurper"
            | "usury"
            | "utensil"
            | "utensils"
            | "uterus"
            | "uterine"
            | "utility"
            | "utilities"
            | "utilitarian"
            | "utilize"
            | "utilized"
            | "utilizes"
            | "utilizing"
            | "utilization"
            | "utopia"
            | "utopian"
            | "utopias"
            | "ubiquitous"
            | "ubiquity"
            | "ukulele"
            | "unicycle"
            | "unimodal"
            | "unipolar"
            | "unison"
            | "unitary"
            | "univariate"
            | "unix"
            | "urea"
            | "ureter"
            | "urethra"
            | "uric"
            | "urinal"
            | "urinary"
            | "urine"
            | "urology"
            | "usability"
            | "usable"
            | "utah"
    ) {
        return true;
    }
    false
}
