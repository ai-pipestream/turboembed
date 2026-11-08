package ai.pipestream.turboembed;

import java.util.Optional;

/**
 * Built-in PII kinds with their stable native ids: the same numbers as the native
 * {@code TURBO_PII_KIND_*} constants, the proto {@code PiiKindId} enum and the binary
 * layer codec. An id is identity only; ids are append-only and never reused. A caller's
 * custom kind has no constant here and is known by its name alone.
 */
public enum TurboPiiKind {
    JWT(1, "jwt"),
    AWS_ACCESS_KEY(2, "aws-access-key"),
    GITHUB_TOKEN(3, "github-token"),
    URL_CREDENTIAL(4, "url-credential"),
    EMAIL(5, "email"),
    IBAN(6, "iban"),
    IMEI(7, "imei"),
    CARD(8, "card"),
    BTC_ADDRESS(9, "btc-address"),
    ETH_ADDRESS(10, "eth-address"),
    MAC(11, "mac"),
    IPV6(12, "ipv6"),
    IPV4(13, "ipv4"),
    US_SSN(14, "us-ssn"),
    US_ITIN(15, "us-itin"),
    UK_NHS(16, "uk-nhs"),
    DE_STEUER_ID(17, "de-steuer-id"),
    ES_DNI(18, "es-dni"),
    ES_NIE(19, "es-nie"),
    ES_CIF(20, "es-cif"),
    DE_PERSONALAUSWEIS(21, "de-personalausweis"),
    FR_NIR(22, "fr-nir"),
    FR_SIREN(23, "fr-siren"),
    FR_SIRET(24, "fr-siret"),
    BR_CPF(25, "br-cpf"),
    BR_CNPJ(26, "br-cnpj"),
    BIC(27, "bic"),
    CA_SIN(28, "ca-sin"),
    ABA_ROUTING(29, "aba-routing"),
    PHONE(30, "phone"),
    VIN(31, "vin"),
    TRACKING_NUMBER(32, "tracking-number"),
    US_EIN(33, "us-ein"),
    URL(34, "url"),
    US_PASSPORT(35, "us-passport"),
    US_DRIVERS_LICENCE(36, "us-drivers-licence"),
    US_ZIP(37, "us-zip"),
    PHONE_EXTENSION(38, "phone-extension");

    private final int nativeCode;
    private final String wireName;

    TurboPiiKind(int nativeCode, String wireName) {
        this.nativeCode = nativeCode;
        this.wireName = wireName;
    }

    /** Returns the native {@code TURBO_PII_KIND_*} id. */
    public int nativeCode() {
        return nativeCode;
    }

    /** Returns the kind's name, as a PII row's label carries it. */
    public String wireName() {
        return wireName;
    }

    /** Returns the built-in kind with native id {@code nativeCode}. */
    public static TurboPiiKind fromNative(long nativeCode) {
        for (TurboPiiKind kind : values()) {
            if (kind.nativeCode == nativeCode) {
                return kind;
            }
        }
        throw new IllegalStateException("Unknown native PII kind id: " + nativeCode);
    }

    /** Returns the built-in kind named {@code wireName}, or empty for a custom kind. */
    public static Optional<TurboPiiKind> fromName(String wireName) {
        for (TurboPiiKind kind : values()) {
            if (kind.wireName.equals(wireName)) {
                return Optional.of(kind);
            }
        }
        return Optional.empty();
    }
}
