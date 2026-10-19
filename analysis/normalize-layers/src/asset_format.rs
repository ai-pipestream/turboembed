//! The format of an embedded asset and how its payload is encoded, the
//! columns of the `Assets` layer that are not spans or numbers.

/// What an embedded asset is, as its decoded header identifies it.
///
/// Each variant is one pair of a format name and a media type as OpenNLP's
/// embedded asset detector reports them (`KnownMagics` and
/// `CursorAssetDetector` in the `embedded-assets` add-on): [`name`] gives
/// the format string and [`media_type`] the media type, both exactly as
/// OpenNLP writes them, so a golden file can be compared field by field.
/// Where OpenNLP gives one format name several media types (`xls` for the
/// BIFF2, BIFF3 and BIFF4 worksheets, `pem-parameters` for DSA, EC and DH
/// parameters), each pair is a variant of its own.
///
/// Two variants are not identified by a header:
///
/// - [`Declared`](AssetFormat::Declared): a `data:` URI whose header no
///   signature matches, reported because the URI declares a media type.
///   OpenNLP names its format by the subtype of the declared type
///   (`text/plain` gives `plain`); the declared type is stored with the
///   asset and [`subtype`] derives the name.
/// - [`Unknown`](AssetFormat::Unknown): a long base64 run with no known
///   signature, reported only when the detector is configured for it.
///
/// [`name`]: AssetFormat::name
/// [`media_type`]: AssetFormat::media_type
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AssetFormat {
    Accdb,
    Activemime,
    Afm,
    Aiff,
    Amr,
    AmrWb,
    Appledouble,
    Applefile,
    Ar,
    Arc,
    Au,
    Avi,
    Axml,
    Axx,
    Bpg,
    Bplist,
    Cab,
    Caf,
    Cda,
    Cdx,
    Cgm,
    Chm,
    Class,
    Cpio,
    Crw,
    Crx,
    Dcx,
    Deb,
    Dex,
    Djvu,
    Doc,
    Dpx,
    Dta,
    Dwf,
    Dwg,
    Dxb,
    Dxf,
    E57,
    Elc,
    Elf,
    Emf,
    Exr,
    Fdf,
    Fh,
    Fits,
    Flac,
    Foxmail,
    Gif,
    Grb,
    Gzip,
    Hdf,
    Hfa,
    Hwp,
    Icns,
    Ico,
    Ics,
    Idl,
    Ifo,
    Indd,
    J2c,
    Jb2,
    Jdf,
    Jng,
    Jp2,
    Jpeg,
    Jpm,
    Jpx,
    Jwt,
    Jxl,
    Las,
    Lotus123,
    Lwp,
    Lz,
    M3u,
    M3u8,
    Macho,
    MachoFat,
    Mat,
    Matroska,
    Mdi,
    Midi,
    Mif,
    Mj2,
    Mng,
    Mny,
    Mod,
    Mov,
    Mp3,
    Mpeg,
    Mpx,
    Nc,
    Nes,
    Niff,
    Ntf,
    Ogg,
    Ole2,
    Orf,
    Otf,
    P7s,
    Parquet,
    Pcapng,
    Pdf,
    Pds,
    Pe,
    PemAttributeCert,
    PemCert,
    PemCms,
    PemCrl,
    PemKey,
    PemParametersDh,
    PemParametersDsa,
    PemParametersEc,
    PemPkcs7,
    PemRequest,
    Pfa,
    Php,
    Png,
    Ps,
    Psd,
    Rar,
    Rgb,
    Rm,
    Rpm,
    Rtf,
    Sap,
    Sd2,
    SevenZip,
    SgiMovie,
    Sib,
    Sid,
    Sit,
    Sqlite,
    Svg,
    Sz,
    Tiff,
    Torrent,
    Tsd,
    Ttf,
    Tzx,
    Uc2,
    Vcf,
    Vhd,
    Vtt,
    Warc,
    Wasm,
    Wav,
    Wb1,
    Wb2,
    Webp,
    Wk1,
    Wk3,
    Wk4,
    Wks,
    Wmf,
    Woff,
    Woff2,
    Wq1,
    Wq2,
    Xar,
    Xcf,
    XlsSheet2,
    XlsSheet3,
    XlsSheet4,
    XlwWorkspace3,
    XlwWorkspace4,
    Xml,
    Xpm,
    Xz,
    Zip,
    Zstd,
    /// A `data:` URI's payload that no signature identifies; the URI's
    /// declared media type names it.
    Declared,
    /// A base64 run that no signature identifies.
    Unknown,
}

impl AssetFormat {
    /// OpenNLP's format name, such as `png`, `pem-key` or `xls`. Empty for
    /// [`Declared`](AssetFormat::Declared), whose name is the subtype of
    /// the media type the URI declared ([`subtype`]).
    pub fn name(self) -> &'static str {
        self.names().0
    }

    /// The media type OpenNLP reports for the format, such as `image/png`.
    /// Empty for [`Declared`](AssetFormat::Declared), whose media type is
    /// the one the URI declared.
    pub fn media_type(self) -> &'static str {
        self.names().1
    }

    /// The format a pair of strings names, the inverse of [`name`] and
    /// [`media_type`] together.
    ///
    /// [`name`]: AssetFormat::name
    /// [`media_type`]: AssetFormat::media_type
    pub fn from_names(name: &str, media_type: &str) -> Option<AssetFormat> {
        AssetFormat::SNIFFED
            .into_iter()
            .chain([AssetFormat::Unknown])
            .find(|f| f.name() == name && f.media_type() == media_type)
    }
}

/// The subtype of a media type, which OpenNLP takes as the format name of
/// a `data:` URI whose header it does not recognize: `plain` for
/// `text/plain;charset=UTF-8`.
///
/// `None` unless the text before any `;` is a type and a subtype around
/// one `/`, each a non-empty token (RFC 9110 section 5.6.2, the code
/// points the WHATWG MIME type parser accepts). Parameters after the `;`
/// are not checked: the WHATWG parser skips ones it cannot read.
pub fn subtype(media_type: &str) -> Option<&str> {
    let essence = media_type.split_once(';').map_or(media_type, |(essence, _)| essence);
    let (kind, subtype) = essence.split_once('/')?;
    (is_token(kind) && is_token(subtype)).then_some(subtype)
}

/// Whether `s` is a non-empty run of token characters (RFC 9110 `tchar`).
fn is_token(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

/// How an asset's payload is written in the text.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AssetEncoding {
    /// Standard base64 (RFC 4648 section 4), possibly wrapped at 64 or 76
    /// columns.
    Base64,
    /// The URL-safe alphabet (RFC 4648 section 5): the payload holds `-`
    /// or `_`, or it is the claims segment of a JSON Web Token.
    Base64Url,
    /// Pairs of hexadecimal digits (RFC 4648 section 8), possibly
    /// separated or prefixed with `0x`.
    Hex,
}
