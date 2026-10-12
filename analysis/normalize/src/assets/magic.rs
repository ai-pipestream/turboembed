// Copyright (c) Ian F. Darwin 1986, 1987, 1989, 1990, 1991, 1992, 1994, 1995.
// Software written by Ian F. Darwin and others;
// maintained 1994- Christos Zoulas.
//
// This software is not subject to any export provision of the United States
// Department of Commerce, and may be exported to any country or planet.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice immediately at the beginning of the file, without modification,
//    this list of conditions, and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

//! File signatures: the leading bytes that identify a format, with the
//! format and media type they stand for.
//!
//! The table is OpenNLP's `KnownMagics`, which adapts fixed signatures at
//! offset zero from Apache Tika's media-type catalog (tika-mimetypes.xml,
//! Apache License 2.0; Copyright 2021 The Apache Software Foundation). The
//! catalog carries MIME type information from file(1) 4.26, whose notice
//! opens this file, and definitions adapted from the PRONOM Technical
//! Registry of The National Archives (Open Government Licence v3.0).
//!
//! A matching signature says what a payload may be; it does not validate
//! the file. Some formats need more than the signature (an AIFF form type,
//! an Excel document type, the EMF signature at byte 40, pcapng byte-order
//! magic, a JPEG 2000 brand, DEX version digits, a WebVTT separator); the
//! checks are in `sniff`. RIFF form types, the CMS timestamped-data
//! identifier, calendars and SVG roots without a table signature are
//! recognized there too.

use turbo_normalize_layers::AssetFormat;

/// One signature: the bytes a file of the format starts with.
pub type Magic = (&'static [u8], AssetFormat);

/// Every signature, longest first, and in OpenNLP's table order among
/// signatures of one length, so the first match is the most specific one
/// and ties resolve as OpenNLP resolves them.
pub const MAGICS: &[Magic] = &[
    (b"-----BEGIN DSA PRIVATE KEY-----", AssetFormat::PemKey),
    (b"-----BEGIN RSA PRIVATE KEY-----", AssetFormat::PemKey),
    (b"-----BEGIN DSA PARAMETERS-----", AssetFormat::PemParametersDsa),
    (b"-----BEGIN EC PARAMETERS-----", AssetFormat::PemParametersEc),
    (b"-----BEGIN CERTIFICATE-----", AssetFormat::PemCert),
    (b"-----BEGIN PRIVATE KEY-----", AssetFormat::PemKey),
    (b"-----BEGIN PUBLIC KEY-----", AssetFormat::PemKey),
    (b"%!PS-Adobe-3.0 EPSF-3.0", AssetFormat::Ps),
    (b"-----BEGIN DSA KEY-----", AssetFormat::PemKey),
    (b"-----BEGIN RSA KEY-----", AssetFormat::PemKey),
    (b"AutoCAD Binary DXF\x0d\x0a\x1a\x00", AssetFormat::Dxf),
    (b"!<arch>\x0adebian-binary", AssetFormat::Deb),
    (b"-----BEGIN PKCS7-----", AssetFormat::P7s),
    (b"!<arch>\x0adebian-split", AssetFormat::Deb),
    (b"AutoCAD DXB 1.0\x0d\x0a\x1a\x00", AssetFormat::Dxb),
    (b"\x00\x01\x00\x00MSISAM Database", AssetFormat::Mny),
    (b"-----BEGIN KEY-----", AssetFormat::PemKey),
    (b"HWP Document File V", AssetFormat::Hwp),
    (b"%!PS-AdobeFont-1.0", AssetFormat::Pfa),
    (b"RIFF$\x00\x00\x00CDDAfmt \x18", AssetFormat::Cda),
    (b"SQLite format 3\x00", AssetFormat::Sqlite),
    (b"\x06\x06\xed\xf5\xd8\x1dF\xe5\xbd1\xef\xe7\xfet\xb7\x1d", AssetFormat::Indd),
    (b"Extended Module:", AssetFormat::Mod),
    (b"StartFontMetrics", AssetFormat::Afm),
    (b"\xc0\xb9\x07.O\x93\xf1F\xa0\x15y,\xa1\xd9\xe8!", AssetFormat::Axx),
    (b"BEGIN:VCALENDAR", AssetFormat::Ics),
    (b"EHFA_HEADER_TAG", AssetFormat::Hfa),
    (b"\x10\x10\x10\x10\x10\x10\x10\x11\x11\x11\x11\x11\x11S", AssetFormat::Foxmail),
    (b"BA(\x00\x00\x00.\x00\x00\x00\x00\x00\x00\x00", AssetFormat::Ico),
    (b"II\x1a\x00\x00\x00HEAPCCDR", AssetFormat::Crw),
    (b"PDS_VERSION_ID", AssetFormat::Pds),
    (b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a", AssetFormat::Jxl),
    (b"\x00\x00\x00\x0cjP  \x0d\x0a\x87\x0a", AssetFormat::Jp2),
    (b"ActiveMime\x00\x00", AssetFormat::Activemime),
    (b"DVDVIDEO-VMG", AssetFormat::Ifo),
    (b"DVDVIDEO-VTS", AssetFormat::Ifo),
    (b"\xfe\xff\x00<\x00?\x00x\x00m\x00l", AssetFormat::Xml),
    (b"\xff\xfe<\x00?\x00x\x00m\x00l\x00", AssetFormat::Xml),
    (b"<stata_dta>", AssetFormat::Dta),
    (b"BEGIN:VCARD", AssetFormat::Vcf),
    (b"SAS     8.0", AssetFormat::Sd2),
    (b"SAS     9.0", AssetFormat::Sd2),
    (b"SIMPLE  =  ", AssetFormat::Fits),
    (b"SR\x00\x04\x00\x00\x00\x0a\x00\x00\x04", AssetFormat::Idl),
    (b"d8:announce", AssetFormat::Torrent),
    (b"filedesc://", AssetFormat::Arc),
    (b"\x00\x00\x02\x00\x06\x04\x06\x00\x08\x00", AssetFormat::Wk1),
    (b"NITF02.000", AssetFormat::Ntf),
    (b"NITF02.100", AssetFormat::Ntf),
    (b"SAS     6.", AssetFormat::Sd2),
    (b"SAS     7.", AssetFormat::Sd2),
    (b"\xff\x06\x00\x00sNaPpY", AssetFormat::Sz),
    (b"\x0fSIBELIUS", AssetFormat::Sib),
    (b"#!AMR-WB\x0a", AssetFormat::AmrWb),
    (b"#EXTM3U\x0d\x0a", AssetFormat::M3u),
    (b"NITF01.10", AssetFormat::Ntf),
    (b"WordPro\x0d\xfb", AssetFormat::Lwp),
    (b"gimp xcf ", AssetFormat::Xcf),
    (b"\xef\xbb\xbfWEBVTT", AssetFormat::Vtt),
    (b"\x89PNG\x0d\x0a\x1a\x0a", AssetFormat::Png),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", AssetFormat::Ole2),
    (b"\x00\x00\x00\x08wide", AssetFormat::Mov),
    (b"\x00\x00\x1a\x00\x00\x10\x04\x00", AssetFormat::Wk3),
    (b"\x00\x00\x1a\x00\x02\x10\x04\x00", AssetFormat::Wk4),
    (b"\x00\x00\x1a\x00\x03\x10\x04\x00", AssetFormat::Lotus123),
    (b"\x00\x01\x00\x00Stan", AssetFormat::Accdb),
    (b"!<arch>\x0a", AssetFormat::Ar),
    (b";ELC\x13\x00\x00\x00", AssetFormat::Elc),
    (b"<MIFFile", AssetFormat::Mif),
    (b"ASTM-E57", AssetFormat::E57),
    (b"AT&TFORM", AssetFormat::Djvu),
    (b"JEOL.NMR", AssetFormat::Jdf),
    (b"RMN.LOEJ", AssetFormat::Jdf),
    (b"VjCD0100", AssetFormat::Cdx),
    (b"WordPro\x00", AssetFormat::Lwp),
    (b"ZXTape!\x1a", AssetFormat::Tzx),
    (b"conectix", AssetFormat::Vhd),
    (b"\x97JB2\x0d\x0a\x1a\x0a", AssetFormat::Jb2),
    (b"\xef\xbb\xbf%PDF-", AssetFormat::Pdf),
    (b"\xef\xbb\xbf<?xml", AssetFormat::Xml),
    (b"\x89HDF\x0d\x0a\x1a\x0a", AssetFormat::Hdf),
    (b"\x8aMNG\x0d\x0a\x1a\x0a", AssetFormat::Mng),
    (b"\x8bJNG\x0d\x0a\x1a\x0a", AssetFormat::Jng),
    (b"#EXTM3U", AssetFormat::M3u8),
    (b"StuffIt", AssetFormat::Sit),
    (b"GIF87a", AssetFormat::Gif),
    (b"GIF89a", AssetFormat::Gif),
    (b"7z\xbc\xaf'\x1c", AssetFormat::SevenZip),
    (b"Rar!\x1a\x07", AssetFormat::Rar),
    (b"WEBVTT", AssetFormat::Vtt),
    (b"\x00\x00\x02\x00\x01\x10", AssetFormat::Wb1),
    (b"\x00\x00\x02\x00\x02\x10", AssetFormat::Wb2),
    (b"\x00\x00\x02\x00\x04\x04", AssetFormat::Wks),
    (b"\x00\x00\x02\x00 Q", AssetFormat::Wq1),
    (b"\x00\x00\x02\x00!Q", AssetFormat::Wq2),
    (b"\x01\x00\x09\x00\x00\x03", AssetFormat::Wmf),
    (b"\x01\xda\x01\x01\x00\x03", AssetFormat::Rgb),
    (b"(DWF V", AssetFormat::Dwf),
    (b"/* XPM", AssetFormat::Xpm),
    (b"070701", AssetFormat::Cpio),
    (b"070702", AssetFormat::Cpio),
    (b"070707", AssetFormat::Cpio),
    (b"8BPS\x00\x01", AssetFormat::Psd),
    (b"8BPS\x00\x02", AssetFormat::Psd),
    (b"<Maker", AssetFormat::Mif),
    (b"AC1.40", AssetFormat::Dwg),
    (b"AC1.50", AssetFormat::Dwg),
    (b"AC2.10", AssetFormat::Dwg),
    (b"AC2.21", AssetFormat::Dwg),
    (b"AC2.22", AssetFormat::Dwg),
    (b"MATLAB", AssetFormat::Mat),
    (b"bplist", AssetFormat::Bplist),
    (b"caff\x00\x00", AssetFormat::Caf),
    (b"caff\x00\x01", AssetFormat::Caf),
    (b"caff\x00\x02", AssetFormat::Caf),
    (b"caff@\x00", AssetFormat::Caf),
    (b"caff\x80\x00", AssetFormat::Caf),
    (b"\xd7\xcd\xc6\x9a\x00\x00", AssetFormat::Wmf),
    (b"\xdb\xa5-\x00\x00\x00", AssetFormat::Doc),
    (b"\xfd7zXZ\x00", AssetFormat::Xz),
    (b"%PDF-", AssetFormat::Pdf),
    (b"#!AMR", AssetFormat::Amr),
    (b"%FDF-", AssetFormat::Fdf),
    (b"<?XML", AssetFormat::Xml),
    (b"<?php", AssetFormat::Php),
    (b"<?xml", AssetFormat::Xml),
    (b"<Book", AssetFormat::Mif),
    (b"=<ar>", AssetFormat::Ar),
    (b"AC1.2", AssetFormat::Dwg),
    (b"BEGMF", AssetFormat::Cgm),
    (b"MC0.0", AssetFormat::Dwg),
    (b"MOVI\x00", AssetFormat::SgiMovie),
    (b"MOVI\x01", AssetFormat::SgiMovie),
    (b"MOVI\x02", AssetFormat::SgiMovie),
    (b"MOVI\xfe", AssetFormat::SgiMovie),
    (b"MOVI\xff", AssetFormat::SgiMovie),
    (b"OTTO\x00", AssetFormat::Otf),
    (b"PO^Q`", AssetFormat::Doc),
    (b"SAP\x0d\x0a", AssetFormat::Sap),
    (b"WARC/", AssetFormat::Warc),
    (b"{\x5crtf", AssetFormat::Rtf),
    (b"PK\x03\x04", AssetFormat::Zip),
    (b"PK\x05\x06", AssetFormat::Zip),
    (b"PK\x07\x08", AssetFormat::Zip),
    (b"II*\x00", AssetFormat::Tiff),
    (b"MM\x00*", AssetFormat::Tiff),
    (b"II+\x00", AssetFormat::Tiff),
    (b"MM\x00+", AssetFormat::Tiff),
    (b"fLaC", AssetFormat::Flac),
    (b"OggS", AssetFormat::Ogg),
    (b"MThd", AssetFormat::Midi),
    (b"\x7fELF", AssetFormat::Elf),
    (b"MZ\x90\x00", AssetFormat::Pe),
    (b"\xca\xfe\xba\xbe", AssetFormat::Class),
    (b"wOFF", AssetFormat::Woff),
    (b"wOF2", AssetFormat::Woff2),
    (b"(\xb5/\xfd", AssetFormat::Zstd),
    (b"\x00asm", AssetFormat::Wasm),
    (b"MPX,", AssetFormat::Mpx),
    (b"MPX;", AssetFormat::Mpx),
    (b"v/1\x01", AssetFormat::Exr),
    (b"FORM", AssetFormat::Aiff),
    (b"\x00\x00\x01\x00", AssetFormat::Ico),
    (b"\x00\x00\x01\xb3", AssetFormat::Mpeg),
    (b"\x00\x00\x01\xba", AssetFormat::Mpeg),
    (b"\x00\x01\x00\x00", AssetFormat::Ttf),
    (b"\x00\x05\x16\x00", AssetFormat::Applefile),
    (b"\x00\x05\x16\x07", AssetFormat::Appledouble),
    (b"\x01\x00\x00\x00", AssetFormat::Emf),
    (b"\x03\x00\x08\x00", AssetFormat::Axml),
    (b"\x09\x00\x04\x00", AssetFormat::XlsSheet2),
    (b"\x09\x02\x06\x00", AssetFormat::XlsSheet3),
    (b"\x09\x04\x06\x00", AssetFormat::XlsSheet4),
    (b"\x0a\x0d\x0d\x0a", AssetFormat::Pcapng),
    (b"\x1aE\xdf\xa3", AssetFormat::Matroska),
    (b".RMF", AssetFormat::Rm),
    (b".snd", AssetFormat::Au),
    (b"<MML", AssetFormat::Mif),
    (b"<svg", AssetFormat::Svg),
    (b"AGD2", AssetFormat::Fh),
    (b"AGD3", AssetFormat::Fh),
    (b"AGD4", AssetFormat::Fh),
    (b"BPG\xfb", AssetFormat::Bpg),
    (b"CDF\x01", AssetFormat::Nc),
    (b"CDF\x02", AssetFormat::Nc),
    (b"Cr24", AssetFormat::Crx),
    (b"EP*\x00", AssetFormat::Mdi),
    (b"GRIB", AssetFormat::Grb),
    (b"IIN1", AssetFormat::Niff),
    (b"IIRO", AssetFormat::Orf),
    (b"IMPM", AssetFormat::Mod),
    (b"ITSF", AssetFormat::Chm),
    (b"LASF", AssetFormat::Las),
    (b"LZIP", AssetFormat::Lz),
    (b"MSCF", AssetFormat::Cab),
    (b"NES\x1a", AssetFormat::Nes),
    (b"PAR1", AssetFormat::Parquet),
    (b"PSID", AssetFormat::Sid),
    (b"SDPX", AssetFormat::Dpx),
    (b"UC2\x1a", AssetFormat::Uc2),
    (b"XPDS", AssetFormat::Dpx),
    (b"dex\x0a", AssetFormat::Dex),
    (b"icns", AssetFormat::Icns),
    (b"xar!", AssetFormat::Xar),
    (b"\xb1h\xde:", AssetFormat::Dcx),
    (b"\xbe\xba\xfe\xca", AssetFormat::MachoFat),
    (b"\xbf\xba\xfe\xca", AssetFormat::MachoFat),
    (b"\xc5\xd0\xd3\xc6", AssetFormat::Ps),
    (b"\xca\xfe\xba\xbf", AssetFormat::MachoFat),
    (b"\xce\xfa\xed\xfe", AssetFormat::Macho),
    (b"\xcf\xfa\xed\xfe", AssetFormat::Macho),
    (b"\xed\xab\xee\xdb", AssetFormat::Rpm),
    (b"\xfe7\x00#", AssetFormat::Doc),
    (b"\xfe\xed\xfa\xce", AssetFormat::Macho),
    (b"\xfe\xed\xfa\xcf", AssetFormat::Macho),
    (b"\xffO\xffQ", AssetFormat::J2c),
    (b"\xff\xd8\xff", AssetFormat::Jpeg),
    (b"\x1f\x8b\x08", AssetFormat::Gzip),
    (b"ID3", AssetFormat::Mp3),
];

/// The format of the longest signature `header` starts with.
pub fn known(header: &[u8]) -> Option<AssetFormat> {
    MAGICS.iter().find(|(magic, _)| header.starts_with(magic)).map(|&(_, format)| format)
}

/// The first bytes any recognized payload can start with: the first byte
/// of every signature, `R` (a RIFF container), 0x30 (a BER SEQUENCE, the
/// timestamped-data envelope), `B` and `b` (a calendar's `BEGIN`, which
/// may be folded or in lower case), and the bytes an XML document can
/// start with (`<`, the four XML white space bytes, 0x00 for UTF-16BE,
/// and the first byte of the UTF-8 and UTF-16 byte-order marks).
///
/// Every check of `sniff` needs its payload to start with one of these, so
/// a run whose first decoded byte is not one of them is not decoded
/// further. OpenNLP filters the same way on the base64 text (its
/// precomputed signature prefixes and the character tests
/// `startsWithSequence`, `startsWithCalendar` and `startsWithXml`). The
/// filter here may let through a payload those tests would stop, never the
/// reverse; the checks of `sniff` then reject it, so the result is the same.
pub static FIRST_BYTES: [bool; 256] = first_bytes();

const fn first_bytes() -> [bool; 256] {
    let mut set = [false; 256];
    let mut i = 0;
    while i < MAGICS.len() {
        set[MAGICS[i].0[0] as usize] = true;
        i += 1;
    }
    let extra = b"R0Bb< \t\r\n\x00\xef\xfe\xff";
    let mut k = 0;
    while k < extra.len() {
        set[extra[k] as usize] = true;
        k += 1;
    }
    set
}
