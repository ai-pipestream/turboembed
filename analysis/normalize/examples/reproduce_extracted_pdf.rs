//! Fetch and verify the evaluation extracts without rewriting fixtures or labels.
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;
use turbo_goldens::{escape::unescape, hex::to_hex, sha256};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
struct Source {
    id: &'static str,
    url: &'static str,
    pdf_sha: &'static str,
    text_sha: &'static str,
    first: u32,
    last: u32,
    mupdf: bool,
}
const SOURCES: &[Source] = &[
    Source {
        id: "P19-1423",
        url: "https://aclanthology.org/P19-1423.pdf",
        pdf_sha: "f9ee341ae788143c48a5d424812ebd40c8f1312b7eaffcbd52c0aa3fdb79f63c",
        text_sha: "4a4c955ebcaac4f100a3eee0d9755b0febaa72394695e172f422f60715956f1c",
        first: 2,
        last: 3,
        mupdf: false,
    },
    Source {
        id: "2020.acl-main.1",
        url: "https://aclanthology.org/2020.acl-main.1.pdf",
        pdf_sha: "bd2680310de89488ca086aa8378ed095ecf5e4b56588812f62b11667de66d70e",
        text_sha: "61d8ea51cba932c21984731f870eb08efa34e280b91d20c81487577a5d8c86c8",
        first: 2,
        last: 3,
        mupdf: false,
    },
    Source {
        id: "2022.acl-long.14-mupdf",
        url: "https://aclanthology.org/2022.acl-long.14.pdf",
        pdf_sha: "b88c8b84ca06bb72965981c29f2c04236c9f8006c9b89cfe7d653134fb1a1b88",
        text_sha: "ff84db26480ba60169e7e8c8a1803d2a297de852638a5a8f79609e540230c8b6",
        first: 5,
        last: 5,
        mupdf: true,
    },
    Source {
        id: "jmlr-25-0035",
        url: "https://jmlr.org/papers/volume26/25-0035/25-0035.pdf",
        pdf_sha: "e5b2e098ca661463f69e4328bb74ad6bf1a0402d4eec757ebcafabc747d70936",
        text_sha: "f902f5558cea9702dd1184e240f71109605b9fe9c1372be649e64c5d45becd83",
        first: 2,
        last: 7,
        mupdf: false,
    },
    Source {
        id: "leodar-111",
        url: "https://data.mendeley.com/public-files/datasets/9mrxwtfh6w/files/e83d0706-ea0f-466e-b3fb-e95a9ccfd802/file_downloaded",
        pdf_sha: "5fa80011faf2661c650cc4d9b7c8cd0e799e5276b29074c81ae51417a721b4e8",
        text_sha: "51cde9a82c4603dcb3696521ce6f41221cfc026a4bb2b41c4ce4e8e022c3079b",
        first: 111,
        last: 111,
        mupdf: false,
    },
    Source {
        id: "leodar-116",
        url: "https://data.mendeley.com/public-files/datasets/9mrxwtfh6w/files/e83d0706-ea0f-466e-b3fb-e95a9ccfd802/file_downloaded",
        pdf_sha: "5fa80011faf2661c650cc4d9b7c8cd0e799e5276b29074c81ae51417a721b4e8",
        text_sha: "116f31dc10e2ddc8075b02f548f5d2c39e226b9e98a7ae3b35b6692239fc43b5",
        first: 116,
        last: 116,
        mupdf: false,
    },
];
fn require(ok: bool, detail: impl Into<String>) -> Result<()> {
    if ok { Ok(()) } else { Err(detail.into().into()) }
}
fn version(program: &str, expected: &str) -> Result<()> {
    let output = Command::new(program).arg("-v").output()?;
    let text = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    require(output.status.success() && text.contains(expected), format!("requires {expected}"))
}
fn reproduce(source: &Source, dir: &Path) -> Result<()> {
    let pdf = dir.join(format!("{}.pdf", source.id));
    if !pdf.exists() {
        let status = Command::new("curl")
            .args(["--fail", "--location", "--silent", "--show-error", "--max-time", "60"])
            .arg(source.url)
            .arg("--output")
            .arg(&pdf)
            .status()?;
        require(status.success(), format!("download failed: {}", source.id))?;
    }
    require(to_hex(&sha256(&fs::read(&pdf)?)) == source.pdf_sha, format!("PDF changed: {}", source.id))?;
    let extracted = dir.join(format!("{}.txt", source.id));
    let status = if source.mupdf {
        Command::new("mutool")
            .args(["draw", "-q", "-F", "txt", "-o"])
            .arg(&extracted)
            .arg(&pdf)
            .arg(format!("{}-{}", source.first, source.last))
            .status()?
    } else {
        Command::new("pdftotext")
            .args(["-layout", "-enc", "UTF-8", "-f"])
            .arg(source.first.to_string())
            .arg("-l")
            .arg(source.last.to_string())
            .arg(&pdf)
            .arg(&extracted)
            .status()?
    };
    require(status.success(), format!("extraction failed: {}", source.id))?;
    let raw = fs::read(extracted)?;
    require(to_hex(&sha256(&raw)) == source.text_sha, format!("extraction changed: {}", source.id))?;
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("data/extracted-pdf").join(format!("{}.escaped", source.id));
    let encoded = fs::read_to_string(fixture)?;
    let decoded = unescape(encoded.trim_end_matches('\n'))?;
    require(decoded.as_bytes() == raw, format!("fixture differs: {}", source.id))?;
    println!("{}: PDF, extracted text and decoded fixture match", source.id);
    Ok(())
}
fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let dir = args.next().ok_or("usage: reproduce_extracted_pdf OUTPUT_DIRECTORY")?;
    require(args.next().is_none(), "usage: reproduce_extracted_pdf OUTPUT_DIRECTORY")?;
    version("pdftotext", "pdftotext version 26.07.0")?;
    version("mutool", "mutool version 1.27.0")?;
    let dir = Path::new(&dir);
    fs::create_dir_all(dir)?;
    for source in SOURCES {
        reproduce(source, dir)?;
    }
    Ok(())
}
