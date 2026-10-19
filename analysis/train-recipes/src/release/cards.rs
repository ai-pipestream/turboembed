use std::path::Path;

use turbo_eval::model_gate::ModelGateRecord;
use turbo_zoo::{Artifact, verify};

use super::ModelsRelease;
use crate::card::{CARD_FILE, GATE_FILE, GateCard, ModelCard, ModelFileCard};
use crate::error::{Error, Result, io};
use crate::langdetect::LanguageCard;
use crate::langdetect_web::{COMPARE_TATOEBA_FILE, COMPARE_WEB_FILE, ComparisonCard, WEB_GATE_FILE, WebLanguageCard};
use crate::ud_card::UdModelCard;

/// What [`super::entries`] reads of a model's card: the MASC recipes'
/// card, the UD recipes' one or the language detectors'.
pub(super) trait Card: Sized {
    fn read(path: &Path) -> Result<Self>;
    fn id(&self) -> &str;
    fn file(&self) -> &ModelFileCard;
    fn gate(&self) -> &GateCard;
    /// Checks the records written beside the card in `dir` against it:
    /// the gate record ([`GATE_FILE`]), and for the web language detector
    /// its second gate and its two comparisons as well.
    fn check_records(&self, dir: &Path) -> Result<()> {
        check_gate(self.id(), self.file(), GATE_FILE, self.gate(), &ModelGateRecord::read(&dir.join(GATE_FILE))?)
    }
    /// The model's zoo entry, its files at their addresses in `release`.
    fn artifact(&self, release: &ModelsRelease) -> Result<Artifact>;
}

impl Card for ModelCard {
    fn read(path: &Path) -> Result<Self> {
        ModelCard::read(path)
    }
    fn id(&self) -> &str {
        &self.id
    }
    fn file(&self) -> &ModelFileCard {
        &self.file
    }
    fn gate(&self) -> &GateCard {
        &self.gate
    }
    fn artifact(&self, release: &ModelsRelease) -> Result<Artifact> {
        ModelCard::artifact(self, &release.url(&self.file.name))
    }
}

impl Card for UdModelCard {
    fn read(path: &Path) -> Result<Self> {
        UdModelCard::read(path)
    }
    fn id(&self) -> &str {
        &self.id
    }
    fn file(&self) -> &ModelFileCard {
        &self.file
    }
    fn gate(&self) -> &GateCard {
        &self.gate
    }
    fn artifact(&self, release: &ModelsRelease) -> Result<Artifact> {
        UdModelCard::artifact(self, &release.url(&self.file.name))
    }
}

impl Card for LanguageCard {
    fn read(path: &Path) -> Result<Self> {
        LanguageCard::read(path)
    }
    fn id(&self) -> &str {
        &self.id
    }
    fn file(&self) -> &ModelFileCard {
        &self.file
    }
    fn gate(&self) -> &GateCard {
        &self.gate
    }
    fn artifact(&self, release: &ModelsRelease) -> Result<Artifact> {
        LanguageCard::artifact(self, &release.url(&self.file.name), &release.url(&self.training_sentences.name))
    }
}

impl Card for WebLanguageCard {
    fn read(path: &Path) -> Result<Self> {
        WebLanguageCard::read(path)
    }
    fn id(&self) -> &str {
        &self.id
    }
    fn file(&self) -> &ModelFileCard {
        &self.file
    }
    fn gate(&self) -> &GateCard {
        &self.gate
    }
    fn check_records(&self, dir: &Path) -> Result<()> {
        let read = |file: &str| ModelGateRecord::read(&dir.join(file));
        check_gate(&self.id, &self.file, GATE_FILE, &self.gate, &read(GATE_FILE)?)?;
        check_gate(&self.id, &self.file, WEB_GATE_FILE, &self.web_gate, &read(WEB_GATE_FILE)?)?;
        for (file, card) in
            [(COMPARE_TATOEBA_FILE, &self.compared_on_sentences), (COMPARE_WEB_FILE, &self.compared_on_windows)]
        {
            let record = read(file)?;
            let mismatch = |message: String| Err(Error::InvalidCard { field: "compared", message });
            if record.candidate.id != self.id || record.candidate.sha256 != self.file.sha256 {
                return mismatch(format!(
                    "{file} scores {} (SHA-256 {}), not {} (SHA-256 {})",
                    record.candidate.id, record.candidate.sha256, self.id, self.file.sha256
                ));
            }
            if record.reference.id != self.compared || record.reference.sha256 != self.compared_sha256 {
                return mismatch(format!(
                    "{file} compares with {} (SHA-256 {}), not {} (SHA-256 {})",
                    record.reference.id, record.reference.sha256, self.compared, self.compared_sha256
                ));
            }
            if ComparisonCard::of(&record, file) != *card {
                return mismatch(format!("{file} of {} has other scores than its card", self.id));
            }
        }
        Ok(())
    }
    fn artifact(&self, release: &ModelsRelease) -> Result<Artifact> {
        WebLanguageCard::artifact(
            self,
            &release.url(&self.file.name),
            &release.url(&self.training_sentences.name),
            &release.url(&self.training_documents.name),
        )
    }
}

/// The zoo entry of model `id` of `release`, from its directory in
/// `models`.
pub(super) fn entry<C: Card>(models: &Path, id: &str, release: &ModelsRelease) -> Result<Artifact> {
    let dir = models.join(id);
    let card = C::read(&dir.join(CARD_FILE))?;
    if card.id() != id {
        return Err(Error::InvalidCard { field: "id", message: format!("{} in the directory of {id}", card.id()) });
    }
    card.check_records(&dir)?;
    let artifact = card.artifact(release)?;
    for file in &artifact.files {
        let path = dir.join(file.path.as_str());
        let bytes = std::fs::read(&path).map_err(io(&path))?;
        verify(&artifact, file, &bytes)?;
    }
    Ok(artifact)
}

/// Checks that `record`, read from `record_file`, is the gate of model
/// `id` (its file `file`), with the outcome its card gives (`gate`), so a
/// gate record published beside a model never contradicts its card.
fn check_gate(
    id: &str,
    file: &ModelFileCard,
    record_file: &str,
    gate: &GateCard,
    record: &ModelGateRecord,
) -> Result<()> {
    let mismatch = |message: String| Err(Error::InvalidCard { field: "gate", message });
    if record.candidate.id != id || record.candidate.sha256 != file.sha256 {
        return mismatch(format!(
            "{record_file} gates {} (SHA-256 {}), not {id} (SHA-256 {})",
            record.candidate.id, record.candidate.sha256, file.sha256
        ));
    }
    let of = GateCard {
        record: record_file.to_owned(),
        ..GateCard::of(record, gate.candidate_tag_format, gate.reference_tag_format)
    };
    if of != *gate {
        return mismatch(format!("{record_file} of {id} has another outcome than its card"));
    }
    Ok(())
}
