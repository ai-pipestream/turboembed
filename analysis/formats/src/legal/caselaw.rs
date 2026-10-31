use super::LegalError;
use super::json::{self, Json};
use crate::sample::DocumentSample;

/// Cases of the Caselaw Access Project's bulk JSON, in source order.
#[derive(Debug)]
pub struct CaselawReader {
    cases: std::vec::IntoIter<Json>,
}

impl CaselawReader {
    pub fn new(source: &str) -> Result<CaselawReader, LegalError> {
        Ok(CaselawReader { cases: json::parse_many(source)?.into_iter() })
    }
}

impl Iterator for CaselawReader {
    type Item = Result<DocumentSample<'static>, LegalError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.cases.next().map(|case| {
            let Json::Object(source, _) = &case else {
                return Err(LegalError::Json { offset: 0, expected: "a case object" });
            };
            let category = case
                .get("name_abbreviation")
                .and_then(Json::string)
                .ok_or(LegalError::MissingField("name_abbreviation"))?
                .field();
            let opinions = case
                .get("casebody")
                .and_then(|body| body.get("opinions"))
                .and_then(Json::array)
                .ok_or(LegalError::MissingField("casebody.opinions"))?;
            let mut tokens = Vec::new();
            for opinion in opinions {
                let text = opinion
                    .get("text")
                    .and_then(Json::string)
                    .ok_or(LegalError::MissingField("casebody.opinions[].text"))?;
                tokens.extend(text.tokens());
            }
            Ok(DocumentSample { category, tokens, source: *source })
        })
    }
}
