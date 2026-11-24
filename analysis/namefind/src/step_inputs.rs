//! What the entity steps share: each document is built only when every
//! layer the step reads is complete on it.

use std::any::Any;

use turbo_analysis::{Document, Error, KeyId, Status};

/// `Ok` when every layer in `reads` is complete on `doc`; else the first
/// that is not: skipped or failed (as [`Error::Skipped`], with why) or
/// absent ([`Error::Missing`]).
pub(crate) fn inputs_complete(doc: &Document, reads: &[KeyId]) -> Result<(), Error> {
    for &k in reads {
        match doc.status(k) {
            Some(Status::Complete) => {}
            Some(Status::Skipped(why)) => return Err(Error::Skipped(format!("{k} was skipped: {why}"))),
            Some(Status::Failed(e)) => return Err(Error::Skipped(format!("{k} failed: {e}"))),
            None => return Err(Error::Missing(k)),
        }
    }
    Ok(())
}

/// A step's output over `docs`: `build` of each document whose `reads`
/// are all complete ([`inputs_complete`]), else why not, boxed for
/// [`Step::store`].
pub(crate) fn per_document<T: Send + 'static>(
    docs: &[Document],
    reads: &[KeyId],
    build: impl Fn(&Document) -> Result<T, Error>,
) -> Box<dyn Any + Send> {
    let out: Vec<Result<T, Error>> = docs
        .iter()
        .map(|d| {
            inputs_complete(d, reads)?;
            build(d)
        })
        .collect();
    Box::new(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Key, Provenance};
    use turbo_analysis_layers::Entities;

    #[test]
    fn inputs_are_complete_only_when_every_layer_read_is() {
        let mut doc = Document::new("a b");
        let done = Key::<turbo_analysis::View>::DEFAULT;
        let (skipped, failed, absent) =
            (Key::<Entities>::named("skipped"), Key::<Entities>::named("failed"), Key::<Entities>::named("absent"));
        doc.skip(skipped.id(), "off".into(), Provenance::by("t")).unwrap();
        doc.fail(failed.id(), Error::Missing(absent.id()), Provenance::by("t")).unwrap();
        assert_eq!(inputs_complete(&doc, &[]), Ok(()));
        assert_eq!(inputs_complete(&doc, &[done.id()]), Ok(()));
        let why = |ids: &[KeyId]| match inputs_complete(&doc, ids) {
            Err(Error::Skipped(why)) => why,
            other => panic!("{other:?}"),
        };
        assert!(why(&[done.id(), skipped.id(), failed.id()]).ends_with("was skipped: off"));
        assert!(why(&[failed.id(), skipped.id()]).contains(" failed: "));
        assert_eq!(inputs_complete(&doc, &[done.id(), absent.id()]), Err(Error::Missing(absent.id())));
    }

    #[test]
    fn per_document_builds_only_documents_whose_inputs_are_complete() {
        let mut skipped = Document::new("b");
        let input = Key::<Entities>::named("input");
        skipped.skip(input.id(), "off".into(), Provenance::by("t")).unwrap();
        let docs = [Document::new("a"), skipped];
        let view = Key::<turbo_analysis::View>::DEFAULT.id();
        let out = per_document(&docs, &[view], |d| Ok(d.has(view) as usize));
        let out = *out.downcast::<Vec<Result<usize, Error>>>().unwrap();
        assert_eq!(out, [Ok(1), Ok(1)]);
        let out = per_document(&docs, &[input.id()], |_| Ok(0u8));
        let out = *out.downcast::<Vec<Result<u8, Error>>>().unwrap();
        assert!(matches!(out[0], Err(Error::Missing(_))) && matches!(out[1], Err(Error::Skipped(_))));
    }
}
