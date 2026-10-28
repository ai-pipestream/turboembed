use super::*;

/// Offers `SatAnnotator` to pipelines built from configuration, as "sat".
/// The spec's location is the bundle directory; its options:
///
/// - `threshold`: a probability; by default 0.025 for a model of limited
///   lookahead and 0.25 for the others, as wtpsplit picks;
/// - `stride` (0, which is 64 or the block size when that is smaller)
///   and `block_size` (0, all a row has room for), in tokens;
/// - `newlines`: `split` (the default) to end a sentence at every line
///   feed, or `model` to leave them to the model;
/// - `device`: `cpu` (the default), `select`, a device index, or a backend
///   name such as `cuda`;
/// - `precision`: `model` (the default), `fastest` or `exact`.
pub struct SatProvider;

fn parse_options(spec: &Spec) -> Result<SatOptions, String> {
    fn number<T: std::str::FromStr>(spec: &Spec, key: &str, default: &str) -> Result<T, String> {
        let v = spec.option(key, default);
        v.parse().map_err(|_| format!("{key} {v:?} is not a number"))
    }
    let device = match spec.option("device", "cpu") {
        "cpu" => Device::Cpu,
        "select" => Device::Select,
        d => match d.parse() {
            Ok(i) => Device::Index(i),
            Err(_) => Device::Backend(d.to_owned()),
        },
    };
    let precision = match spec.option("precision", "model") {
        "model" => Precision::Model,
        "fastest" => Precision::Fastest,
        "exact" => Precision::Exact,
        p => return Err(format!("precision {p:?}: model, fastest or exact")),
    };
    let split_on_newlines = match spec.option("newlines", "split") {
        "split" => true,
        "model" => false,
        n => return Err(format!("newlines {n:?}: split or model")),
    };
    Ok(SatOptions {
        encoder: EncoderOptions { device, precision, ..EncoderOptions::default() },
        threshold: match spec.option("threshold", "") {
            "" => None,
            _ => Some(number(spec, "threshold", "")?),
        },
        stride: number(spec, "stride", "0")?,
        block_size: number(spec, "block_size", "0")?,
        split_on_newlines,
        label: BOUNDARY_LABEL,
    })
}

impl Provider<dyn Step> for SatProvider {
    fn name(&self) -> &'static str {
        "sat"
    }

    fn supports(&self, spec: &Spec) -> bool {
        spec.location.is_some()
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let dir = spec.location.as_deref().ok_or_else(|| Error::Pipeline("sat: no bundle location".into()))?;
        let o = parse_options(spec).map_err(|e| Error::Pipeline(format!("sat: {e}")))?;
        let a = SatAnnotator::open(dir, &o).map_err(|e| Error::Pipeline(format!("sat: {}: {e}", dir.display())))?;
        Ok(Box::new(Bound::new(a)))
    }
}
