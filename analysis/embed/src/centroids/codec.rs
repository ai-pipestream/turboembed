use super::*;
impl LabelCentroids {
    /// Serializes these centroids to the plain typed `TCENT001` binary format.
    pub fn write_to<W: Write>(&self, mut w: W) -> io::Result<()> {
        w.write_all(CENTROID_MAGIC)?;
        w.write_all(&self.dim.to_le_bytes())?;
        w.write_all(&(self.tags.len() as u32).to_le_bytes())?;
        w.write_all(&[self.norm as u8])?;
        w.write_all(&self.temperature.to_le_bytes())?;
        write_str(&mut w, &self.model.id)?;
        write_str(&mut w, &self.model.revision)?;
        write_str(&mut w, &self.model.sha256)?;
        write_str(&mut w, &self.tags.name)?;
        for i in 0..self.tags.len() {
            write_str(&mut w, self.tags.label(Tag(i as u16)))?;
        }
        for &val in &self.centroids {
            w.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Serializes these centroids into an in-memory byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to(&mut buf).expect("writing to Vec<u8> cannot fail");
        buf
    }

    /// Writes these centroids to `path` in the `TCENT001` binary format.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let bytes = self.to_bytes();
        fs::write(path, bytes)
    }

    /// Loads `LabelCentroids` from a byte slice in the `TCENT001` binary
    /// format.
    pub fn from_bytes(bytes: &[u8]) -> Result<LabelCentroids, String> {
        let mut cur = Cursor::new(bytes);
        let magic = cur.read_array::<8>()?;
        if &magic != CENTROID_MAGIC {
            return Err("invalid LabelCentroids header (expected TCENT001)".into());
        }
        let dim = cur.read_u32()?;
        let n_labels = cur.read_u32()? as usize;
        let norm = CentroidNorm::from_byte(cur.read_u8()?)?;
        let temperature = cur.read_f64()?;
        let model = ModelRef { id: cur.read_string()?, revision: cur.read_string()?, sha256: cur.read_string()? };
        let tag_set_name = cur.read_string()?;
        if n_labels == 0 {
            return Err("TCENT001 file has 0 labels".into());
        }
        if dim == 0 {
            return Err("TCENT001 file has dim 0".into());
        }
        if n_labels > usize::from(u16::MAX) + 1 {
            return Err("TCENT001 label count exceeds the u16 tag range".into());
        }
        let mut labels = Vec::new();
        for _ in 0..n_labels {
            labels.push(cur.read_string()?);
        }
        let total =
            n_labels.checked_mul(dim as usize).ok_or_else(|| "TCENT001 dimensions overflow usize".to_string())?;
        if total > cur.remaining() / size_of::<f32>() {
            return Err("unexpected EOF in centroid values".into());
        }
        let mut centroids = Vec::with_capacity(total);
        for _ in 0..total {
            centroids.push(cur.read_f32()?);
        }
        cur.finish()?;
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let tags = TagSet::new(&tag_set_name, &label_refs);
        Self::from_centroids(model, dim, tags, centroids, temperature, norm)
    }

    /// Reads `LabelCentroids` from any reader in the `TCENT001` format.
    pub fn read_from<R: Read>(mut r: R) -> Result<LabelCentroids, String> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        Self::from_bytes(&buf)
    }

    /// Loads `LabelCentroids` from `path`.
    pub fn load(path: &Path) -> Result<LabelCentroids, String> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }
}
