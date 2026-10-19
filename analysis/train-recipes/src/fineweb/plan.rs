//! Host extraction, window sampling, and FineWeb document selection.

use std::collections::{BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use turbo_eval::read::parquet::Parquet;

use super::store::{plan_file, read_err, reader_name};
use super::{Avoid, Result, Side, Store, WebError, WebFile, WebRecipe, excluded_host, files_of};

/// The host of a URL, in lowercase and without a leading `www.` or a
/// trailing dot: what lies between `://` (or a leading `//`) and the next
/// `/`, `?` or `#`, less a user name and a port. A string without either
/// is read from its start.
pub fn host(url: &str) -> String {
    let rest = match url.split_once("://") {
        Some((scheme, r)) if !scheme.contains(['/', '?', '#']) => r,
        _ => url.strip_prefix("//").unwrap_or(url),
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let authority = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    // A port follows the last colon, except inside an IPv6 address's
    // brackets.
    let h = match authority.rfind(':') {
        Some(i) if !authority[i..].contains(']') => &authority[..i],
        _ => authority,
    };
    let h = h.trim_end_matches('.').to_lowercase();
    match h.strip_prefix("www.") {
        Some(s) => s.to_owned(),
        None => h,
    }
}

/// The first `n` characters of `text`, or all of it when it has fewer.
pub fn window(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((i, _)) => &text[..i],
        None => text,
    }
}

/// A window trained on or held out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebSample {
    /// The language's index in [`WebRecipe::languages`].
    pub language: usize,
    /// The index of its file in the files list.
    pub file: usize,
    /// The document's row in its file.
    pub row: u64,
    /// The document's `id`.
    pub id: String,
    pub window: String,
}

/// How one file's documents were read and chosen.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileChoice {
    pub tatoeba: String,
    pub repo: String,
    pub path: String,
    pub side: Side,
    /// The documents wanted of the file, the most it gives.
    pub wanted: usize,
    pub file_rows: u64,
    pub file_row_groups: usize,
    pub row_groups_read: usize,
    pub rows_read: u64,
    /// Rows passed over for having fewer than [`super::WebSplit::min_chars`]
    /// characters.
    pub too_short: usize,
    /// Rows passed over for their host ([`super::EXCLUDED_HOSTS`]).
    pub excluded_host: usize,
    /// Windows passed over as one the language already has or an earlier
    /// row of the file gave (taken or not), or, held out, one any language
    /// trains on.
    pub duplicates: usize,
    /// Windows passed over for holding a Tatoeba sentence of their
    /// language from the other side of the split ([`Avoid`]).
    pub tatoeba_overlap: usize,
    /// Documents passed over at first for their host's share.
    pub host_deferred: usize,
    /// Of those, the ones taken afterwards to fill the file's share.
    pub host_taken_later: usize,
    pub chosen: usize,
}

/// The web documents trained on and held out.
#[derive(Clone, Debug, Default)]
pub struct WebPlan {
    /// Language by language in [`WebRecipe::languages`]' order, each source's in
    /// row order.
    pub training: Vec<WebSample>,
    /// The same way.
    pub held_out: Vec<WebSample>,
    /// Each file read, in the order of [`WebRecipe::wants`].
    pub files: Vec<FileChoice>,
}

/// Reads the documents of every file of `files` from the directory and
/// chooses each language's training and held-out windows, none holding a
/// sentence of `avoid`.
///
/// # Errors
///
/// A range missing or not the listed bytes, a file whose leading row
/// groups cannot be read, or a null `text` or `id`.
pub fn plan_web(files: &[WebFile], web: &WebRecipe, store: &Store, avoid: &Avoid) -> Result<WebPlan> {
    let (wants, split) = (web.wants(), &web.split);
    let chosen_files = files_of(&wants, files)?;
    let index: HashMap<(&str, &str), usize> =
        files.iter().enumerate().map(|(i, f)| ((f.repo.as_str(), f.path.as_str()), i)).collect();
    let mut plan = WebPlan::default();
    // Each language's windows, and every training window.
    let mut by_language: Vec<HashSet<String>> = vec![HashSet::new(); web.languages.len()];
    let mut training: HashSet<String> = HashSet::new();
    for (w, f) in wants.iter().zip(chosen_files) {
        let planned = plan_file(f, split.rows(w.documents), split, store)?;
        let leading = planned.leading.ok_or_else(|| WebError::Missing {
            path: store.file(f, *planned.ranges.known().last().expect("the tail is always known")),
        })?;
        let (head_span, groups) = planned.ranges.head.expect("known once the footer's first bytes are read");
        let head = store.get(f, head_span)?.ok_or_else(|| WebError::Missing { path: store.file(f, head_span) })?;
        let name = reader_name(f);
        let table = Parquet::open_head(&name, &head, &leading, groups).map_err(read_err)?;
        let texts = table.strings("text").map_err(read_err)?;
        let urls = table.strings("url").map_err(read_err)?;
        let ids = table.strings("id").map_err(read_err)?;
        let mut choice = FileChoice {
            tatoeba: web.languages[w.language].tatoeba.to_owned(),
            repo: f.repo.clone(),
            path: f.path.clone(),
            side: w.side,
            wanted: w.documents,
            file_rows: leading.rows(),
            file_row_groups: leading.row_groups(),
            row_groups_read: groups,
            rows_read: table.rows() as u64,
            too_short: 0,
            excluded_host: 0,
            duplicates: 0,
            tatoeba_overlap: 0,
            host_deferred: 0,
            host_taken_later: 0,
            chosen: 0,
        };
        // The rows that give a window, in order, with their hosts.
        let seen = &mut by_language[w.language];
        let overlap = match w.side {
            Side::Training => &avoid.training,
            Side::HeldOut => &avoid.held_out,
        };
        let mut candidates: Vec<(usize, String, &str)> = Vec::new();
        let mut in_file: HashSet<&str> = HashSet::new();
        for row in 0..table.rows() {
            let text = texts.at(row).map_err(read_err)?;
            if text.chars().take(split.min_chars).count() < split.min_chars {
                choice.too_short += 1;
                continue;
            }
            let h = host(urls.get(row).unwrap_or(""));
            if excluded_host(&h) {
                choice.excluded_host += 1;
                continue;
            }
            let win = window(text, split.window);
            let other_side = w.side == Side::HeldOut && training.contains(win);
            if other_side || seen.contains(win) || in_file.contains(win) {
                choice.duplicates += 1;
                continue;
            }
            if overlap.found(w.language, win) {
                choice.tatoeba_overlap += 1;
                continue;
            }
            in_file.insert(win);
            candidates.push((row, h, win));
        }
        let limit = split.host_limit(w.documents);
        let mut per_host: HashMap<&str, usize> = HashMap::new();
        let mut taken: Vec<usize> = Vec::new();
        let mut deferred: Vec<usize> = Vec::new();
        for (k, (_, h, _)) in candidates.iter().enumerate() {
            if taken.len() == w.documents {
                break;
            }
            let n = per_host.entry(h.as_str()).or_insert(0);
            if *n < limit {
                *n += 1;
                taken.push(k);
            } else {
                deferred.push(k);
            }
        }
        choice.host_deferred = deferred.len();
        for k in deferred {
            if taken.len() == w.documents {
                break;
            }
            taken.push(k);
            choice.host_taken_later += 1;
        }
        taken.sort_unstable();
        choice.chosen = taken.len();
        let file = index[&(f.repo.as_str(), f.path.as_str())];
        for k in taken {
            let (row, _, win) = candidates[k];
            let id = ids.at(row).map_err(read_err)?;
            seen.insert(win.to_owned());
            if w.side == Side::Training {
                training.insert(win.to_owned());
            }
            let sample =
                WebSample { language: w.language, file, row: row as u64, id: id.to_owned(), window: win.to_owned() };
            match w.side {
                Side::Training => plan.training.push(sample),
                Side::HeldOut => plan.held_out.push(sample),
            }
        }
        plan.files.push(choice);
    }
    Ok(plan)
}

/// The contents of the list of web documents trained on: for each, its
/// dataset's repository, its file, its row and its `id`, tab separated, a
/// line each in that order, every line ending in a line feed.
pub fn documents_file(plan: &WebPlan, files: &[WebFile]) -> Vec<u8> {
    let rows: BTreeSet<(&str, &str, u64, &str)> = plan
        .training
        .iter()
        .map(|s| (files[s.file].repo.as_str(), files[s.file].path.as_str(), s.row, s.id.as_str()))
        .collect();
    let mut out = String::new();
    for (repo, path, row, id) in rows {
        out.push_str(&format!("{repo}\t{path}\t{row}\t{id}\n"));
    }
    out.into_bytes()
}
