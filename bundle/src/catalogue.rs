//! Models `turbo-bundle` can make by name: dense, static and sparse
//! embeddings and rerankers, each a recipe in bundle/recipes pinned to a
//! commit of its repository, with the SHA-256 of every file it fetches.
//! Nothing is fetched until a `fetch` or `make` names the model; no weights
//! are in this repository or its release archives. The other recipes are
//! made by path ([`BY_PATH`]).

/// Each model's id and its recipe.
pub const MODELS: &[(&str, &str)] = &[
    ("BAAI/bge-small-en-v1.5", include_str!("../recipes/bge-small-en-v1.5.json")),
    ("BAAI/bge-base-en-v1.5", include_str!("../recipes/bge-base-en-v1.5.json")),
    ("BAAI/bge-large-en-v1.5", include_str!("../recipes/bge-large-en-v1.5.json")),
    ("BAAI/bge-m3", include_str!("../recipes/bge-m3.json")),
    ("cross-encoder/ms-marco-MiniLM-L6-v2", include_str!("../recipes/ms-marco-minilm-l6-v2-rerank.json")),
    ("BAAI/bge-reranker-base", include_str!("../recipes/bge-reranker-base.json")),
    ("minishlab/potion-base-2M", include_str!("../recipes/potion/potion-base-2M.json")),
    ("minishlab/potion-base-4M", include_str!("../recipes/potion/potion-base-4M.json")),
    ("minishlab/potion-base-8M", include_str!("../recipes/potion/potion-base-8M.json")),
    ("minishlab/potion-base-32M", include_str!("../recipes/potion/potion-base-32M.json")),
    ("minishlab/potion-retrieval-32M", include_str!("../recipes/potion/potion-retrieval-32M.json")),
    ("minishlab/potion-science-32M", include_str!("../recipes/potion/potion-science-32M.json")),
    ("minishlab/potion-code-16M", include_str!("../recipes/potion/potion-code-16M.json")),
    ("minishlab/potion-code-16M-v2", include_str!("../recipes/potion/potion-code-16M-v2.json")),
    ("minishlab/potion-multilingual-128M", include_str!("../recipes/potion/potion-multilingual-128M.json")),
];

/// The recipes under bundle/recipes made only by their path, and why.
pub const BY_PATH: &[(&str, &str)] = &[
    ("all-minilm-l6-v2.json", "reads a calibration file beside the recipe"),
    ("all-minilm-l6-v2-static.json", "reads a quality file beside the recipe"),
    ("bge-base-en-v1.5-static.json", "reads a quality file beside the recipe"),
    ("bge-m3-8192.json", "a second build of BAAI/bge-m3, whose id the catalogue gives bge-m3.json"),
    ("roberta-base-squad2.json", "trained on SQuAD 2.0 (CC BY-SA 4.0); the zoo fetches it after acceptance"),
    ("dslim-bert-base-ner.json", "trained on CoNLL-2003, whose text is not under an open licence"),
    ("sat-1l-sm.json", "trained on UD treebanks of several licences, some share-alike"),
];

/// The recipe of the model `id`, if the catalogue has it.
pub fn recipe(id: &str) -> Option<&'static str> {
    MODELS.iter().find(|(m, _)| *m == id).map(|(_, r)| *r)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::Path;

    use super::*;
    use crate::recipe::Recipe;

    /// Every `.json` file under `dir`, as a path relative to the recipes.
    fn recipe_files(dir: &Path, under: &str, out: &mut BTreeSet<String>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            let name = format!("{under}{}", p.file_name().unwrap().to_str().unwrap());
            if p.is_dir() {
                recipe_files(&p, &format!("{name}/"), out);
            } else if name.ends_with(".json") {
                out.insert(name);
            }
        }
    }

    #[test]
    fn every_recipe_is_made_by_name_or_listed_as_made_by_path() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes");
        let mut files = BTreeSet::new();
        recipe_files(&dir, "", &mut files);
        let by_path: BTreeSet<String> = BY_PATH.iter().map(|(f, _)| f.to_string()).collect();
        let mut named = BTreeSet::new();
        for (id, json) in MODELS {
            let r = Recipe::load(Path::new(id)).unwrap_or_else(|e| panic!("{id}: {e}"));
            assert_eq!(r.str_at("/model/id").unwrap(), *id, "a catalogue id is its recipe's model id");
            assert!(r.local.is_empty(), "{id}: a recipe made by name reads no file beside it");
            let file = files.iter().find(|f| std::fs::read_to_string(dir.join(f)).unwrap() == *json).unwrap();
            named.insert(file.clone());
        }
        assert!(named.is_disjoint(&by_path));
        let covered: BTreeSet<String> = named.union(&by_path).cloned().collect();
        assert_eq!(covered, files, "a recipe file is neither in MODELS nor in BY_PATH");
    }
}
