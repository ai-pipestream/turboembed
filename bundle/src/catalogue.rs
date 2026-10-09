//! The models `turbo-bundle` can make by name: Model2Vec's potion models,
//! each a recipe in bundle/recipes/potion pinned to a commit of its
//! repository, with the SHA-256 of every file it fetches. Nothing is
//! fetched until a `fetch` or `make` names the model; no weights are in
//! this repository or its release archives.

/// Each model's id and its recipe.
pub const MODELS: &[(&str, &str)] = &[
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

/// The recipe of the model `id`, if the catalogue has it.
pub fn recipe(id: &str) -> Option<&'static str> {
    MODELS.iter().find(|(m, _)| *m == id).map(|(_, r)| *r)
}
