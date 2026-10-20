//! The features of an anaphor and candidate antecedent pair, and of the
//! option that starts a new chain (`CorefContextGenerator`), after
//! Durrett and Klein (EMNLP 2013), "Easy Victories and Uphill Battles in
//! Coreference Resolution".
//!
//! Features are opaque `name=value` strings. With contextual token
//! vectors, each mention also has a span vector, the mean of its token
//! vectors, and a pair adds their products (`vp<d>`) and absolute
//! differences (`vd<d>`) as real-valued features; the new-chain option
//! adds the anaphor's span vector (`vn<d>`).

use crate::mention::Mention;
use crate::sieve::Resolver;

mod vectors;

pub(crate) use vectors::cosine;

/// Name prefix of the products of two span vectors.
pub const PRODUCT: &str = "vp";
/// Name prefix of the absolute differences of two span vectors.
pub const DIFFERENCE: &str = "vd";
/// Name prefix of the anaphor's span vector for the new-chain option.
pub const NEW_CHAIN: &str = "vn";

/// The features of one option: binary features by name and, with
/// contextual vectors, the span vectors.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Features {
    pub names: Vec<String>,
    pub anaphor: Option<Vec<f32>>,
    /// The candidate's span vector, `None` for the new-chain option.
    pub antecedent: Option<Vec<f32>>,
}

fn kind(m: &Mention) -> &'static str {
    if m.pronoun() {
        "pronoun"
    } else if m.named_entity() {
        "entity"
    } else if m.proper {
        "proper"
    } else {
        "nominal"
    }
}

fn first(m: &Mention) -> &str {
    m.words.first().map_or("", String::as_str)
}

fn shape(m: &Mention) -> &'static str {
    if m.proper {
        "proper"
    } else if m.indefinite {
        "indefinite"
    } else {
        "definite"
    }
}

/// A count's distance and length bucket.
fn bucket(value: i64) -> String {
    match value {
        ..=3 => value.to_string(),
        4..=7 => "4-7".into(),
        8..=15 => "8-15".into(),
        _ => "16+".into(),
    }
}

/// A type as Java string concatenation prints it: `null` for none.
fn ty(m: &Mention) -> &str {
    m.ty.as_deref().unwrap_or("null")
}

fn sieve(features: &mut Vec<String>, name: &str, holds: bool, kinds: &str) {
    if holds {
        features.push(name.to_owned());
        features.push(format!("{name}|{kinds}"));
    }
}

impl Resolver<'_> {
    fn position_in_sentence(&self, j: usize) -> usize {
        let s = self.mentions[j].sentence;
        (0..j).rev().take_while(|&i| self.mentions[i].sentence == s).count()
    }

    /// The features of the option that leaves anaphor `j` a new chain.
    pub fn new_chain_features(&mut self, j: usize) -> Result<Features, String> {
        let a = &self.mentions[j];
        let k = format!("new|{}", kind(a));
        let earlier_head = self.mentions[..j].iter().any(|m| m.head == a.head);
        let earlier_exact = self.mentions[..j].iter().any(|m| m.normalized == a.normalized);
        let mut f = vec![
            k.clone(),
            format!("{k}|head={}", a.head),
            format!("{k}|first={}", first(a)),
            format!("{k}|shape={}", shape(a)),
            format!("{k}|length={}", bucket(a.words.len() as i64)),
            format!("{k}|type={}", ty(a)),
            format!("{k}|sentence={}", bucket(a.sentence as i64)),
            format!("{k}|position={}", bucket(self.position_in_sentence(j) as i64)),
            format!("{k}|earlierHead={earlier_head}"),
            format!("{k}|earlierExact={earlier_exact}"),
        ];
        if a.pronoun() {
            f.push(format!("{k}|form={}", a.head));
        }
        let anaphor = if self.token_vectors.is_some() { Some(self.span_vector(j)) } else { None };
        Ok(Features { names: f, anaphor, antecedent: None })
    }

    /// The features of candidate antecedent `i` for anaphor `j`.
    pub fn features(&mut self, i: usize, j: usize) -> Result<Features, String> {
        let (ant_pronoun, ana_pronoun) = (self.mentions[i].pronoun(), self.mentions[j].pronoun());
        let kinds = format!("{}>{}", kind(&self.mentions[j]), kind(&self.mentions[i]));
        let mut f: Vec<String> = Vec::with_capacity(80);
        let size = bucket(self.clusters.size(i) as i64);
        f.push(format!("kinds={kinds}"));
        f.push(format!("antClusterSize={size}"));
        f.push(format!("{kinds}|antClusterSize={size}"));
        let cluster_exact = self.clusters.normalized_forms(i).contains(&self.mentions[j].normalized);
        f.push(format!("clusterExact={cluster_exact}"));
        let cluster_head = self.clusters.heads(i).contains(&self.mentions[j].head);
        f.push(format!("clusterHead={cluster_head}"));
        f.push(format!("antPosition={}", bucket(self.position_in_sentence(i) as i64)));
        self.speaker_of(i);
        let speakers = self.speaker_of.as_ref().expect("computed by speaker_of");
        let same_speaker = speakers[i] == speakers[j];
        f.push(format!("sameSpeaker={same_speaker}"));
        if ant_pronoun {
            f.push(format!("antForm={}", self.mentions[i].head));
            f.push(format!("antForm={}|anaHead={}", self.mentions[i].head, self.mentions[j].head));
        }
        if self.vectors.is_some() && !ana_pronoun {
            self.similarity_features(i, j, &kinds, &mut f)?;
        }
        sieve(&mut f, "exact", self.exact_match(i, j), &kinds);
        sieve(&mut f, "relaxed", self.relaxed_string_match(i, j), &kinds);
        sieve(&mut f, "acronym", self.acronym(i, j), &kinds);
        let holds = self.person_name(i, j);
        sieve(&mut f, "personName", holds, &kinds);
        let holds = self.strict_head_match(i, j, true, true);
        sieve(&mut f, "strictHead", holds, &kinds);
        let holds = self.strict_head_match(i, j, true, false);
        sieve(&mut f, "headNoModifiers", holds, &kinds);
        let holds = self.strict_head_match(i, j, false, true);
        sieve(&mut f, "headNoInclusion", holds, &kinds);
        let holds = self.proper_head_match(i, j);
        sieve(&mut f, "properHead", holds, &kinds);
        let holds = self.relaxed_head_match(i, j);
        sieve(&mut f, "relaxedHead", holds, &kinds);
        let holds = self.clusters.attributes_agree(i, j);
        sieve(&mut f, "agree", holds, &kinds);
        let holds = self.clusters.numbers_agree(i, j);
        sieve(&mut f, "numberAgree", holds, &kinds);
        if ana_pronoun {
            let holds = self.pronoun_match(i, j);
            sieve(&mut f, "pronounMatch", holds, &kinds);
        }
        let ant_first_in_cluster = self.clusters.find(i) == i;
        let (ant, ana) = (&self.mentions[i], &self.mentions[j]);
        let same_head = ana.head == ant.head;
        f.push(format!("sameHead={same_head}"));
        f.push(format!("anaHead={}", ana.head));
        f.push(format!("antHead={}", ant.head));
        f.push(format!("heads={}|{}", ana.head, ant.head));
        f.push(format!("anaFirst={}", first(ana)));
        f.push(format!("antFirst={}", first(ant)));
        f.push(format!("anaShape={}", shape(ana)));
        f.push(format!("antShape={}", shape(ant)));
        f.push(format!("anaLength={}", bucket(ana.words.len() as i64)));
        f.push(format!("antLength={}", bucket(ant.words.len() as i64)));
        f.push(format!("anaType={}", ty(ana)));
        f.push(format!("antType={}", ty(ant)));
        f.push(format!("types={}|{}", ty(ana), ty(ant)));
        f.push(format!("gender={}|{}", ana.gender.java_name(), ant.gender.java_name()));
        f.push(format!("number={}|{}", ana.number.java_name(), ant.number.java_name()));
        f.push(format!("animacy={}|{}", ana.animacy.java_name(), ant.animacy.java_name()));
        f.push(format!("antPerson={}", ant.person.java_name()));
        f.push(format!("antFirstInCluster={ant_first_in_cluster}"));
        let sentences = ana.sentence as i64 - ant.sentence as i64;
        let distance = format!("dist={}|{}", bucket(sentences), bucket(j as i64 - i as i64));
        f.push(distance.clone());
        f.push(format!("{kinds}|{distance}"));
        f.push(format!("{kinds}|sameHead={same_head}"));
        if ana_pronoun {
            let form = format!("form={}", ana.head);
            f.push(form.clone());
            f.push(format!("{form}|{}", kind(ant)));
            f.push(format!("{form}|antGender={}", ant.gender.java_name()));
            f.push(format!("{form}|antNumber={}", ant.number.java_name()));
            f.push(format!("{form}|antAnimacy={}", ant.animacy.java_name()));
            f.push(format!("{form}|antType={}", ty(ant)));
            f.push(format!("{form}|{distance}"));
            f.push(format!("{form}|antHead={}", ant.head));
        } else {
            f.push(format!("anaFirst={}|{}", first(ana), kind(ant)));
            f.push(format!("anaIndefinite={}", ana.indefinite));
            f.push(format!("antIndefinite={}", ant.indefinite));
            f.push(format!("bothProper={}", ana.proper && ant.proper));
        }
        if self.token_vectors.is_none() {
            return Ok(Features { names: f, anaphor: None, antecedent: None });
        }
        self.context_features(i, j, &kinds, f)
    }
}
