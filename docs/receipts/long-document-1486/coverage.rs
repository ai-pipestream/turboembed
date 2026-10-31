use std::{collections::BTreeSet,path::Path};
fn main() -> Result<(),String> {
 let cache=std::env::args().nth(1).ok_or("coverage <cache>")?;
 let mut rows=Vec::new();
 for name in ["qasper","gutenqa","cuad","contract-nli"] {
  let s=turbo_eval::sets::find(name)?.load(Path::new(&cache))?;
  let groups=turbo_eval::grouped::query_groups(&s)?;
  let doc_queries:BTreeSet<_>=s.qrels.scored_queries().collect();
  let mut span_queries=BTreeSet::new();let mut spans=0;let mut pairs=0;
  for q in &s.queries {for d in &s.docs {
   pairs+=usize::from(s.qrels.grade(&q.id,&d.id)>0);
   let n=s.spans.as_ref().map_or(0,|r|r.spans(&q.id,&d.id).len());
   spans+=n;if n>0 {span_queries.insert(q.id.as_str());}
  }}
  rows.push(serde_json::json!({"set":name,"sha256":s.sha256,"documents":s.docs.len(),
   "loaded_queries":s.queries.len(),"scored_document_queries":doc_queries.len(),
   "scored_evidence_queries":span_queries.len(),"placed_span_judgements":spans,
   "positive_document_query_pairs":pairs,"unplaced_judgements":s.unplaced.len(),
   "absent_document_judgements":s.absent.len(),
   "queries_without_positive_document_judgement":s.queries.len()-doc_queries.len(),
   "queries_without_placed_evidence":s.queries.len()-span_queries.len(),
   "document_score_groups":doc_queries.iter().map(|q| &groups[*q]).collect::<BTreeSet<_>>().len(),
   "evidence_score_groups":span_queries.iter().map(|q| &groups[*q]).collect::<BTreeSet<_>>().len()}));
 }
 println!("{}",serde_json::to_string_pretty(&serde_json::json!({"library_commit":turbo_eval::source::BUILD_COMMIT,"sets":rows})).unwrap());
 Ok(())
}
