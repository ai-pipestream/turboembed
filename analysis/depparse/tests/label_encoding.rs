use turbo_analysis::Ix;
use turbo_analysis_layers::Tokens;
use turbo_depparse::*;

#[test]
fn label_contents_do_not_select_decoding_and_escaped_roundtrips_are_exact() {
    for label in ["root^literal", "root^>literal", r"root\^literal", r"root\\^>literal", "r<oot>", "根^>𝄞"] {
        for heads in [vec![None], vec![Some(Ix::new(2)), None, Some(Ix::new(1))]] {
            let labels =
                if heads.len() == 1 { vec![label.into()] } else { vec!["det".into(), label.into(), "obj".into()] };
            let graph = DependencyGraph::<Tokens>::new(heads, labels).unwrap();
            let plain = graph.deprojectivize().unwrap();
            let decoded = graph.projectivize().unwrap().deprojectivize().unwrap();
            assert_eq!(decoded.to_layer().head, graph.to_layer().head);
            for i in 0..graph.len() {
                assert_eq!(
                    plain.arc(Ix::new(i as u32)).unwrap().relation,
                    graph.arc(Ix::new(i as u32)).unwrap().relation
                );
                assert_eq!(
                    decoded.arc(Ix::new(i as u32)).unwrap().relation,
                    graph.arc(Ix::new(i as u32)).unwrap().relation
                );
            }
        }
    }
}
