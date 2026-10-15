//! The Spanish time rules on texts of their own. The Recognizers-Text
//! cases are in `tests/spec/spanish.rs`.

use turbo_analysis_layers::DateTime;

use super::*;
use crate::engine::Parser;
use crate::time::resolve;

/// Each mention as `text|kind|timex|modifier`, with its preferred
/// reading, against the reference `at`.
fn found(text: &str, at: &str) -> Vec<String> {
    let now = DateTime::parse(at).unwrap();
    let mut p = Parser::new(&SPANISH_TIMES);
    p.parse(text);
    let mut ms = Vec::new();
    p.select(text, Output::Time, &mut ms);
    ms.iter()
        .map(|m| {
            let rule = SPANISH_TIMES.rules()[usize::from(p.chart().nodes[m.node as usize].rule)].name;
            let shown = &text[m.start as usize..m.end as usize];
            match p.times().get(&m.value).and_then(|v| resolve(&v, now)) {
                Some(r) => {
                    let a = r.alternatives[r.preferred];
                    format!("{shown}|{}|{}|{}", r.kind.name(), a.timex, r.modifier)
                }
                None => format!("{shown}|unresolved {rule}"),
            }
        })
        .collect()
}

const AT: &str = "2024-05-15T12:00:00";

fn one(text: &str) -> String {
    let f = found(text, AT);
    assert_eq!(f.len(), 1, "{text}: {f:?}");
    f[0].clone()
}

fn none(text: &str) {
    let f = found(text, AT);
    assert!(f.is_empty(), "{text}: {f:?}");
}

#[test]
fn clock_times() {
    assert_eq!(one("Me voy a las cinco"), "cinco|time|T05|");
    assert_eq!(one("a las doce y cuarto"), "doce y cuarto|time|T12:15|");
    assert_eq!(one("a las tres menos diez"), "tres menos diez|time|T02:50|");
    assert_eq!(one("El concierto empieza a las 21.30"), "21.30|time|T21:30|");
    assert_eq!(one("sale a las 9 de la noche"), "9 de la noche|time|T21|");
    // An hour needs `las` before it; `00` names none.
    none("Tengo dos hermanos y tres primas");
    none("Hay 00 resultados");
    // `5a` is no clock time in Spanish: `1ª` is an ordinal.
    none("Quedó en 1ª posición");
}

#[test]
fn dates() {
    assert_eq!(one("El 2 de febrero de 1990 nació"), "2 de febrero de 1990|date|1990-02-02|");
    assert_eq!(one("Vivo en la calle 5 de mayo"), "5 de mayo|date|XXXX-05-05|");
    assert_eq!(one("Voy a comprar pan mañana"), "mañana|date|2024-05-16|");
    assert_eq!(one("vence el 3/4/2025"), "3/4/2025|date|2025-04-03|");
    assert_eq!(one("hasta el sáb. temprano"), "hasta el sáb.|daterange|XXXX-WXX-6|before");
    // `la mañana` is a morning; `el mar` is the sea.
    none("La mañana fue fría");
    none("El mar estaba tranquilo");
}

#[test]
fn relative_weekdays() {
    // 2024-05-15 is a Wednesday.
    assert_eq!(one("el lunes que viene"), "lunes que viene|date|2024-05-20|");
    assert_eq!(one("el viernes pasado"), "viernes pasado|date|2024-05-10|");
    assert_eq!(one("este pasado lunes"), "este pasado lunes|date|2024-05-13|");
    assert_eq!(one("el jueves de la próxima semana"), "jueves de la próxima semana|date|2024-05-23|");
}

#[test]
fn dates_with_times() {
    assert_eq!(
        one("Nos vemos el próximo martes a las 3 de la tarde"),
        "próximo martes a las 3 de la tarde|datetime|2024-05-21T15|"
    );
    assert_eq!(
        one("El partido es el 15 de agosto de 2024 a las 20:00"),
        "15 de agosto de 2024 a las 20:00|datetime|2024-08-15T20:00|"
    );
    assert_eq!(one("Llegamos pasado mañana por la mañana"), "pasado mañana por la mañana|datetimerange|2024-05-17TMO|");
    assert_eq!(one("Esta noche cenamos pescado"), "Esta noche|datetimerange|2024-05-15TNI|");
}

#[test]
fn periods_and_modifiers() {
    assert_eq!(one("Desde el año pasado no viene"), "Desde el año pasado|daterange|2023|since");
    assert_eq!(one("Vuelvo a mediados de marzo"), "mediados de marzo|daterange|XXXX-03|mid");
    assert_eq!(one("Hay que entregarlo antes del viernes"), "antes del viernes|daterange|XXXX-WXX-5|before");
    assert_eq!(one("el fin de semana que viene"), "el fin de semana que viene|daterange|2024-W21-WE|");
    assert_eq!(one("En 1998 nació mi hermana"), "1998|daterange|1998|");
    // The year of `el año anterior a la boda` is the reference's own.
    assert_eq!(one("el año anterior a la boda"), "año|daterange|2024|");
}

#[test]
fn ranges() {
    assert_eq!(one("Trabajo de lunes a viernes"), "de lunes a viernes|daterange|(2024-05-13,2024-05-17,P4D)|");
    assert_eq!(one("de lunes a mar"), "de lunes a mar|daterange|(2024-05-20,2024-05-21,P1D)|");
    assert_eq!(one("Abrimos de 8 a 14"), "de 8 a 14|timerange|(T08,T14,PT6H)|");
    assert_eq!(
        one("Termino dentro de una hora"),
        "dentro de una hora|datetimerange|(2024-05-15T12:00:00,2024-05-15T13:00:00,PT1H)|"
    );
    // `martes 4 a las siete` is a day and a time, not a range from 4.
    assert_eq!(one("el martes 4 a las siete"), "martes 4 a las siete|datetime|XXXX-WXX-2T07|");
    // A day does not run to a bare year.
    assert_eq!(one("vendió el 31-dic-1994"), "31-dic-1994|date|1994-12-31|");
}

#[test]
fn durations_shifts_and_sets() {
    assert_eq!(one("Estudió durante tres horas y media"), "tres horas y media|duration|PT3.5H|");
    assert_eq!(one("Nos vimos hace un año"), "hace un año|date|2023-05-15|");
    assert_eq!(one("Hace dos semanas que no la veo"), "Hace dos semanas|date|2024-05-01|");
    assert_eq!(one("Riego las plantas cada dos semanas"), "cada dos semanas|set|P2W|");
    assert_eq!(one("Abrimos todos los días"), "todos los días|set|P1D|");
    // `diario` is also a newspaper.
    none("Lo leí en el diario");
}
