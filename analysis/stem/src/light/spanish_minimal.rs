//! `SpanishMinimalStemmer`: plural endings only, from Lucene's
//! `SpanishPluralStemmer`, with its lists of words that keep their -s and
//! of irregular plurals.

use super::util::{ch, unit};

/// Words ending in -s that are not plurals, after accents are removed.
const INVARIANTS: [&[u16]; 108] = [
    u16s!("abrebotellas"),
    u16s!("abrecartas"),
    u16s!("abrelatas"),
    u16s!("afueras"),
    u16s!("albatros"),
    u16s!("albricias"),
    u16s!("aledaños"),
    u16s!("alexis"),
    u16s!("alicates"),
    u16s!("analisis"),
    u16s!("andurriales"),
    u16s!("antitesis"),
    u16s!("añicos"),
    u16s!("apendicitis"),
    u16s!("apocalipsis"),
    u16s!("arcoiris"),
    u16s!("aries"),
    u16s!("bilis"),
    u16s!("boletus"),
    u16s!("boris"),
    u16s!("brindis"),
    u16s!("cactus"),
    u16s!("canutas"),
    u16s!("caries"),
    u16s!("cascanueces"),
    u16s!("cascarrabias"),
    u16s!("ciempies"),
    u16s!("cifosis"),
    u16s!("cortaplumas"),
    u16s!("corpus"),
    u16s!("cosmos"),
    u16s!("cosquillas"),
    u16s!("creces"),
    u16s!("crisis"),
    u16s!("cuatrocientas"),
    u16s!("cuatrocientos"),
    u16s!("cuelgacapas"),
    u16s!("cuentacuentos"),
    u16s!("cuentapasos"),
    u16s!("cumpleaños"),
    u16s!("doscientas"),
    u16s!("doscientos"),
    u16s!("dosis"),
    u16s!("enseres"),
    u16s!("entonces"),
    u16s!("esponsales"),
    u16s!("estatus"),
    u16s!("exequias"),
    u16s!("fauces"),
    u16s!("forceps"),
    u16s!("fotosintesis"),
    u16s!("gafas"),
    u16s!("gafotas"),
    u16s!("gargaras"),
    u16s!("gris"),
    u16s!("honorarios"),
    u16s!("ictus"),
    u16s!("jueves"),
    u16s!("lapsus"),
    u16s!("lavacoches"),
    u16s!("lavaplatos"),
    u16s!("limpiabotas"),
    u16s!("lunes"),
    u16s!("maitines"),
    u16s!("martes"),
    u16s!("mondadientes"),
    u16s!("novecientas"),
    u16s!("novecientos"),
    u16s!("nupcias"),
    u16s!("ochocientas"),
    u16s!("ochocientos"),
    u16s!("pais"),
    u16s!("paris"),
    u16s!("parabrisas"),
    u16s!("paracaidas"),
    u16s!("parachoques"),
    u16s!("paraguas"),
    u16s!("pararrayos"),
    u16s!("pisapapeles"),
    u16s!("piscis"),
    u16s!("portaaviones"),
    u16s!("portamaletas"),
    u16s!("portamantas"),
    u16s!("quinientas"),
    u16s!("quinientos"),
    u16s!("quitamanchas"),
    u16s!("recogepelotas"),
    u16s!("rictus"),
    u16s!("rompeolas"),
    u16s!("sacacorchos"),
    u16s!("sacapuntas"),
    u16s!("saltamontes"),
    u16s!("salvavidas"),
    u16s!("seis"),
    u16s!("seiscientas"),
    u16s!("seiscientos"),
    u16s!("setecientas"),
    u16s!("setecientos"),
    u16s!("sintesis"),
    u16s!("tenis"),
    u16s!("tifus"),
    u16s!("trabalenguas"),
    u16s!("vacaciones"),
    u16s!("venus"),
    u16s!("versus"),
    u16s!("viacrucis"),
    u16s!("virus"),
    u16s!("viveres"),
];

/// Plurals that lose -es where the rules below would remove less.
const SPECIAL_CASES: [&[u16]; 13] = [
    u16s!("albalaes"),
    u16s!("albumes"),
    u16s!("bojes"),
    u16s!("carcajes"),
    u16s!("clubes"),
    u16s!("contrarrelojes"),
    u16s!("faralaes"),
    u16s!("itemes"),
    u16s!("noes"),
    u16s!("relojes"),
    u16s!("sandwiches"),
    u16s!("sies"),
    u16s!("yoes"),
];

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

pub(crate) fn stem(s: &mut [u16], len: usize) -> usize {
    if len < 4 {
        return len;
    }
    for c in &mut s[..len] {
        *c = match ch(*c) {
            'à' | 'á' | 'â' | 'ä' => unit('a'),
            'è' | 'é' | 'ê' | 'ë' => unit('e'),
            'ì' | 'í' | 'î' | 'ï' => unit('i'),
            'ò' | 'ó' | 'ô' | 'ö' => unit('o'),
            'ù' | 'ú' | 'û' | 'ü' => unit('u'),
            _ => *c,
        };
    }
    if ch(s[len - 1]) != 's' {
        return len;
    }
    let word = &s[..len];
    if INVARIANTS.contains(&word) {
        return len;
    }
    if SPECIAL_CASES.contains(&word) {
        return len - 2;
    }
    let (a, b, c) = (ch(s[len - 2]), ch(s[len - 3]), ch(s[len - 4]));
    if !is_vowel(a) {
        return len - 1;
    }
    if c == 'q' || (c == 'g' && b == 'u' && (a == 'i' || a == 'e')) {
        return len - 1;
    }
    if is_vowel(c) && b == 'r' && a == 'e' {
        return len - 2;
    }
    if is_vowel(c) && matches!(b, 'd' | 'l' | 'n' | 'x') && a == 'e' {
        return len - 2;
    }
    if (b == 'y' || b == 'u') && a == 'e' {
        return len - 2;
    }
    if matches!(c, 'u' | 'l' | 'r' | 't' | 'n') && b == 'i' && a == 'e' {
        return len - 2;
    }
    if b == 's' && a == 'e' {
        return len - 2;
    }
    if is_vowel(b) && a == 'i' {
        s[len - 2] = unit('y');
        return len - 1;
    }
    if b == 'd' && a == 'i' {
        s[len - 2] = unit('y');
        return len - 1;
    }
    if b == 'c' && a == 'e' {
        s[len - 3] = unit('z');
        return len - 2;
    }
    if is_vowel(a) {
        return len - 1;
    }
    len
}
