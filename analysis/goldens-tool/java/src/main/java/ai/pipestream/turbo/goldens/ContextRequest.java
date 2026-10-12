package ai.pipestream.turbo.goldens;

/**
 * One request an annotator made to its context generator: the text it passed and the UTF-16
 * index into that text it asked about. The text is the very object the annotator passed, so a
 * caller can tell by identity which text the request was about.
 */
record ContextRequest(CharSequence text, int index) {}
