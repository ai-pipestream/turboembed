#include <stdio.h>
#include <stdlib.h>
#include "iga/iga.h"
int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"); fseek(f, 0, 2); long n = ftell(f); rewind(f);
  unsigned char *b = malloc(n); if (fread(b, 1, n, f) != (size_t)n) return 1;
  long off = argc > 2 ? strtol(argv[2], 0, 0) : 0, len = argc > 3 ? strtol(argv[3], 0, 0) : n - off;
  iga_context_options_t co = IGA_CONTEXT_OPTIONS_INIT(IGA_XE2);
  iga_context_t ctx; if (iga_context_create(&co, &ctx)) { printf("ctx fail\n"); return 1; }
  iga_disassemble_options_t o = IGA_DISASSEMBLE_OPTIONS_INIT();
  char *text = 0;
  iga_status_t s = iga_context_disassemble(ctx, &o, b + off, (size_t)len, 0, 0, &text);
  if (text) fputs(text, stdout); else printf("status %d\n", s);
  return 0;
}
