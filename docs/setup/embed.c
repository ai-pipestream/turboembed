/* Embeds two texts with a bundle on a device, from the C interface alone.
 *
 *   cc -std=c11 -I include docs/setup/embed.c -L target/release -lturbo \
 *      -Wl,-rpath,$PWD/target/release -o embed
 *   ./embed testdata/tiny-bert-bundle 0 model
 *
 * Arguments: the bundle directory, a runtime device index (as
 * `turbo_runtime_device_info` lists them; the CPU is the last), and a
 * tier: model, fastest or exact. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <turbo/turbo.h>

static turbo_text text(const char *s) { return (turbo_text){s, strlen(s)}; }

static int check(int32_t code, const turbo_error *err, const char *call) {
    if (code == TURBO_OK) return 0;
    fprintf(stderr, "%s: %s (field %u): %s\n", call, turbo_status_name(code), err->field, err->message);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <bundle-dir> <device> <model|fastest|exact>\n", argv[0]);
        return 2;
    }
    uint32_t device = (uint32_t)strtoul(argv[2], NULL, 10);
    uint32_t precision = !strcmp(argv[3], "fastest") ? TURBO_PRECISION_FASTEST
                       : !strcmp(argv[3], "exact")   ? TURBO_PRECISION_EXACT
                                                     : TURBO_PRECISION_MODEL;
    printf("library: %s\n", turbo_version());

    turbo_error err = {.struct_size = sizeof err};
    turbo_runtime *rt = NULL;
    turbo_runtime_desc rd = {.struct_size = sizeof rd};
    if (check(turbo_runtime_create(&rd, &rt, &err), &err, "turbo_runtime_create")) return 1;

    uint32_t count = 0;
    turbo_runtime_device_count(rt, &count, &err);
    for (uint32_t i = 0; i < count; i++) {
        turbo_device_info info = {.struct_size = sizeof info};
        if (turbo_runtime_device_info(rt, i, &info, &err) == TURBO_OK)
            printf("device %u: %s (%s)\n", i, info.name, info.arch);
    }

    turbo_context *ctx = NULL;
    turbo_model *model = NULL;
    turbo_session *session = NULL;
    turbo_result *result = NULL;
    float *vectors = NULL;
    int failed = 1;

    if (check(turbo_context_create(rt, device, &ctx, &err), &err, "turbo_context_create")) goto done;
    if (check(turbo_model_load(ctx, text(argv[1]), &model, &err), &err, "turbo_model_load")) goto done;

    /* 0 for max_batch and max_seq: the model's. */
    turbo_session_desc sd = {.struct_size = sizeof sd, .precision = precision, .tuning = TURBO_AUTOTUNE_OFF};
    if (check(turbo_session_create(model, &sd, &session, &err), &err, "turbo_session_create")) goto done;
    turbo_session_info si = {.struct_size = sizeof si};
    if (check(turbo_session_get_info(session, &si, &err), &err, "turbo_session_get_info")) goto done;
    printf("session: max_batch %u, max_seq %u, compute dtype %u\n", si.max_batch, si.max_seq, si.compute_dtype);

    turbo_text texts[2] = {text("The cat sat on the mat."), text("A feline rested on a rug.")};
    /* NULL options: what the bundle says (truncation, pooling, normalization). */
    if (check(turbo_embed_write_text(session, texts, 2, NULL, &err), &err, "turbo_embed_write_text")) goto done;
    if (check(turbo_session_run(session, &result, &err), &err, "turbo_session_run")) goto done;

    turbo_result_info ri = {.struct_size = sizeof ri};
    if (check(turbo_result_get_info(result, &ri, &err), &err, "turbo_result_get_info")) goto done;
    vectors = malloc(ri.bytes);
    uint64_t written = 0;
    if (check(turbo_result_read(result, vectors, ri.bytes, &written, &err), &err, "turbo_result_read")) goto done;

    double dot = 0;
    for (uint32_t j = 0; j < ri.dim; j++) dot += (double)vectors[j] * vectors[ri.dim + j];
    printf("%u vectors of %u values; first values %.6f %.6f; dot product %.6f\n", ri.batch, ri.dim, vectors[0],
           vectors[1], dot);
    printf("copies: %llu bytes to the device, %llu back\n", (unsigned long long)ri.h2d_bytes,
           (unsigned long long)ri.d2h_bytes);
    failed = 0;

done:
    free(vectors);
    turbo_result_release(result);
    turbo_session_release(session);
    turbo_model_release(model);
    turbo_context_release(ctx);
    turbo_runtime_release(rt);
    return failed;
}
