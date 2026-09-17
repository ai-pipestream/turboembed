/*
 * External consumer example for the packaged turboembed.h C ABI.
 *
 * Usage: turboembed_embed <catalog.toml> <alias> [auto|cuda|cpu|tensorrt]
 *
 * Creates an engine on the requested device (default cuda), loads the
 * catalog alias, embeds one short and one Unicode text as a batch, and
 * verifies the reported dimension and L2 norms. Device policy is the
 * library's: a missing accelerator is a loud error, never a CPU fallback.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <turboembed.h>

static int fail(const char *what, const turboembed_engine *engine) {
    fprintf(stderr, "%s failed: %s\n", what, turboembed_last_error(engine));
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <catalog.toml> <alias> [auto|cuda|cpu|tensorrt]\n", argv[0]);
        return 2;
    }
    const char *catalog = argv[1];
    const char *alias = argv[2];
    const char *device_name = argc == 4 ? argv[3] : "cuda";

    turboembed_device device;
    if (strcmp(device_name, "auto") == 0) {
        device = TURBOEMBED_DEVICE_AUTO;
    } else if (strcmp(device_name, "cuda") == 0) {
        device = TURBOEMBED_DEVICE_CUDA;
    } else if (strcmp(device_name, "cpu") == 0) {
        device = TURBOEMBED_DEVICE_CPU;
    } else if (strcmp(device_name, "tensorrt") == 0) {
        device = TURBOEMBED_DEVICE_TENSORRT;
    } else {
        fprintf(stderr, "unknown device %s\n", device_name);
        return 2;
    }

    printf("abi_version=%u\n", turboembed_abi_version());

    turboembed_engine *engine = NULL;
    if (turboembed_engine_create(device, catalog, &engine) != TURBOEMBED_OK) {
        return fail("engine create", NULL);
    }
    if (turboembed_load_model(engine, alias, strlen(alias)) != TURBOEMBED_OK) {
        int rc = fail("load_model", engine);
        turboembed_engine_destroy(engine);
        return rc;
    }

    turboembed_model_info *infos = NULL;
    size_t n_infos = 0;
    if (turboembed_list_models(engine, &infos, &n_infos) != TURBOEMBED_OK) {
        int rc = fail("list_models", engine);
        turboembed_engine_destroy(engine);
        return rc;
    }
    for (size_t i = 0; i < n_infos; ++i) {
        printf("model=%.*s device=%s dim=%u\n",
               (int)infos[i].alias.len, infos[i].alias.ptr,
               turboembed_device_name(infos[i].device), infos[i].dim);
    }
    turboembed_model_list_free(infos, n_infos);

    const turboembed_str texts[2] = {
        {"hello world", strlen("hello world")},
        {"das Straßenpflaster glänzt — 東京", strlen("das Straßenpflaster glänzt — 東京")},
    };
    turboembed_embed_options opts;
    memset(&opts, 0, sizeof(opts));
    opts.pooling = TURBOEMBED_POOLING_DEFAULT;
    opts.normalize = -1; /* catalog default */

    turboembed_embed_result *result = NULL;
    if (turboembed_embed(engine, alias, strlen(alias), texts, 2, &opts, &result) !=
        TURBOEMBED_OK) {
        int rc = fail("embed", engine);
        turboembed_engine_destroy(engine);
        return rc;
    }
    if (result->dim == 0 || result->count != 2 || result->values == NULL) {
        fprintf(stderr, "embed returned an empty result\n");
        turboembed_embed_result_free(result);
        turboembed_engine_destroy(engine);
        return 1;
    }
    int norms_ok = 1;
    for (unsigned row = 0; row < result->count; ++row) {
        double ss = 0.0;
        for (unsigned d = 0; d < result->dim; ++d) {
            const double v = result->values[(size_t)row * result->dim + d];
            ss += v * v;
        }
        const double norm = sqrt(ss);
        printf("row=%u dim=%u norm=%.6f\n", row, result->dim, norm);
        if (fabs(norm - 1.0) > 1e-3) {
            norms_ok = 0;
        }
    }
    turboembed_embed_result_free(result);
    turboembed_engine_destroy(engine);

    if (!norms_ok) {
        fprintf(stderr, "L2 norms are off; catalog normalize=true expected\n");
        return 1;
    }
    printf("embed=PASS device=%s\n", device_name);
    return 0;
}
