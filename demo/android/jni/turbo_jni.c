/* SPDX-License-Identifier: Apache-2.0
 *
 * JNI shim over the Turbo C ABI for Android (and any JVM without FFM). It
 * exposes the few calls the demo app needs: open a bundle on the best
 * device, describe it, embed a batch of strings, and close. Handles are
 * passed to Java as longs; every libturbo failure is thrown as
 * ai.pipestream.turbo.android.TurboException carrying the status name, the
 * field index, and the library's message. Nothing is swallowed.
 *
 * Built by the app's CMakeLists.txt with the NDK, and by hosttest/run.sh
 * with the host compiler against the JDK's JNI headers for a device-free
 * check of the same source.
 */
#include "turbo/turbo.h"

#include <jni.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    turbo_runtime *rt;
    turbo_context *ctx;
    turbo_model *model;
    turbo_session *session;
    turbo_model_info info;
    turbo_device_info device;
    uint32_t max_batch;
} engine;

static void throw_oom(JNIEnv *env, const char *what) {
    jclass oom = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
    if (oom != NULL) {
        (*env)->ThrowNew(env, oom, what);
    }
}

static void throw_turbo(JNIEnv *env, const char *what, int32_t rc, const turbo_error *err) {
    jclass cls = (*env)->FindClass(env, "ai/pipestream/turbo/android/TurboException");
    if (cls == NULL) {
        return; /* a pending NoClassDefFoundError is already set */
    }
    jmethodID ctor = (*env)->GetMethodID(env, cls, "<init>", "(Ljava/lang/String;ILjava/lang/String;ILjava/lang/String;)V");
    if (ctor == NULL) {
        return;
    }
    jstring jwhat = (*env)->NewStringUTF(env, what);
    jstring jname = (*env)->NewStringUTF(env, turbo_status_name(rc));
    jstring jmsg = (*env)->NewStringUTF(env, err->message);
    jobject ex = (*env)->NewObject(env, cls, ctor, jwhat, (jint)rc, jname, (jint)err->field, jmsg);
    if (ex != NULL) {
        (*env)->Throw(env, (jthrowable)ex);
    }
}

#define CALL(what, expr)                                                                           \
    do {                                                                                           \
        turbo_error err_;                                                                          \
        memset(&err_, 0, sizeof err_);                                                             \
        err_.struct_size = (uint32_t)sizeof err_;                                                  \
        int32_t rc_ = (expr);                                                                      \
        if (rc_ != TURBO_OK) {                                                                     \
            throw_turbo(env, what, rc_, &err_);                                                    \
            goto fail;                                                                             \
        }                                                                                          \
    } while (0)

/* GetStringUTFChars hands back JNI modified UTF-8 (a supplementary
 * character as a CESU-8 surrogate pair, U+0000 as 0xC0 0x80), which is not
 * the UTF-8 turbo_text is specified as and the library validates. The Java
 * side encodes with StandardCharsets.UTF_8 and passes byte[]; these pin
 * those bytes for the duration of one call. turbo_text carries its length,
 * so no NUL terminator is needed. */
typedef struct {
    jbyteArray array;
    jbyte *bytes;
} pinned;

static int pin_utf8(JNIEnv *env, jbyteArray array, pinned *p, turbo_text *out) {
    p->array = array;
    p->bytes = (*env)->GetByteArrayElements(env, array, NULL);
    if (p->bytes == NULL) {
        return 0; /* an OutOfMemoryError is already pending */
    }
    out->ptr = (const char *)p->bytes;
    out->len = (uint64_t)(*env)->GetArrayLength(env, array);
    return 1;
}

static void unpin_utf8(JNIEnv *env, pinned *p) {
    if (p->bytes != NULL) {
        (*env)->ReleaseByteArrayElements(env, p->array, p->bytes, JNI_ABORT);
        p->bytes = NULL;
    }
}

static void engine_close(engine *e) {
    if (e == NULL) {
        return;
    }
    if (e->session) turbo_session_release(e->session);
    if (e->model) turbo_model_release(e->model);
    if (e->ctx) turbo_context_release(e->ctx);
    if (e->rt) turbo_runtime_release(e->rt);
    free(e);
}

/* long open(byte[] bundle, byte[] providerLib, int maxBatch): UTF-8 bytes;
 * providerLib may be null. */
JNIEXPORT jlong JNICALL Java_ai_pipestream_turbo_android_TurboJni_open(JNIEnv *env, jclass cls, jbyteArray jbundle,
                                                                       jbyteArray jprovider_lib, jint max_batch) {
    (void)cls;
    engine *e = calloc(1, sizeof *e);
    pinned bundle = {NULL, NULL}, provider_lib = {NULL, NULL};
    turbo_text bundle_text = {NULL, 0}, lib_text = {NULL, 0};
    if (e == NULL) {
        throw_oom(env, "engine allocation failed");
        return 0;
    }
    if (!pin_utf8(env, jbundle, &bundle, &bundle_text)) goto fail;
    if (jprovider_lib != NULL && !pin_utf8(env, jprovider_lib, &provider_lib, &lib_text)) goto fail;
    {
        turbo_runtime_desc rd;
        memset(&rd, 0, sizeof rd);
        rd.struct_size = (uint32_t)sizeof rd;
        if (provider_lib.bytes != NULL) {
            rd.n_provider_paths = 1;
            rd.provider_paths = &lib_text;
        }
        CALL("turbo_runtime_create", turbo_runtime_create(&rd, &e->rt, &err_));
        uint32_t dev = 0;
        CALL("turbo_runtime_select_device", turbo_runtime_select_device(e->rt, NULL, &dev, &err_));
        memset(&e->device, 0, sizeof e->device);
        e->device.struct_size = (uint32_t)sizeof e->device;
        CALL("turbo_runtime_device_info", turbo_runtime_device_info(e->rt, dev, &e->device, &err_));
        CALL("turbo_context_create", turbo_context_create(e->rt, dev, NULL, &e->ctx, &err_));
        CALL("turbo_model_load", turbo_model_load(e->ctx, bundle_text, NULL, &e->model, &err_));
        memset(&e->info, 0, sizeof e->info);
        e->info.struct_size = (uint32_t)sizeof e->info;
        CALL("turbo_model_get_info", turbo_model_get_info(e->model, &e->info, &err_));
        if (e->info.kind != TURBO_MODEL_EMBEDDING) {
            jclass iae = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            if (iae) (*env)->ThrowNew(env, iae, "the bundle is not an embedding model");
            goto fail;
        }
        turbo_session_desc sd;
        memset(&sd, 0, sizeof sd);
        sd.struct_size = (uint32_t)sizeof sd;
        sd.max_batch = (uint32_t)(max_batch > 0 ? max_batch : 1);
        sd.max_seq = 0;
        e->max_batch = sd.max_batch;
        CALL("turbo_session_create", turbo_session_create(e->model, &sd, &e->session, &err_));
    }
    unpin_utf8(env, &bundle);
    unpin_utf8(env, &provider_lib);
    return (jlong)(intptr_t)e;
fail:
    unpin_utf8(env, &bundle);
    unpin_utf8(env, &provider_lib);
    engine_close(e);
    return 0;
}

/* String describe(long handle): one line about the device and model. */
JNIEXPORT jstring JNICALL Java_ai_pipestream_turbo_android_TurboJni_describe(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;
    engine *e = (engine *)(intptr_t)handle;
    char buf[512];
    snprintf(buf, sizeof buf, "%s (%s:%u) | %s dim=%u max_seq=%u fully_accelerated=%u", e->device.name,
             e->device.provider_id, e->device.ordinal, e->info.model_id, e->info.dim, e->info.max_seq,
             e->info.fully_accelerated);
    return (*env)->NewStringUTF(env, buf);
}

JNIEXPORT jint JNICALL Java_ai_pipestream_turbo_android_TurboJni_dim(JNIEnv *env, jclass cls, jlong handle) {
    (void)env;
    (void)cls;
    return (jint)((engine *)(intptr_t)handle)->info.dim;
}

/* float[] embed(long handle, byte[][] texts): UTF-8 rows in, rows of dim
 * floats out, in order. */
JNIEXPORT jfloatArray JNICALL Java_ai_pipestream_turbo_android_TurboJni_embed(JNIEnv *env, jclass cls, jlong handle,
                                                                             jobjectArray jtexts) {
    (void)cls;
    engine *e = (engine *)(intptr_t)handle;
    jsize n = (*env)->GetArrayLength(env, jtexts);
    pinned *rows = NULL;
    turbo_text *views = NULL;
    turbo_result *result = NULL;
    jfloatArray out = NULL;
    float *buf = NULL;
    if (n <= 0 || (uint32_t)n > e->max_batch) {
        jclass iae = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
        char msg[128];
        snprintf(msg, sizeof msg, "batch of %d texts; the engine holds 1..%u", (int)n, e->max_batch);
        if (iae) (*env)->ThrowNew(env, iae, msg);
        return NULL;
    }
    rows = calloc((size_t)n, sizeof *rows);
    views = calloc((size_t)n, sizeof *views);
    if (!rows || !views) {
        throw_oom(env, "could not allocate the per-row views for this batch");
        goto fail;
    }
    for (jsize i = 0; i < n; ++i) {
        jbyteArray row = (jbyteArray)(*env)->GetObjectArrayElement(env, jtexts, i);
        if (row == NULL) {
            jclass npe = (*env)->FindClass(env, "java/lang/NullPointerException");
            char msg[64];
            snprintf(msg, sizeof msg, "texts[%d] is null", (int)i);
            if (npe) (*env)->ThrowNew(env, npe, msg);
            goto fail;
        }
        if (!pin_utf8(env, row, &rows[i], &views[i])) goto fail;
    }
    {
        CALL("turbo_session_write_text", turbo_session_write_text(e->session, views, (uint32_t)n, NULL, &err_));
        CALL("turbo_session_run", turbo_session_run(e->session, NULL, &result, &err_));
        turbo_result_info ri;
        uint64_t expected, written = 0;
        memset(&ri, 0, sizeof ri);
        ri.struct_size = (uint32_t)sizeof ri;
        CALL("turbo_result_get_info", turbo_result_get_info(result, &ri, &err_));
        /* This shim hands Java a float[]; any other dtype would be
         * reinterpreted f32 pairs, which the caller cannot detect. */
        if (ri.dtype != TURBO_DTYPE_F32) {
            jclass ise = (*env)->FindClass(env, "java/lang/IllegalStateException");
            char msg[160];
            snprintf(msg, sizeof msg, "the bundle's output dtype is %u, not TURBO_DTYPE_F32 (%u); this shim returns f32 rows only",
                     ri.dtype, (unsigned)TURBO_DTYPE_F32);
            if (ise) (*env)->ThrowNew(env, ise, msg);
            goto fail;
        }
        expected = (uint64_t)ri.batch * ri.dim * sizeof(float);
        if (ri.bytes != expected) {
            jclass ise = (*env)->FindClass(env, "java/lang/IllegalStateException");
            char msg[160];
            snprintf(msg, sizeof msg, "the result claims %llu bytes for %u x %u f32 (%llu expected)",
                     (unsigned long long)ri.bytes, ri.batch, ri.dim, (unsigned long long)expected);
            if (ise) (*env)->ThrowNew(env, ise, msg);
            goto fail;
        }
        buf = malloc((size_t)ri.bytes);
        if (buf == NULL) {
            throw_oom(env, "could not allocate the result buffer");
            goto fail;
        }
        CALL("turbo_result_read", turbo_result_read(result, 0, buf, ri.bytes, &written, &err_));
        if (written != ri.bytes) {
            jclass ise = (*env)->FindClass(env, "java/lang/IllegalStateException");
            char msg[128];
            snprintf(msg, sizeof msg, "turbo_result_read wrote %llu of %llu bytes", (unsigned long long)written,
                     (unsigned long long)ri.bytes);
            if (ise) (*env)->ThrowNew(env, ise, msg);
            goto fail;
        }
        out = (*env)->NewFloatArray(env, (jsize)(written / sizeof(float)));
        if (out == NULL) {
            throw_oom(env, "could not allocate the float[] for the embeddings");
            goto fail;
        }
        (*env)->SetFloatArrayRegion(env, out, 0, (jsize)(written / sizeof(float)), buf);
    }
fail:
    free(buf);
    if (result) turbo_result_release(result);
    if (rows) {
        for (jsize i = 0; i < n; ++i) {
            unpin_utf8(env, &rows[i]);
        }
    }
    free(views);
    free(rows);
    return out;
}

JNIEXPORT void JNICALL Java_ai_pipestream_turbo_android_TurboJni_close(JNIEnv *env, jclass cls, jlong handle) {
    (void)env;
    (void)cls;
    engine_close((engine *)(intptr_t)handle);
}
