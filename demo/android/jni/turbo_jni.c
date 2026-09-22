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

static turbo_text text_of(const char *s) {
    turbo_text t;
    t.ptr = s;
    t.len = (uint64_t)strlen(s);
    return t;
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

/* long open(String bundle, String providerLib, int maxBatch): providerLib may be null. */
JNIEXPORT jlong JNICALL Java_ai_pipestream_turbo_android_TurboJni_open(JNIEnv *env, jclass cls, jstring jbundle,
                                                                       jstring jprovider_lib, jint max_batch) {
    (void)cls;
    engine *e = calloc(1, sizeof *e);
    const char *bundle = NULL, *provider_lib = NULL;
    if (e == NULL) {
        jclass oom = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
        if (oom) (*env)->ThrowNew(env, oom, "engine allocation failed");
        return 0;
    }
    bundle = (*env)->GetStringUTFChars(env, jbundle, NULL);
    if (bundle == NULL) goto fail;
    if (jprovider_lib != NULL) {
        provider_lib = (*env)->GetStringUTFChars(env, jprovider_lib, NULL);
        if (provider_lib == NULL) goto fail;
    }
    {
        turbo_runtime_desc rd;
        turbo_text lib_text;
        memset(&rd, 0, sizeof rd);
        rd.struct_size = (uint32_t)sizeof rd;
        if (provider_lib != NULL) {
            lib_text = text_of(provider_lib);
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
        CALL("turbo_model_load", turbo_model_load(e->ctx, text_of(bundle), NULL, &e->model, &err_));
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
    (*env)->ReleaseStringUTFChars(env, jbundle, bundle);
    if (provider_lib) (*env)->ReleaseStringUTFChars(env, jprovider_lib, provider_lib);
    return (jlong)(intptr_t)e;
fail:
    if (bundle) (*env)->ReleaseStringUTFChars(env, jbundle, bundle);
    if (provider_lib) (*env)->ReleaseStringUTFChars(env, jprovider_lib, provider_lib);
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

/* float[] embed(long handle, String[] texts): rows of dim floats, in order. */
JNIEXPORT jfloatArray JNICALL Java_ai_pipestream_turbo_android_TurboJni_embed(JNIEnv *env, jclass cls, jlong handle,
                                                                             jobjectArray jtexts) {
    (void)cls;
    engine *e = (engine *)(intptr_t)handle;
    jsize n = (*env)->GetArrayLength(env, jtexts);
    const char **utf = NULL;
    jstring *strs = NULL;
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
    utf = calloc((size_t)n, sizeof *utf);
    strs = calloc((size_t)n, sizeof *strs);
    views = calloc((size_t)n, sizeof *views);
    if (!utf || !strs || !views) goto fail;
    for (jsize i = 0; i < n; ++i) {
        strs[i] = (jstring)(*env)->GetObjectArrayElement(env, jtexts, i);
        if (strs[i] == NULL) goto fail;
        utf[i] = (*env)->GetStringUTFChars(env, strs[i], NULL);
        if (utf[i] == NULL) goto fail;
        views[i] = text_of(utf[i]);
    }
    {
        CALL("turbo_session_write_text", turbo_session_write_text(e->session, views, (uint32_t)n, NULL, &err_));
        CALL("turbo_session_run", turbo_session_run(e->session, NULL, &result, &err_));
        turbo_result_info ri;
        memset(&ri, 0, sizeof ri);
        ri.struct_size = (uint32_t)sizeof ri;
        CALL("turbo_result_get_info", turbo_result_get_info(result, &ri, &err_));
        buf = malloc((size_t)ri.bytes);
        if (buf == NULL) goto fail;
        uint64_t written = 0;
        CALL("turbo_result_read", turbo_result_read(result, 0, buf, ri.bytes, &written, &err_));
        out = (*env)->NewFloatArray(env, (jsize)(written / sizeof(float)));
        if (out == NULL) goto fail;
        (*env)->SetFloatArrayRegion(env, out, 0, (jsize)(written / sizeof(float)), buf);
    }
fail:
    free(buf);
    if (result) turbo_result_release(result);
    if (utf && strs) {
        for (jsize i = 0; i < n; ++i) {
            if (utf[i]) (*env)->ReleaseStringUTFChars(env, strs[i], utf[i]);
            if (strs[i]) (*env)->DeleteLocalRef(env, strs[i]);
        }
    }
    free(views);
    free(strs);
    free(utf);
    return out;
}

JNIEXPORT void JNICALL Java_ai_pipestream_turbo_android_TurboJni_close(JNIEnv *env, jclass cls, jlong handle) {
    (void)env;
    (void)cls;
    engine_close((engine *)(intptr_t)handle);
}
