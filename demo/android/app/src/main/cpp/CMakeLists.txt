# The JNI shim for the Android app. libturbo.so for the ABI is expected at
# app/src/main/jniLibs/<abi>/libturbo.so (a cross-compiled
# `cargo build -p turbo-shared --target aarch64-linux-android`); the build
# fails with that path named when it is missing.
cmake_minimum_required(VERSION 3.22)
project(turbo_jni LANGUAGES C)
get_filename_component(TURBO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
set(TURBO_SO "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libturbo.so")
if(NOT EXISTS "${TURBO_SO}")
    message(FATAL_ERROR "missing ${TURBO_SO}: cross-compile libturbo for ${ANDROID_ABI} first (see demo/android/README.md)")
endif()
add_library(turbo_jni SHARED "${TURBO_ROOT}/demo/android/jni/turbo_jni.c")
target_include_directories(turbo_jni PRIVATE "${TURBO_ROOT}/include")
target_compile_options(turbo_jni PRIVATE -Wall -Wextra)
target_link_libraries(turbo_jni PRIVATE "${TURBO_SO}")
