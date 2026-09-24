# Turbo Android demo

An Android app over a JNI shim. Android has no FFM, so the app talks to the
C ABI through `jni/turbo_jni.c`, which exposes four calls (`open`,
`describe`, `embed`, `close`) and throws `TurboException` with the status
name, field, and message for every libturbo failure. `TurboEngine` wraps
them; `MainActivity` copies the bundle from the APK's assets to the app's
files directory once and embeds whatever is typed, one sentence per line.

## Check without a device

```sh
cargo build -p turbo-shared            # target/debug/libturbo.so
demo/android/hosttest/run.sh           # compiles the shim with the host compiler, drives TurboEngine on the mock bundle
```

The host test compiles the same `turbo_jni.c` against the JDK's JNI
headers, loads it with `System.loadLibrary` the way the app does, embeds
three sentences, and checks the shape, unit norm, the batch refusal, and
that a missing bundle is a `TurboException` naming
`TURBO_E_BUNDLE_NOT_FOUND`. It needs a JDK and `cc`; no SDK, NDK, or
emulator.

## Build the APK

Needs the Android SDK (compileSdk 36), NDK 27, and a `libturbo.so` for each
ABI in `app/src/main/jniLibs/<abi>/`:

```sh
rustup target add aarch64-linux-android x86_64-linux-android
# with the NDK's clang on PATH as the linker for each target (cargo-ndk does this):
cargo ndk -t arm64-v8a -t x86_64 -o demo/android/app/src/main/jniLibs build -p turbo-shared --release
cp -r testdata/bundles/mock/embedding demo/android/app/src/main/assets/bundle   # or a real bundle
cd demo/android && gradle assembleDebug
```

`app/src/main/cpp/CMakeLists.txt` fails with the missing path named when
`libturbo.so` for the ABI is not there, and `useLegacyPackaging = false`
keeps the libraries 16 KB page aligned for Android 15 and later. No APK
has been built or run in this tree yet: there is no Android SDK on the
machines it has receipts from, so the app is untested beyond the JNI layer.
