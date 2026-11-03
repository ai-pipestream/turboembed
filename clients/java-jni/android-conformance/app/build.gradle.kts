plugins {
    id("com.android.application")
    id("de.mannodermaus.android-junit5")
}

android {
    namespace = "ai.pipestream.turboembed.jni.android"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.pipestream.turboembed.jni.android.conformance"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        testInstrumentationRunnerArguments["runnerBuilder"] = "de.mannodermaus.junit5.AndroidJUnit5Builder"
        // Pack both ABIs when present under jniLibs; the emulator job ships x86_64 only.
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
        isCoreLibraryDesugaringEnabled = true
    }

    buildTypes {
        getByName("debug") {
            isMinifyEnabled = false
        }
        getByName("release") {
            isMinifyEnabled = false
        }
    }

    sourceSets {
        getByName("main") {
            // Shared API contracts (#1814) plus the JNI implementation (includes JniRuntime).
            java.srcDir("../../../java-api/src/main/java")
            java.srcDir("../../src/main/java")
            // build-android.sh / run-android-conformance.sh drop libturbo.so here.
            jniLibs.srcDir("src/main/jniLibs")
            assets.srcDir("src/main/assets")
        }
        getByName("androidTest") {
            // Directory roots only (setSrcDirs does not accept individual .java files).
            // Maven wrappers (Jni*ConformanceTest) and host-only RuntimeConformance /
            // LoaderProbe are dropped on the AndroidTest JavaCompile tasks below, where
            // PatternFilterable.exclude is valid. Android* suites own Analyzer/Session on ART.
            java.setSrcDirs(
                listOf(
                    "../../../java-conformance/src/main/java",
                    "../../src/test/java",
                    "src/androidTest/java",
                ),
            )
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    lint {
        abortOnError = true
        warningsAsErrors = false
    }
}

// exclude() is valid on JavaCompile (PatternFilterable), not on AndroidSourceSet.
tasks.withType<JavaCompile>().configureEach {
    if (name.contains("AndroidTest", ignoreCase = true)) {
        exclude(
            "**/JniAnalyzerConformanceTest.java",
            "**/JniSessionConformanceTest.java",
            "**/JniRuntimeConformanceTest.java",
            "**/RuntimeConformance.java",
            "**/LoaderProbe.java",
        )
    }
}

dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.4")
    implementation(project(":helpers"))

    androidTestImplementation(project(":helpers"))
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("org.junit.jupiter:junit-jupiter-api:5.11.4")
    androidTestImplementation("de.mannodermaus.junit5:android-test-core:1.6.0")
    androidTestRuntimeOnly("de.mannodermaus.junit5:android-test-runner:1.6.0")
    androidTestRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.11.4")
}
