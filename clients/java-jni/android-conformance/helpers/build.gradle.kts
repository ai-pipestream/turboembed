plugins {
    id("java-library")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

tasks.withType<JavaCompile>().configureEach {
    // Emit Java 17 bytecode so the Android app module (compileOptions = VERSION_17) can consume :helpers.
    options.release.set(17)
}

tasks.test {
    useJUnitPlatform()
    include("**/*Test.class")
}

tasks.register<JavaExec>("verifyAndroidTestResults") {
    group = "verification"
    description = "Verifies connectedDebugAndroidTest XML reports have non-zero passing tests across all suites."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("ai.pipestream.turboembed.jni.android.AndroidTestReportVerifier")
    args(layout.projectDirectory.dir("../app/build/outputs/androidTest-results/connected").asFile.absolutePath)
}

dependencies {
    compileOnly("junit:junit:4.13.2")
    compileOnly("org.junit.jupiter:junit-jupiter-api:5.11.4")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.11.4")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.11.4")
}
