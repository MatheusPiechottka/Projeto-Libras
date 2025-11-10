plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.libras"

    // ✅ Sintaxe correta
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.libras"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // Ex.: se quiser campos de config:
            // buildConfigField("boolean", "LOGS", "false")
        }
        debug {
            // buildConfigField("boolean", "LOGS", "true")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
        // Caso use APIs Java 8+: ativar desugaring (opcional)
        // isCoreLibraryDesugaringEnabled = true
    }
    kotlinOptions {
        jvmTarget = "11"
        // freeCompilerArgs += listOf("-Xjvm-default=all") // opcional
    }

    buildFeatures {
        compose = true
        // ✅ Garante geração do BuildConfig
        buildConfig = true
    }

    // Mantém bundles dos modelos sem compressão
    androidResources {
        noCompress += setOf("tflite", "lite", "task")
    }

    packaging {
        resources {
            excludes += setOf(
                "META-INF/**",
                "LICENSE*",
                "license/**",
                "third_party/**"
            )
        }
    }
}

dependencies {
    // --- Compose / AndroidX (via Version Catalog)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)

    // CameraX
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)

    // TensorFlow Lite
    implementation(libs.tensorflow.lite)
    implementation(libs.tensorflow.lite.support)
    implementation(libs.tensorflow.lite.gpu)

    // MediaPipe Tasks (artefatos corretos)
    val mpVersion = "0.10.26.1"
    implementation("com.google.mediapipe:tasks-vision:$mpVersion")
    implementation("com.google.mediapipe:tasks-core:$mpVersion")

    // Se ativar desugaring nas compileOptions:
    // coreLibraryDesugaring(libs.desugar.jdk.libs)
}
