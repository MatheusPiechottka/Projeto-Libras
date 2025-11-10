pluginManagement {
    repositories {
        // Required for AndroidX and Google artifacts (CameraX, MediaPipe Tasks, etc.)
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        // Keep the same order here too
        google()
        mavenCentral()
    }
}

rootProject.name = "Libras"
include(":app")
