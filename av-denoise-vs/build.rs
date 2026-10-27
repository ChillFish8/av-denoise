fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        // CubeCL reads its build ID through `_mh_execute_header`, which only executables define.
        // Deferring it to load time binds it to the host process instead.
        println!("cargo:rustc-cdylib-link-arg=-Wl,-U,__mh_execute_header");
    }
}
