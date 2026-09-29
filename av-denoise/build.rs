fn main() {
    #[cfg(feature = "static-ffms2")]
    link_static_ffms2();
}

/// Links the libraries a static ffms2 depends on.
///
/// `ffms2-sys` links the ffms2 archive but none of its dependencies, so the FFmpeg, codec and
/// system libraries are read from the static pkg-config description of ffms2. The description
/// must come from the same tree as `FFMS_LIB_DIR`, or a system ffms2 would pull in shared
/// FFmpeg libraries.
#[cfg(feature = "static-ffms2")]
fn link_static_ffms2() {
    println!("cargo:rerun-if-env-changed=FFMS_LIB_DIR");

    let lib_dir = std::env::var_os("FFMS_LIB_DIR");
    let Some(lib_dir) = lib_dir else {
        panic!("FFMS_LIB_DIR must point at vcpkg_installed/<triplet>/lib");
    };
    let Ok(expected_dir) = std::path::Path::new(&lib_dir).canonicalize() else {
        panic!("FFMS_LIB_DIR must point at vcpkg_installed/<triplet>/lib, but it does not resolve");
    };

    let probe = pkg_config::Config::new().statik(true).probe("ffms2");
    let library = match probe {
        Ok(library) => library,
        Err(error) => panic!(
            "no static ffms2 found, point PKG_CONFIG_PATH at \
             vcpkg_installed/<triplet>/lib/pkgconfig: {error}"
        ),
    };

    let found_expected = library
        .link_paths
        .iter()
        .filter_map(|path| path.canonicalize().ok())
        .any(|path| path == expected_dir);

    if !found_expected {
        panic!(
            "pkg-config resolved ffms2 outside FFMS_LIB_DIR, \
             point PKG_CONFIG_PATH at the same vcpkg tree"
        );
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" => println!("cargo:rustc-link-lib=dylib=stdc++"),
        "macos" => println!("cargo:rustc-link-lib=dylib=c++"),
        _ => {},
    }
}
