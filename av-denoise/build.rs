fn main() {
    println!("cargo:rerun-if-changed=build.rs");

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

    check_ffmpeg_dir(&expected_dir);

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" => println!("cargo:rustc-link-lib=dylib=stdc++"),
        "macos" => println!("cargo:rustc-link-lib=dylib=c++"),
        _ => {},
    }
}

/// Checks `ffmpeg-sys-the-third` links FFmpeg from the same tree as ffms2.
///
/// That crate reads its headers and archives from `FFMPEG_DIR`. Without it the crate falls back to
/// pkg-config, which can resolve a system FFmpeg and put a second copy in the binary.
#[cfg(feature = "static-ffms2")]
fn check_ffmpeg_dir(ffms_lib_dir: &std::path::Path) {
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");

    let ffmpeg_dir = std::env::var_os("FFMPEG_DIR");
    let Some(ffmpeg_dir) = ffmpeg_dir else {
        panic!("FFMPEG_DIR must point at vcpkg_installed/<triplet>");
    };

    let ffmpeg_lib_dir = std::path::Path::new(&ffmpeg_dir).join("lib");
    let resolved = ffmpeg_lib_dir.canonicalize().ok();

    if resolved.as_deref() != Some(ffms_lib_dir) {
        panic!(
            "FFMPEG_DIR must be the vcpkg tree FFMS_LIB_DIR sits in, so ffms2 and the probe share one FFmpeg"
        );
    }
}
