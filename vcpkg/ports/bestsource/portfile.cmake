# Keep REF in step with the thirdparty/bestsource/vendor submodules.
vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO vapoursynth/bestsource
    REF a0fab5708b28920957679825797692f6e4a23674
    SHA512 5510dd757393f0d61d7f0244b107b10b26eb0503207897f0cc32dc21c6968f471f7fedade9fac3d1a68a6a7187a08555d3541e0fd26c7ad5d3cc15f06f481182
    HEAD_REF master
)

file(COPY
    "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_LIST_DIR}/bestsource.pc.in"
    DESTINATION "${SOURCE_PATH}"
)

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")
vcpkg_cmake_install()
vcpkg_fixup_pkgconfig()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
