# Keep REF in step with the thirdparty/bestsource/vendor submodules.
vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO sekrit-twc/libp2p
    REF 869fa993041f9f3af7d9ac8b10158920c6ddce66
    SHA512 b5a575648db9b74cd27c5feb1740050864f2329d65c7b05744faf9200a20efb5cb3a374856c4139bb49630dca021aac7377a4b0df7a263abaa33c429a6dbd505
    HEAD_REF master
)

file(COPY
    "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_LIST_DIR}/libp2p.pc.in"
    DESTINATION "${SOURCE_PATH}"
)

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")
vcpkg_cmake_install()
vcpkg_fixup_pkgconfig()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")
