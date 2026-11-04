cmake_minimum_required(VERSION 3.21)
project(libp2p LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(P2P_SOURCES
    p2p_api.cpp
    v210.cpp
    simd/cpuinfo_x86.cpp
    simd/p2p_simd.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    list(APPEND P2P_SOURCES simd/p2p_sse41.cpp)
    add_compile_definitions(P2P_SIMD)

    if(NOT MSVC)
        set_source_files_properties(simd/p2p_sse41.cpp PROPERTIES COMPILE_OPTIONS "-march=x86-64-v2")
    endif()
endif()

add_library(p2p STATIC ${P2P_SOURCES})
target_include_directories(p2p PRIVATE .)

configure_file(libp2p.pc.in libp2p.pc @ONLY)

install(TARGETS p2p ARCHIVE DESTINATION lib)
install(FILES p2p_api.h DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libp2p.pc DESTINATION lib/pkgconfig)
