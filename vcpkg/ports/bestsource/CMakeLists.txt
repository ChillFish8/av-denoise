cmake_minimum_required(VERSION 3.21)
project(bestsource VERSION 22 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(FFMPEG REQUIRED)
find_path(XXHASH_INCLUDE_DIR xxhash.h REQUIRED)
find_path(P2P_INCLUDE_DIR p2p_api.h REQUIRED)

add_library(bestsource STATIC
    src/audiosource.cpp
    src/bsshared.cpp
    src/gpuhash.cpp
    src/tracklist.cpp
    src/videosource.cpp
)

target_include_directories(bestsource PRIVATE src ${FFMPEG_INCLUDE_DIRS} ${XXHASH_INCLUDE_DIR} ${P2P_INCLUDE_DIR})
target_compile_definitions(bestsource PRIVATE _FILE_OFFSET_BITS=64)

set(BESTSOURCE_VERSION ${PROJECT_VERSION_MAJOR})
configure_file(bestsource.pc.in bestsource.pc @ONLY)

install(TARGETS bestsource ARCHIVE DESTINATION lib)
install(FILES
    src/audiosource.h
    src/bsshared.h
    src/tracklist.h
    src/version.h
    src/videosource.h
    DESTINATION include/bestsource
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bestsource.pc DESTINATION lib/pkgconfig)
