cmake_minimum_required(VERSION 3.21)
project(ffms2 VERSION 5.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(FFMPEG REQUIRED)
find_package(ZLIB REQUIRED)

add_library(ffms2 STATIC
    src/core/audiosource.cpp
    src/core/ffms.cpp
    src/core/filehandle.cpp
    src/core/indexing.cpp
    src/core/track.cpp
    src/core/utils.cpp
    src/core/videosource.cpp
    src/core/videoutils.cpp
    src/core/zipfile.cpp
)

target_include_directories(ffms2 PRIVATE include src/core ${FFMPEG_INCLUDE_DIRS})
target_compile_definitions(ffms2 PRIVATE FFMS_EXPORTS FFMS_STATIC __STDC_CONSTANT_MACROS _FILE_OFFSET_BITS=64)
target_link_libraries(ffms2 PRIVATE ZLIB::ZLIB)

if(MSVC)
    target_compile_definitions(ffms2 PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set(FFMS2_VERSION ${PROJECT_VERSION})
configure_file(ffms2.pc.in ffms2.pc @ONLY)

install(TARGETS ffms2 ARCHIVE DESTINATION lib)
install(FILES include/ffms.h include/ffmscompat.h DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ffms2.pc DESTINATION lib/pkgconfig)
