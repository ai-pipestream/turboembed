# Turbo Hailo provider: libturbo_provider_hailo.so
#
# Configure on a host with HailoRT 4.x installed (Raspberry Pi OS:
# `sudo apt install dkms hailo-all`; the header is /usr/include/hailo/hailort.h
# and the library /usr/lib/libhailort.so):
#   cmake -S providers/hailo -B build/hailo
#   cmake --build build/hailo
# Point HAILORT_INCLUDE_DIR / HAILORT_LIBRARY at another install.
#
# The library exports one symbol, turbo_provider_get, and links HailoRT and
# the in-tree WordPiece tokenizer.

cmake_minimum_required(VERSION 3.20)
project(turbo_provider_hailo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

find_path(HAILORT_INCLUDE_DIR hailo/hailort.h PATHS /usr/include /usr/local/include)
find_library(HAILORT_LIBRARY NAMES hailort PATHS /usr/lib /usr/local/lib)
if(NOT HAILORT_INCLUDE_DIR OR NOT HAILORT_LIBRARY)
    message(FATAL_ERROR
        "HailoRT was not found (hailo/hailort.h and libhailort). Install the hailo-all "
        "package or pass -DHAILORT_INCLUDE_DIR=... -DHAILORT_LIBRARY=...")
endif()
message(STATUS "HailoRT: ${HAILORT_LIBRARY} (${HAILORT_INCLUDE_DIR})")

get_filename_component(TURBO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_library(turbo_provider_hailo SHARED
    src/provider.cpp
    ${TURBO_ROOT}/native/wordpiece/vocab_load.cpp
    ${TURBO_ROOT}/native/wordpiece/encode.cpp
    ${TURBO_ROOT}/third_party/utf8proc/utf8proc.c
)

target_include_directories(turbo_provider_hailo PRIVATE
    ${TURBO_ROOT}/include
    ${TURBO_ROOT}/native/provider_common
    ${TURBO_ROOT}/native/wordpiece
    ${TURBO_ROOT}/native/wordpiece/include
    ${TURBO_ROOT}/third_party
    ${TURBO_ROOT}/third_party/utf8proc
    ${HAILORT_INCLUDE_DIR}
)

target_compile_definitions(turbo_provider_hailo PRIVATE UTF8PROC_STATIC)

target_compile_options(turbo_provider_hailo PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror=return-type -fno-strict-aliasing>
)

target_link_libraries(turbo_provider_hailo PRIVATE ${HAILORT_LIBRARY})

# Export only the provider entry point.
set_target_properties(turbo_provider_hailo PROPERTIES
    OUTPUT_NAME turbo_provider_hailo
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map -Wl,-z,defs"
)

install(TARGETS turbo_provider_hailo LIBRARY DESTINATION lib)

# Vtable-level tests (tests/provider_test.cpp). They dlopen the library
# rather than link it; the bundle comes from TURBO_LIVE_BUNDLE as in the
# Rust live tests, and a test whose bundle is not named skips and says so.
enable_testing()

add_executable(turbo_provider_hailo_test tests/provider_test.cpp)
add_dependencies(turbo_provider_hailo_test turbo_provider_hailo)
target_include_directories(turbo_provider_hailo_test PRIVATE ${TURBO_ROOT}/include)
target_compile_definitions(turbo_provider_hailo_test PRIVATE
    TURBO_PROVIDER_LIB_PATH="$<TARGET_FILE:turbo_provider_hailo>"
)
target_compile_options(turbo_provider_hailo_test PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror=return-type>
)
target_link_libraries(turbo_provider_hailo_test PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME provider COMMAND turbo_provider_hailo_test)
