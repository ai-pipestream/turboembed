# Turbo OpenVINO provider: libturbo_provider_openvino.so
#
# Configure with the OpenVINO archive distribution:
#   cmake -S providers/openvino -B build/openvino -DOpenVINO_DIR=$OV/runtime/cmake
#   cmake --build build/openvino
#
# The library exports one symbol, turbo_provider_get, and links OpenVINO,
# OpenCL (for GPU remote tensors), and the in-tree WordPiece tokenizer.

cmake_minimum_required(VERSION 3.20)
project(turbo_provider_openvino LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCL REQUIRED)

get_filename_component(TURBO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_library(turbo_provider_openvino SHARED
    src/provider.cpp
    ${TURBO_ROOT}/native/wordpiece/vocab_load.cpp
    ${TURBO_ROOT}/native/wordpiece/encode.cpp
    ${TURBO_ROOT}/third_party/utf8proc/utf8proc.c
)

target_include_directories(turbo_provider_openvino PRIVATE
    ${TURBO_ROOT}/include
    ${TURBO_ROOT}/native/wordpiece
    ${TURBO_ROOT}/native/wordpiece/include
    ${TURBO_ROOT}/third_party
    ${TURBO_ROOT}/third_party/utf8proc
)

target_compile_definitions(turbo_provider_openvino PRIVATE
    CL_HPP_TARGET_OPENCL_VERSION=300
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    CL_HPP_ENABLE_EXCEPTIONS
    UTF8PROC_STATIC
)

target_compile_options(turbo_provider_openvino PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror=return-type -fno-strict-aliasing>
)

target_link_libraries(turbo_provider_openvino PRIVATE openvino::runtime OpenCL::OpenCL)

# Export only the provider entry point.
set_target_properties(turbo_provider_openvino PROPERTIES
    OUTPUT_NAME turbo_provider_openvino
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map -Wl,-z,defs"
)

install(TARGETS turbo_provider_openvino LIBRARY DESTINATION lib)
