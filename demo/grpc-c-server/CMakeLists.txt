# Turbo gRPC demo server (C++ over the C ABI) and a matching client.
#
#   cargo build -p turbo-shared                      # target/debug/libturbo.so
#   cmake -S demo/grpc-c-server -B demo/grpc-c-server/build && cmake --build demo/grpc-c-server/build
#   demo/grpc-c-server/test.sh                       # starts the server on the mock bundle and calls it
#
# Needs gRPC C++ and protobuf (Ubuntu: libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev).
cmake_minimum_required(VERSION 3.20)
project(turbo_grpc_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED IMPORTED_TARGET grpc++ protobuf)
# protoc must match the libprotobuf pkg-config found; a newer protoc on PATH
# (a pip or release download) generates code the system library cannot
# compile, so the system protoc is looked up first.
find_program(PROTOC protoc PATHS /usr/bin /usr/local/bin NO_DEFAULT_PATH)
if(NOT PROTOC)
    find_program(PROTOC protoc REQUIRED)
endif()
execute_process(COMMAND "${PROTOC}" --version OUTPUT_VARIABLE PROTOC_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "protoc: ${PROTOC} (${PROTOC_VERSION}); libprotobuf ${GRPC_protobuf_VERSION}")
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

get_filename_component(TURBO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(TURBO_LIB_DIR "${TURBO_ROOT}/target/debug" CACHE PATH "Directory holding libturbo.so")
find_library(TURBO_LIB NAMES turbo PATHS "${TURBO_LIB_DIR}" NO_DEFAULT_PATH)
if(NOT TURBO_LIB)
    message(FATAL_ERROR "libturbo.so not found in ${TURBO_LIB_DIR}; run 'cargo build -p turbo-shared' or set -DTURBO_LIB_DIR")
endif()

set(PROTO "${CMAKE_CURRENT_SOURCE_DIR}/proto/turbo_demo.proto")
set(GEN "${CMAKE_CURRENT_BINARY_DIR}/gen")
file(MAKE_DIRECTORY "${GEN}")
add_custom_command(
    OUTPUT "${GEN}/turbo_demo.pb.cc" "${GEN}/turbo_demo.pb.h" "${GEN}/turbo_demo.grpc.pb.cc" "${GEN}/turbo_demo.grpc.pb.h"
    COMMAND "${PROTOC}" -I "${CMAKE_CURRENT_SOURCE_DIR}/proto" --cpp_out="${GEN}" --grpc_out="${GEN}"
            --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}" "${PROTO}"
    DEPENDS "${PROTO}"
    COMMENT "protoc turbo_demo.proto")

add_library(turbo_demo_proto STATIC "${GEN}/turbo_demo.pb.cc" "${GEN}/turbo_demo.grpc.pb.cc")
target_include_directories(turbo_demo_proto PUBLIC "${GEN}")
target_link_libraries(turbo_demo_proto PUBLIC PkgConfig::GRPC)

add_executable(turbo-grpc-server src/server.cpp)
target_include_directories(turbo-grpc-server PRIVATE "${TURBO_ROOT}/include")
target_compile_options(turbo-grpc-server PRIVATE -Wall -Wextra)
target_link_libraries(turbo-grpc-server PRIVATE turbo_demo_proto "${TURBO_LIB}")
set_target_properties(turbo-grpc-server PROPERTIES BUILD_RPATH "${TURBO_LIB_DIR}")

add_executable(turbo-grpc-client src/client.cpp)
target_compile_options(turbo-grpc-client PRIVATE -Wall -Wextra)
target_link_libraries(turbo-grpc-client PRIVATE turbo_demo_proto)
