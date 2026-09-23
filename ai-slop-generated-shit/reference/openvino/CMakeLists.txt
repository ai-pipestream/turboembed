# Direct-native reference for the openvino provider: OpenVINO's C++ API
# alone on the token rows turbo-bench dumped.
#
#   cmake -S reference/openvino -B build/reference-openvino -DOpenVINO_DIR=$OV/runtime/cmake -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/reference-openvino
#   build/reference-openvino/reference-openvino --tokens tokens.json --device CPU --out native.json

cmake_minimum_required(VERSION 3.20)
project(turbo_reference_openvino LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

get_filename_component(TURBO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(reference-openvino main.cpp)
target_include_directories(reference-openvino PRIVATE ${TURBO_ROOT}/third_party)
target_link_libraries(reference-openvino PRIVATE openvino::runtime)
target_compile_options(reference-openvino PRIVATE -Wall -Wextra)
