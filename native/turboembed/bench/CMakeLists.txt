cmake_minimum_required(VERSION 3.20)
project(turboembed_openvino_probe LANGUAGES CXX)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCL REQUIRED)

add_executable(openvino_remote_probe openvino_remote_probe.cpp)
target_compile_features(openvino_remote_probe PRIVATE cxx_std_17)
target_compile_definitions(openvino_remote_probe PRIVATE OV_GPU_USE_OPENCL_HPP)
target_include_directories(openvino_remote_probe PRIVATE ../../../third_party)
target_link_libraries(openvino_remote_probe PRIVATE openvino::runtime OpenCL::OpenCL)
