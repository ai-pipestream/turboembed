cmake_minimum_required(VERSION 3.20)
project(TurboEmbedPrepared VERSION 1.0.0 LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The prepared SDK currently targets Linux OpenVINO only")
endif()
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCL REQUIRED)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
get_filename_component(TE_ROOT "../../.." ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(turboembed_prepared SHARED prepared.cpp
  "${TE_ROOT}/native/wordpiece/vocab_load.cpp"
  "${TE_ROOT}/native/wordpiece/encode.cpp"
  "${TE_ROOT}/third_party/utf8proc/utf8proc.c")
add_library(TurboEmbed::prepared ALIAS turboembed_prepared)
target_compile_features(turboembed_prepared PRIVATE cxx_std_17)
target_compile_definitions(turboembed_prepared PRIVATE
  TURBOEMBED_PREPARED_BUILD UTF8PROC_STATIC OV_GPU_USE_OPENCL_HPP)
target_include_directories(turboembed_prepared
  PUBLIC "$<BUILD_INTERFACE:${TE_ROOT}/include>" "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${TE_ROOT}/native/wordpiece" "${TE_ROOT}/third_party")
target_link_libraries(turboembed_prepared PRIVATE openvino::runtime OpenCL::OpenCL)
target_link_options(turboembed_prepared PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
  "LINKER:--disable-new-dtags")
set_property(TARGET turboembed_prepared APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_target_properties(turboembed_prepared PROPERTIES
  VERSION ${PROJECT_VERSION} SOVERSION 1 EXPORT_NAME prepared
  CXX_VISIBILITY_PRESET hidden C_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
  INSTALL_RPATH "$ORIGIN")

add_executable(turboembed-export-model export_model.cpp)
target_compile_features(turboembed-export-model PRIVATE cxx_std_17)
target_link_libraries(turboembed-export-model PRIVATE openvino::runtime)
target_link_options(turboembed-export-model PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(turboembed-export-model PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

option(TE_BUILD_CONTRACT_TEST "Build contract test requiring an Intel GPU and explicit bundle" OFF)
if(TE_BUILD_CONTRACT_TEST)
  add_executable(prepared_contract_test contract_test.cpp)
  target_compile_features(prepared_contract_test PRIVATE cxx_std_17)
  target_include_directories(prepared_contract_test PRIVATE "${TE_ROOT}/third_party")
  target_link_libraries(prepared_contract_test PRIVATE TurboEmbed::prepared OpenCL::OpenCL)
endif()

option(TE_BUILD_BENCHMARK "Build bounded direct OpenVINO versus ABI benchmark" OFF)
if(TE_BUILD_BENCHMARK)
  add_executable(binding_benchmark ../bench/binding_benchmark.cpp)
  target_compile_features(binding_benchmark PRIVATE cxx_std_17)
  target_include_directories(binding_benchmark PRIVATE "${TE_ROOT}/third_party")
  target_link_libraries(binding_benchmark PRIVATE TurboEmbed::prepared)
  add_executable(prepared_benchmark ../bench/prepared_benchmark.cpp)
  target_compile_features(prepared_benchmark PRIVATE cxx_std_17)
  target_compile_definitions(prepared_benchmark PRIVATE OV_GPU_USE_OPENCL_HPP)
  target_include_directories(prepared_benchmark PRIVATE "${TE_ROOT}/third_party")
  target_link_libraries(prepared_benchmark PRIVATE TurboEmbed::prepared openvino::runtime OpenCL::OpenCL)
endif()

install(TARGETS turboembed_prepared EXPORT TurboEmbedPreparedTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS turboembed-export-model RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES "${TE_ROOT}/include/turboembed_prepared.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(PROGRAMS "${TE_ROOT}/scripts/prepare-native-bundle.py" DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY examples/ DESTINATION ${CMAKE_INSTALL_DATADIR}/turboembed/examples)
install(FILES "${TE_ROOT}/LICENSE" DESTINATION ${CMAKE_INSTALL_DATADIR}/turboembed/licenses RENAME TurboEmbed-LICENSE)
install(FILES "${TE_ROOT}/third_party/utf8proc/LICENSE.md"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/turboembed/licenses RENAME utf8proc-LICENSE)
install(FILES "${TE_ROOT}/third_party/nlohmann/LICENSE.MIT"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/turboembed/licenses RENAME nlohmann-LICENSE)
install(FILES "${TE_ROOT}/third_party/picosha2/LICENSE"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/turboembed/licenses RENAME picosha2-LICENSE)
install(FILES "${TE_ROOT}/third_party/unicode_categories/LICENSE-APACHE"
  "${TE_ROOT}/third_party/unicode_categories/LICENSE-MIT"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/turboembed/licenses/unicode_categories)
install(EXPORT TurboEmbedPreparedTargets NAMESPACE TurboEmbed:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TurboEmbedPrepared)
configure_package_config_file(TurboEmbedPreparedConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/TurboEmbedPreparedConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TurboEmbedPrepared)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/TurboEmbedPreparedConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TurboEmbedPreparedConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/TurboEmbedPreparedConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TurboEmbedPrepared)
