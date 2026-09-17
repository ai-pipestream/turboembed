# External consumer build for the packaged turboembed.h ABI. This project
# sees only the extracted SDK prefix through CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 3.20)
project(TurboEmbedConsumer C)

find_package(TurboEmbed CONFIG REQUIRED)

add_executable(turboembed_embed embed.c)
target_link_libraries(turboembed_embed PRIVATE TurboEmbed::turboembed m)

# The example resolves libturboembed and the ORT provider plugins from the
# extracted prefix without requiring the caller to export LD_LIBRARY_PATH.
find_library(TE_LIB_DIR_PROBE turboembed PATHS ${CMAKE_PREFIX_PATH}/lib NO_DEFAULT_PATH)
get_filename_component(TE_LIB_DIR ${TE_LIB_DIR_PROBE} DIRECTORY)
set_target_properties(turboembed_embed PROPERTIES
    BUILD_RPATH ${TE_LIB_DIR}
    INSTALL_RPATH ${TE_LIB_DIR})
