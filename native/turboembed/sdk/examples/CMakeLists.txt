cmake_minimum_required(VERSION 3.20)
project(TurboEmbedPreparedExample LANGUAGES C)

find_package(TurboEmbedPrepared CONFIG REQUIRED)

add_executable(turboembed_prepared_embed embed.c)
target_compile_features(turboembed_prepared_embed PRIVATE c_std_11)
target_link_libraries(turboembed_prepared_embed PRIVATE TurboEmbed::prepared)
if(UNIX)
  target_link_libraries(turboembed_prepared_embed PRIVATE m)
endif()
