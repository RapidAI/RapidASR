cmake_minimum_required(VERSION 3.15)
project(ctc_decoder_prj)

set(LIBNAME ctc_decoder)
set(APPNAME "testapp")
add_compile_options(-fPIC)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(WIN32)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/win-dep/include)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/win-dep/lib/x64)
else()


include_directories(/opt/qmcds/include /opt/qmcds/include/kenlm)

link_directories(/opt/qmcds/lib)
endif()

set(MAIN_SRC "ctc_beam_search_decoder.cpp" "decoder_utils.cpp" "path_trie.cpp" "scorer.cpp")
add_library(${LIBNAME} SHARED ${MAIN_SRC})

target_link_libraries(${LIBNAME} PUBLIC pthread fst kenlm kenlm_builder kenlm_filter kenlm_interpolate kenlm_util bz2 lzma z)


add_executable(${APPNAME} test/test.cpp)

target_link_libraries(${APPNAME} PUBLIC ctc_decoder kenlm kenlm_builder kenlm_filter kenlm_interpolate kenlm_util)



