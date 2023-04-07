
cmake_minimum_required(VERSION 3.15)
project(rapidasr)



if(WIN32)
add_definitions(-D_RAPIDASR_API_EXPORT)
else()

SET(SDK_ROOT  ${CMAKE_CURRENT_SOURCE_DIR}/sdk/linux)
include_directories(${SDK_ROOT}/include/  ${SDK_ROOT}/include/kenlm   ${SDK_ROOT}/include/ctc_decoder  ${SDK_ROOT}/include/kaldi)
message(UNIX)

link_directories(${SDK_ROOT}/lib /usr/lib)
endif()



include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SRC_FILE source/librpasrapi.cpp source/rapidasr.cpp )

set(SRC_FEAT feat/feature_pipeline.cpp  feat/fft.cpp)
add_library(rapidasr SHARED  ${SRC_FILE} ${SRC_FEAT})

link_libraries(fst kenlm kenlm_builder kenlm_filter kenlm_interpolate kenlm_util  ctc_decoder onnxruntime yaml-cpp bz2 lzma z)

add_executable(rapidasrtest source/testapp.cpp)
target_link_libraries(rapidasrtest PUBLIC rapidasr  ctc_decoder kenlm kenlm_builder kenlm_filter kenlm_interpolate kenlm_util)