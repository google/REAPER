cmake_minimum_required(VERSION 2.8)
project(Reaper)

# Use C11 for unique_ptr
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

include_directories(".")

# Core functionality.
add_library(core STATIC core/file_resource.h
  core/file_resource.cc core/float_matrix.h
  core/float_matrix-inl.h core/float_matrix.cc core/track.h
  core/track.cc)

# Wave functionality.
add_library(wave STATIC wave/codec_riff.h wave/codec_riff.cc
  wave/codec_api.h wave/codec_api-inl.h wave/wave.h wave/wave.cc
  wave/wave_io.h wave/wave_io-inl.h wave/wave_io.cc)

# Epoch tracker.
add_library(epoch_tracker STATIC epoch_tracker/fft.h epoch_tracker/fft.cc
  epoch_tracker/fd_filter.h epoch_tracker/fd_filter.cc
  epoch_tracker/lpc_analyzer.cc epoch_tracker/lpc_analyzer.h
  epoch_tracker/epoch_tracker.cc epoch_tracker/epoch_tracker.h)

# Binary to extract F0.
add_executable(reaper epoch_tracker_main.cc)
target_link_libraries(wave core)
target_link_libraries(epoch_tracker wave)
target_link_libraries(reaper epoch_tracker)
