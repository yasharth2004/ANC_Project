cmake_minimum_required(VERSION 3.20)

project(HybridANC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hybridanc_dsp
    src/dsp.cpp
)

target_include_directories(hybridanc_dsp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(hybridanc_dsp PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(hybridanc_dsp PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(HYBRIDANC_BUILD_TESTS "Build HybridANC C++ tests" ON)
option(HYBRIDANC_BUILD_BENCHMARK "Build HybridANC C++ benchmark" ON)

if(HYBRIDANC_BUILD_TESTS)
    enable_testing()
    add_executable(hybridanc_dsp_tests
        tests/dsp_tests.cpp
    )
    target_link_libraries(hybridanc_dsp_tests PRIVATE hybridanc_dsp)
    add_test(NAME hybridanc_dsp_tests COMMAND hybridanc_dsp_tests)
endif()

if(HYBRIDANC_BUILD_BENCHMARK)
    add_executable(hybridanc_dsp_benchmark
        benchmark/dsp_benchmark.cpp
    )
    target_link_libraries(hybridanc_dsp_benchmark PRIVATE hybridanc_dsp)
endif()
