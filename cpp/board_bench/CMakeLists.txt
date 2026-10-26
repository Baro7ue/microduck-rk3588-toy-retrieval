# Board C++ benchmark (Stage 2, plan 2H). Built natively on the RK3588:
#   cmake -S cpp/board_bench -B build -DRKNN_ROOT=<folder with include/rknn_api.h>
#   cmake --build build -j4 && ./build/unit_tests
# Without the NPU runtime (e.g. on a laptop) only the unit tests build: -DMD_WITH_RKNN=OFF.
cmake_minimum_required(VERSION 3.16)
project(md_board_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -ffp-contract=off: no fused multiply-add, so float arithmetic rounds as numpy's does (uyvy.cpp).
add_compile_options(-Wall -Wextra -ffp-contract=off)

option(MD_WITH_RKNN "build md_bench (needs rknn_api.h and librknnrt.so)" ON)

add_library(md_core STATIC src/uyvy.cpp src/frame_pack.cpp src/preprocess.cpp src/postprocess.cpp)
target_include_directories(md_core PUBLIC include)

add_executable(unit_tests tests/unit_tests.cpp)
target_link_libraries(unit_tests PRIVATE md_core)
enable_testing()
add_test(NAME unit_tests COMMAND unit_tests)

if(MD_WITH_RKNN)
  find_path(RKNN_INCLUDE_DIR rknn_api.h
            PATHS ${RKNN_ROOT}/include /usr/include /usr/local/include NO_DEFAULT_PATH)
  find_library(RKNN_LIBRARY rknnrt PATHS ${RKNN_ROOT}/lib /usr/lib /usr/lib64 /usr/local/lib)
  if(NOT RKNN_INCLUDE_DIR OR NOT RKNN_LIBRARY)
    message(FATAL_ERROR "rknn_api.h or librknnrt.so not found (RKNN_ROOT=${RKNN_ROOT}); "
                        "set -DRKNN_ROOT=... or -DMD_WITH_RKNN=OFF")
  endif()
  message(STATUS "rknn_api.h in ${RKNN_INCLUDE_DIR}, runtime ${RKNN_LIBRARY}")
  add_executable(md_bench src/main.cpp src/rknn_model.cpp)
  target_include_directories(md_bench PRIVATE ${RKNN_INCLUDE_DIR})
  target_link_libraries(md_bench PRIVATE md_core ${RKNN_LIBRARY})
endif()
