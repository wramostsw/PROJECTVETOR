cmake_minimum_required(VERSION 3.21)
project(cnx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8 /Zc:__cplusplus)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(cnx_core
  src/segment/PixelMap.cpp
  src/segment/QuadTree.cpp
  src/segment/PixelSegmenter.cpp
  src/vector/VectorImage.cpp
  src/vector/BoundaryTracer.cpp
  src/vector/GenerativeModel.cpp)
target_include_directories(cnx_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

enable_testing()
foreach(test_name header_smoke_test pixel_map_test pixel_segmenter_test vector_image_test boundary_tracer_test
                 generative_model_test)
  add_executable(${test_name} tests/unit/${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE cnx_core)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME validate_math
           COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/tests/math/validate_math.py)
endif()
