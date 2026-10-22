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
  src/engine/Parameters.cpp
  src/engine/CoreEngine.cpp
  src/engine/ParameterFiler.cpp
  src/export/ExporterSettings.cpp
  src/export/Exporter.cpp
  src/export/SvgWriter.cpp
  src/export/ExportOutputStage.cpp
  src/preprocess/Preprocessor.cpp
  src/preprocess/EdgeImage.cpp
  src/preprocess/AADetector.cpp
  src/preprocess/FlowField.cpp
  src/preprocess/SignalImage.cpp
  src/classify/Classifier.cpp
  src/segment/PixelMap.cpp
  src/segment/QuadTree.cpp
  src/segment/PixelSegmenter.cpp
  src/segment/SubPixelSegmenter.cpp
  src/palette/ColorHistogram.cpp
  src/palette/KMeans.cpp
  src/palette/PaletteFinder.cpp
  src/palette/WeightsFinder.cpp
  src/vector/VectorImage.cpp
  src/vector/BoundaryTracer.cpp
  src/vector/GenerativeModel.cpp
  src/vector/EnergyTerms.cpp
  src/vector/Optimizer.cpp
  src/vector/CrossingGuard.cpp
  src/vector/ContourSmoother.cpp
  src/sla/BezierCurve.cpp
  src/fit/BezierFitter.cpp
  src/io/Inflate.cpp
  src/io/PngDecoder.cpp
  src/io/JpegDecoder.cpp
  src/io/ImageDecode.cpp)
target_include_directories(cnx_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(cnx_core PUBLIC Threads::Threads)

add_executable(cnx_cli apps/cnx_cli.cpp)
target_link_libraries(cnx_cli PRIVATE cnx_core)

enable_testing()
foreach(test_name header_smoke_test pixel_map_test pixel_segmenter_test vector_image_test boundary_tracer_test
                 generative_model_test energy_terms_test optimizer_test crossing_guard_test
                 contour_smoother_test bezier_curve_test bezier_fitter_test
                 sub_pixel_segmenter_test palette_test weights_finder_test
                 parameters_test preprocess_classify_test core_engine_test parameter_filer_test edge_classifier_test
                 flow_field_test signal_image_test image_decoder_test svg_export_test contour_quality_test)
  add_executable(${test_name} tests/unit/${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE cnx_core)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
# Arquivos de referÃªncia dos decodificadores (gerados por tests/data/make_fixtures.py e make_gdiplus_fixtures.ps1).
target_compile_definitions(image_decoder_test PRIVATE CNX_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME validate_math
           COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/tests/math/validate_math.py)
endif()
