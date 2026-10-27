project( d2co )

cmake_minimum_required(VERSION 3.0)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type automatically set to 'Release'")
    set(CMAKE_BUILD_TYPE "Release" )
else(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type: '${CMAKE_BUILD_TYPE}'") 
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "WARNING: you will get poor performance!")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)


option(USE_SSE "Build with SSE3+ support" ON)
option(USE_AVX "Build with AVX/AVX2 support" OFF)
option(USE_NEON "Build with NEON support" OFF)
option(USE_CAFFE_TRIPLET "Build with Caffe Triplet CNN support" OFF)

# This is to be complinat with OpenCV >= 3.3.1
set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -DOPENCV_TRAITS_ENABLE_DEPRECATED")

find_package(OpenMP REQUIRED)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  message(STATUS "Compiling for x86 / x86_64 architectures")
  
  if(USE_SSE)
      add_definitions(-DD2CO_USE_SSE )
      set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -msse3")
  endif(USE_SSE)
    
  if(USE_AVX)
      add_definitions(-DD2CO_USE_AVX )
      set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -mavx2 -mfma -march=core-avx2")
  endif(USE_AVX)
  
  set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Wall -Wpedantic -Wno-narrowing -Wno-deprecated -std=c++14 -march=nocona")
  
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} ${EXTRA_CXX_FLAGS} -O3")
  
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7)")
  message(STATUS "Compiling for ARM v7 architectures")
  
  if(USE_NEON)
      add_definitions(-DD2CO_USE_NEON )
  endif(USE_NEON)
  
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS} -std=c++14")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -std=c++14 -mcpu=cortex-a53 -mfloat-abi=hard -mfpu=neon-fp-armv8 -mneon-for-64bits -O3")  
else()
  message(STATUS "Unsupported architecture: using default compiler flags")
endif ()

find_package( OpenCV REQUIRED )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(Eigen3 REQUIRED)
find_package(PCL 1.6 REQUIRED )
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

add_definitions( ${PCL_DEFINITIONS}
                 -DTW_NO_LIB_PRAGMA
                 -DTW_NO_DIRECT3D
                 -DGLM_FORCE_RADIANS)

# Ceres disabled due to version 2.0 incompatibility
find_package(Ceres 1.14 REQUIRED)

find_library( DIME_LIBS dime )
find_library( GLEW_LIBS GLEW )
find_library( OPENMESHCORE_LIBS OpenMeshCore PATHS "/usr/local/lib/OpenMesh" )
find_package( Boost COMPONENTS program_options filesystem system REQUIRED )

if( NOT DEFINED CV_EXT_INCLUDE_DIR OR NOT DEFINED CV_EXT_LIBS )
  find_path(CV_EXT_INCLUDE_DIR cv_ext PATHS ${PROJECT_SOURCE_DIR}/externals/cv_ext/include ${PROJECT_SOURCE_DIR}/../cv_ext/include )
  find_library(CV_EXT_LIBS cv_ext PATHS ${PROJECT_SOURCE_DIR}/externals/cv_ext/lib ${PROJECT_SOURCE_DIR}/../cv_ext/lib )
endif( NOT DEFINED CV_EXT_INCLUDE_DIR OR NOT DEFINED CV_EXT_LIBS )

if(NOT DEFINED EIGEN3_INCLUDE_DIR)
  find_path(EIGEN3_INCLUDE_DIR eigen3)
endif(NOT DEFINED EIGEN3_INCLUDE_DIR)

include_directories( ./include
                      /usr/include
                      /usr/local/include
                      ${CERES_INCLUDE_DIRS}
                      ${CV_EXT_INCLUDE_DIR}
                      ${OpenCV_INCLUDE_DIRS}
                      ${Boost_INCLUDE_DIRS}
                      ${EIGEN3_INCLUDE_DIR}
                      ${PCL_INCLUDE_DIRS}
                      ${OPENGL_INCLUDE_DIRS}
                      ${GLFW_INCLUDE_DIRS})

if(USE_CAFFE_TRIPLET)
  find_library(Caffe_LIBS NAMES caffe HINTS /opt/caffe_triplet/lib)
  include_directories(/opt/caffe_triplet/include)
endif(USE_CAFFE_TRIPLET)                      

aux_source_directory( src D2CO_SRC )

add_library(d2co SHARED ${D2CO_SRC})
target_link_libraries( d2co PUBLIC
                       ${Boost_LIBRARIES}
                       ${DIME_LIBS}
                       ${OpenCV_LIBS}
                       ${CV_EXT_LIBS}
                       ${PCL_LIBRARIES}
                       ${CERES_LIBRARIES}
                       ${OPENGL_LIBRARY}
                       ${GLEW_LIBS}
                       ${GLFW_STATIC_LIBRARIES}
                       ${OPENMESHCORE_LIBS})
                       
set_target_properties(d2co PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set_target_properties(d2co PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# add_executable(test_multiview apps/test_multiview.cpp apps/apps_utils.cpp )
# target_link_libraries( test_multiview
#                        d2co)
# set_target_properties( test_multiview PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(test_localization apps/test_localization.cpp apps/apps_utils.cpp )
target_link_libraries( test_localization
                       d2co)
set_target_properties( test_localization PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(test_model apps/test_model.cpp apps/apps_utils.cpp )
target_link_libraries( test_model
                       d2co)
set_target_properties( test_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(test_line2D apps/test_line2D.cpp apps/apps_utils.cpp )
target_link_libraries( test_line2D
                       d2co
                       )
set_target_properties( test_line2D PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(test_distance_maps apps/test_distance_maps.cpp apps/apps_utils.cpp )
target_link_libraries( test_distance_maps
                       d2co)
set_target_properties( test_distance_maps PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(test_charger_arm apps/test_charger_arm.cpp apps/apps_utils.cpp )
target_link_libraries( test_charger_arm
                       d2co)
set_target_properties( test_charger_arm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(generate_models apps/generate_models.cpp apps/apps_utils.cpp )
target_link_libraries( generate_models
                       d2co)
set_target_properties( generate_models PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(detect_object apps/detect_object.cpp apps/apps_utils.cpp )
target_link_libraries( detect_object
                       d2co)
set_target_properties( detect_object PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if(USE_CAFFE_TRIPLET)

  add_executable(sphereview_data apps/sphereview_data.cpp )
  target_link_libraries( sphereview_data
                         d2co
                         ${Caffe_LIBS})
  set_target_properties( sphereview_data PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

  add_executable(cnn_3dobj_classify apps/cnn_3dobj_classify.cpp )
  target_link_libraries( cnn_3dobj_classify
                         d2co
                         ${Caffe_LIBS})
  set_target_properties( cnn_3dobj_classify PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

  add_executable(test_cnn_triplet apps/test_cnn_triplet.cpp )
  target_link_libraries( test_cnn_triplet
                         d2co
                         ${Caffe_LIBS})
  set_target_properties( test_cnn_triplet PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
  
endif(USE_CAFFE_TRIPLET)   
