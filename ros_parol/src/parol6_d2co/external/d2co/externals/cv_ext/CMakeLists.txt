project( cv_ext )

cmake_minimum_required(VERSION 3.0)

message("** Project ${PROJECT_NAME} **")

set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   1   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   0   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

# Setup the default build environment
SET(CMAKE_AR "gcc-ar")
SET(CMAKE_NM "gcc-nm")
SET(CMAKE_RANLIB "gcc-ranlib")
SET(C "gcc")
SET(CXX "g++")

aux_source_directory( src CV_EXT_SRC )
add_library(${PROJECT_NAME} STATIC ${CV_EXT_SRC})

target_compile_options(${PROJECT_NAME} PUBLIC -std=c++14 -Wall -Wpedantic -Wno-narrowing -Wno-deprecated -O3 -ffast-math -fPIC)
# The following option enable  the standard link-time optimizer: small performance improvements
# at a cost of a much larger compilation time
# target_compile_options(${PROJECT_NAME} PUBLIC -flto -Wl,-flto)

if(NOT CMAKE_BUILD_TYPE)
    message("Build type automatically set to 'Release'")
    set(CMAKE_BUILD_TYPE "Release" )
else(NOT CMAKE_BUILD_TYPE)
    message("Build type: '${CMAKE_BUILD_TYPE}'")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "WARNING: you will get poor performance!")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

option(BUILD_EXAMPLES "Build examples" OFF)
option(GENERATE_DOCS "Generate documentation" OFF)

option(USE_SSE "Build with SSE3+ support" ON)
option(USE_AVX "Build with AVX/AVX2 support" OFF)
option(USE_NEON "Build with NEON support" OFF)


if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    message("Compiling for x86 / x86_64 architectures")

    target_compile_options(${PROJECT_NAME} PUBLIC -march=nocona)

    if(USE_AVX)
        message("AVX/AVX2 support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USE_AVX)
        target_compile_options(${PROJECT_NAME} PUBLIC -mavx2 -mfma -march=core-avx2)
    elseif(USE_SSE)
        message("SSE3+ support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USE_SSE)
        target_compile_options(${PROJECT_NAME} PUBLIC -msse3)
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7)")
    message("Compiling for ARM v7 architectures")

    target_compile_options(${PROJECT_NAME} PUBLIC -mcpu=cortex-a53 -mfloat-abi=hard)

    if(USE_NEON)
        message("NEON support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USE_NEON)
        target_compile_options(${PROJECT_NAME} PUBLIC -mfpu=neon-fp-armv8 -mneon-for-64bits)
    endif(USE_NEON)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)")
    message("Compiling for aarch64 architectures")

    target_compile_options(${PROJECT_NAME} PUBLIC -march=armv8-a+crypto -mcpu=cortex-a57+crypto)

    if(USE_NEON)
        message("NEON support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USE_NEON)
    endif(USE_NEON)

else()
    message(WARNING "Unsupported architecture: using default compiler flags")
endif ()


find_package(OpenMP REQUIRED)
target_compile_options(${PROJECT_NAME} PUBLIC  ${OpenMP_CXX_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# The previous line should be replaced in the future with:
# target_link_options(${PROJECT_NAME} PUBLIC  ${OpenMP_EXE_LINKER_FLAGS})

find_package( OpenCV REQUIRED )
# This is to be complinat with OpenCV >= 3.3.1
target_compile_definitions(${PROJECT_NAME} PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED )

find_package( Eigen3 REQUIRED )

find_package( PCL 1.6 REQUIRED )
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

find_package( Boost COMPONENTS program_options filesystem system REQUIRED )
# Ceres disabled due to version 2.0 incompatibility
# find_package( Ceres REQUIRED)
find_library( LIBUUID_LIBRARIES uuid)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

target_link_libraries( ${PROJECT_NAME} PUBLIC
                       ${OpenMP_CXX_LIBRARIES}
                       ${Boost_LIBRARIES}
                       ${OpenCV_LIBS}
                       ${PCL_LIBRARIES}
                       ${LIBUUID_LIBRARIES})
                       
target_include_directories( ${PROJECT_NAME} PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

if( BUILD_EXAMPLES )
add_subdirectory(examples)
endif( BUILD_EXAMPLES )

if( GENERATE_DOCS )
    add_subdirectory(doc)
endif( GENERATE_DOCS )

