cmake_minimum_required(VERSION 3.8)
project(parol6_d2co)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# ── OpenCV: fijar SIEMPRE el del sistema (4.5.4, el mismo que usa cv_bridge) ─
# En esta máquina existe un OpenCV 4.14 en /usr/local que find_package
# encontraría primero; mezclar dos OpenCV en el mismo proceso corrompe el ABI.
if(NOT DEFINED OpenCV_DIR)
  set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
endif()
find_package(OpenCV REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
# cv_ext/types.h (incluida por las cabeceras de d2co) usa tipos de PCL.
find_package(PCL REQUIRED COMPONENTS common)

# ── Librería externa D2CO (vendorizada en external/d2co) ────────────────────
# colcon la compila sola: primero cv_ext (estática) y después libd2co.so,
# con el OpenCV del sistema y el Ceres 1.14 precompilado de externals/. Son
# proyectos CMake independientes (no paquetes ROS), así que se construyen
# como ExternalProject en el build/ de colcon; sus propios CMakeLists dejan
# las librerías en external/d2co/lib y externals/cv_ext/lib (ignoradas por
# git). Las dependencias de sistema (OpenMesh, glog, GLFW...) siguen siendo
# necesarias: ver README.md, "Compilación".
include(ExternalProject)
set(D2CO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/d2co")
set(CV_EXT_SRC_DIR "${D2CO_SRC_DIR}/externals/cv_ext")
set(CERES_LOCAL_DIR "${D2CO_SRC_DIR}/externals/ceres_1_14/install")
set(D2CO_INCLUDE_DIR "${D2CO_SRC_DIR}/include")
set(CV_EXT_INCLUDE_DIR "${CV_EXT_SRC_DIR}/include")
set(CV_EXT_LIB "${CV_EXT_SRC_DIR}/lib/libcv_ext.a")
set(D2CO_LIB "${D2CO_SRC_DIR}/lib/libd2co.so")

ExternalProject_Add(cv_ext_build
  SOURCE_DIR ${CV_EXT_SRC_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cv_ext
  # cv_ext incluye cabeceras de Ceres pero su CMakeLists no lo busca: se le
  # pasan las del Ceres 1.14 vendorizado (las mismas que usa libd2co).
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=${OpenCV_DIR}
             -DCMAKE_CXX_FLAGS=-I${CERES_LOCAL_DIR}/include
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target cv_ext --parallel
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${CV_EXT_LIB})

ExternalProject_Add(d2co_build
  SOURCE_DIR ${D2CO_SRC_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/d2co
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=${OpenCV_DIR}
             -DCeres_DIR=${CERES_LOCAL_DIR}/lib/cmake/Ceres
             -DCV_EXT_INCLUDE_DIR=${CV_EXT_INCLUDE_DIR} -DCV_EXT_LIBS=${CV_EXT_LIB}
             # libd2co.so busca libceres a su lado (install/<pkg>/lib)
             -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH=$ORIGIN
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target d2co --parallel
  INSTALL_COMMAND ""
  DEPENDS cv_ext_build
  BUILD_BYPRODUCTS ${D2CO_LIB})

add_executable(d2co_pose_estimator_node src/d2co_pose_estimator_node.cpp)
add_dependencies(d2co_pose_estimator_node d2co_build)

target_include_directories(d2co_pose_estimator_node PRIVATE
  ${D2CO_INCLUDE_DIR}
  ${CV_EXT_INCLUDE_DIR}
  ${CERES_LOCAL_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS})

# OpenMP: cv_ext/d2co lo usan en cabeceras y libd2co.so lo requiere en enlace.
target_link_libraries(d2co_pose_estimator_node
  ${D2CO_LIB}
  ${CERES_LOCAL_DIR}/lib/libceres.so
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX)

ament_target_dependencies(d2co_pose_estimator_node
  rclcpp sensor_msgs geometry_msgs vision_msgs std_msgs std_srvs
  cv_bridge tf2 tf2_ros tf2_geometry_msgs ament_index_cpp)

# libd2co.so y el Ceres 1.14 se instalan junto al paquete (install/<pkg>/lib)
# y el ejecutable (install/<pkg>/lib/<pkg>/) los busca ahí vía RPATH: el
# install no depende del árbol de fuentes ni de LD_LIBRARY_PATH.
set_target_properties(d2co_pose_estimator_node PROPERTIES
  INSTALL_RPATH "$ORIGIN/.."
  BUILD_WITH_INSTALL_RPATH TRUE)

install(TARGETS d2co_pose_estimator_node
  DESTINATION lib/${PROJECT_NAME})
install(FILES ${D2CO_LIB} DESTINATION lib)
install(DIRECTORY ${CERES_LOCAL_DIR}/lib/ DESTINATION lib
  FILES_MATCHING PATTERN "libceres.so*" PATTERN "cmake" EXCLUDE)

# ── Herramientas offline: STL pre-escalado y test sin Gazebo ────────────────
install(PROGRAMS
  scripts/make_scaled_tube_stl.py
  scripts/test_d2co_synthetic.py
  scripts/test_d2co_pose.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
