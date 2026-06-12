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

# ── Librería externa D2CO (CMake puro, compilada fuera de colcon) ───────────
# Vendorizada dentro del propio paquete (external/d2co): el repo PAROL6 es
# autosuficiente. Hay que compilarla UNA vez antes del primer colcon build
# (ver README.md, sección "Compilación").
# Se puede sobreescribir con:  colcon build --cmake-args -DD2CO_ROOT=/otra/ruta
if(NOT DEFINED D2CO_ROOT)
  set(D2CO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/d2co")
endif()

set(D2CO_INCLUDE_DIR "${D2CO_ROOT}/include")
set(D2CO_LIB "${D2CO_ROOT}/lib/libd2co.so")
set(CV_EXT_INCLUDE_DIR "${D2CO_ROOT}/externals/cv_ext/include")
set(CERES_LOCAL_LIB_DIR "${D2CO_ROOT}/externals/ceres_1_14/install/lib")

if(NOT EXISTS "${D2CO_LIB}")
  message(FATAL_ERROR
    "No se encuentra libd2co.so en ${D2CO_LIB}.\n"
    "Compila primero la libreria D2CO (ver parol6_d2co/README.md) o pasa "
    "-DD2CO_ROOT=<ruta> a colcon.")
endif()

add_executable(d2co_pose_estimator_node src/d2co_pose_estimator_node.cpp)

target_include_directories(d2co_pose_estimator_node PRIVATE
  ${D2CO_INCLUDE_DIR}
  ${CV_EXT_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS})

# OpenMP: cv_ext/d2co lo usan en cabeceras y libd2co.so lo requiere en enlace.
target_link_libraries(d2co_pose_estimator_node
  ${D2CO_LIB}
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX)

ament_target_dependencies(d2co_pose_estimator_node
  rclcpp sensor_msgs geometry_msgs vision_msgs std_msgs std_srvs
  cv_bridge tf2 tf2_ros tf2_geometry_msgs ament_index_cpp)

# RPATH: que el ejecutable instalado encuentre libd2co.so y el Ceres 1.14
# local sin necesidad de LD_LIBRARY_PATH.
set_target_properties(d2co_pose_estimator_node PROPERTIES
  INSTALL_RPATH "${D2CO_ROOT}/lib;${CERES_LOCAL_LIB_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE)

install(TARGETS d2co_pose_estimator_node
  DESTINATION lib/${PROJECT_NAME})

# ── Herramientas offline: STL pre-escalado y test sin Gazebo ────────────────
install(PROGRAMS
  scripts/make_scaled_tube_stl.py
  scripts/test_d2co_synthetic.py
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
