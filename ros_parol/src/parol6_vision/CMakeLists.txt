cmake_minimum_required(VERSION 3.8)
project(parol6_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


# ── Nodos de visión (ros2 run parol6_vision <nodo>) ─────────────────────────
install(PROGRAMS
    nodos/yolo_camera_subscriber.py
    nodos/pose_estimator_node.py
    nodos/pose_estimator_node2.py
    DESTINATION lib/${PROJECT_NAME}
)

# ── Configuración (pesos YOLO, parámetros, etc.) ────────────────────────────
install(DIRECTORY
    models
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
