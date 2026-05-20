cmake_minimum_required(VERSION 3.8)
project(parol6_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

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


# ── Módulo Python compartido (parol6_vision/) ───────────────────────────────
# Instala el paquete Python `parol6_vision` de forma que sea importable desde
# los nodos:  from parol6_vision.linemod_db import LinemodDatabase
ament_python_install_package(${PROJECT_NAME})


# ── Nodos de visión (ros2 run parol6_vision <nodo>) ─────────────────────────
install(PROGRAMS
    nodos/yolo_camera_subscriber.py
    nodos/yolo26n_camera_subscriber.py
    nodos/pose_estimator_node.py
    nodos/pose_estimator_node2.py
    nodos/synthetic_dataset_generator.py
    nodos/linemod_pose_estimator_node.py
    nodos/stereo_pointcloud_node.py
    DESTINATION lib/${PROJECT_NAME}
)

# ── Herramientas offline (NO son nodos ROS, pero se exponen con ros2 run) ───
install(PROGRAMS
    scripts/train_linemod_templates.py            # NUEVO: genera los templates
    scripts/eval_linemod_offline.py               # NUEVO: verificación offline
    DESTINATION lib/${PROJECT_NAME}
)


# ── Recursos: configuración, modelos y launch ───────────────────────────────
install(DIRECTORY
    models
    config
    launch
    DESTINATION share/${PROJECT_NAME}
)


ament_package()
