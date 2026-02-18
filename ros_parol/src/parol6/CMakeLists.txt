cmake_minimum_required(VERSION 3.5)
project(parol6)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/virtual_gripper_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY urdf launch meshes config worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
