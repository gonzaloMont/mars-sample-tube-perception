cmake_minimum_required(VERSION 3.8)
project(parol6_apps)

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

# instalar los programas como ejecutables 
install(PROGRAMS
    scripts/get_pose.py
    scripts/pose_ik_subscriber.py
    scripts/test_moveit.py
    scripts/test_moveit_cartesian.py
    scripts/udp_pose_receiver.py
    DESTINATION lib/${PROJECT_NAME}
)

# El script bash se instala en share (no necesita ser ejecutable por ros2 run)
install(FILES scripts/test_tubo.sh
    DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()
