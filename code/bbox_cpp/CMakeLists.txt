cmake_minimum_required( VERSION 2.8 )
project(test)

set( CMAKE_CXX_FLAGS "-std=c++11 -std=c++14 -O3" )

# eigen 
include_directories( "/usr/include/eigen3/" )

# pcl 
find_package( PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

add_executable(test label2pcd.cpp )
target_link_libraries(test ${PCL_LIBRARIES} )
