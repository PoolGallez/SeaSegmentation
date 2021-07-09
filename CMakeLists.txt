cmake_minimum_required(VERSION 2.8)
project( SeaSegmentation )

add_subdirectory(libs/graph-segmentation)
add_subdirectory(libs/lbp)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( ${PROJECT_NAME} main.cpp )
target_link_libraries(${PROJECT_NAME} PUBLIC GraphSegmentation)
target_link_libraries(${PROJECT_NAME} PUBLIC LBP)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

