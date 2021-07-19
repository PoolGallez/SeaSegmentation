add_library(GraphSegmentation GraphSegmentation.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_include_directories (GraphSegmentation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
