add_library(ImageLoader Loader.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_include_directories (ImageLoader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
