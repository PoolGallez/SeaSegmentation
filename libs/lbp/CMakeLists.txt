add_library(LBP lbp.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_include_directories (LBP PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
