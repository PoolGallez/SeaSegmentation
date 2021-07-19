add_library(SemanticSegmentor SemanticSegmentor.cpp)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_include_directories (SemanticSegmentor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

