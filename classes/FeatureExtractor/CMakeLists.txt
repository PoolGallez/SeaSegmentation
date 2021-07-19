add_library(Extractor Extractor.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_include_directories (Extractor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
