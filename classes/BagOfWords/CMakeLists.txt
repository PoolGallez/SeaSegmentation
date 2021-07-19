add_library(BagOfWords BagOfWords.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_include_directories (BagOfWords PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
