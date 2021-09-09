cmake_minimum_required(VERSION 2.8)
### c++11 standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project( SeaSegmentation )

find_package( OpenCV REQUIRED )

add_executable( ${PROJECT_NAME} main.cpp )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )


include_directories( include )
#add_library(GraphSegmentation srcs/GraphSegmentation.cpp)
add_library(Extractor srcs/Extractor.cpp)
add_library(ImageLoader srcs/Loader.cpp)
add_library(BagOfWords srcs/BagOfWords.cpp)
add_library(SemanticSegmentor srcs/SemanticSegmentor.cpp)
#target_link_libraries( GraphSegmentation ${OpenCV_LIBS} )
target_link_libraries( Extractor ${OpenCV_LIBS} )
target_link_libraries( ImageLoader ${OpenCV_LIBS} )
target_link_libraries( BagOfWords ${OpenCV_LIBS} )
target_link_libraries( SemanticSegmentor ${OpenCV_LIBS} )
target_link_libraries(BagOfWords Extractor)
target_link_libraries(SemanticSegmentor BagOfWords)
target_link_libraries(${PROJECT_NAME} SemanticSegmentor)
target_link_libraries(${PROJECT_NAME} ImageLoader)

