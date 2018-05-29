cmake_minimum_required( VERSION 2.8 )
project( LMedS-GMS )

#opencv
find_package( OpenCV REQUIRED )
include_directories(
	include
	${OpenCV_INCLUDE_DIRS}
)
add_executable( LMedS-GMS ./src/main.cpp)
target_link_libraries( LMedS-GMS ${OpenCV_LIBS} )
