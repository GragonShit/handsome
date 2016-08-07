#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 2.8.4)

PROJECT(face-tracker)

IF(NOT CMAKE_BUILD_TYPE )
	SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

include_directories(/usr/X11/include)

SET(DLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/dlib")
include_directories(${DLIB_DIR})

include(${DLIB_DIR}/dlib/cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

ADD_EXECUTABLE(fhog_object_detector_ex fhog_object_detector_ex.cpp)
TARGET_LINK_LIBRARIES(fhog_object_detector_ex dlib ${OpenCV_LIBS})

