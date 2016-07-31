
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(eye-rec)

IF(NOT CMAKE_BUILD_TYPE )
	SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

SET(TINY_CNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/tiny-cnn")
include_directories(${TINY_CNN_DIR})
SET(DLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/dlib")
include_directories(${DLIB_DIR})

include(${DLIB_DIR}/dlib/cmake)

OPTION(USE_TBB        "Set to ON to use TBB"             OFF)
OPTION(USE_OMP        "Set to ON to use OMP"             OFF)
OPTION(USE_SSE        "Set to ON to use sse"             ON)
OPTION(USE_AVX        "Set to ON to use avx"             ON)

# ----------------------------------------------------------------------------
#   Find Dependencies
# ----------------------------------------------------------------------------

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

IF(USE_TBB)
    find_package(TBB REQUIRED)
    add_definitions(-DCNN_USE_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})
    SET(REQUIRED_LIBRARIES  ${REQUIRED_LIBRARIES} ${TBB_LIBRARIES}  )
ELSEIF(USE_OMP)
    find_package(OpenMP  REQUIRED)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions( -DCNN_USE_OMP)
ENDIF()
# ----------------------------------------------------------------------------
# COMPILER OPTIONS
# ----------------------------------------------------------------------------

IF(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    IF(USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        SET(EXTRA_C_FLAGS  " ${EXTRA_C_FLAGS} -msse3 ")
    ENDIF()
    IF(USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        SET(EXTRA_C_FLAGS  "${EXTRA_C_FLAGS} -mavx ")
    ENDIF()
    IF((NOT USE_TBB) AND (NOT USE_OMP))
        SET(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -pthread ")
    ENDIF()
    SET(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -std=c++11 -Wall -Wpedantic")
    SET(EXTRA_C_FLAGS_RELEASE " -O3 ")
    SET(EXTRA_C_FLAGS_DEBUG " -g3 ")
ELSE()#MSVC
    IF(USE_SSE )
        add_definitions(-DCNN_USE_SSE)
        SET(EXTRA_C_FLAGS  " ${EXTRA_C_FLAGS} /arch:SSE2 ")
    ENDIF()
    IF(USE_AVX  )
        add_definitions(-DCNN_USE_AVX)
        SET(EXTRA_C_FLAGS  "${EXTRA_C_FLAGS}  /arch:AVX ")
    ENDIF()
    SET(EXTRA_C_FLAGS_RELEASE " /Ox /bigobj ")
    add_definitions( -D _CRT_SECURE_NO_WARNINGS)
ENDIF()


    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE}")
    SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG}")

ADD_EXECUTABLE(train_lenet train_lenet.cpp)
TARGET_LINK_LIBRARIES(train_lenet ${OpenCV_LIBS})

ADD_EXECUTABLE(test_lenet test_lenet.cpp)
TARGET_LINK_LIBRARIES(test_lenet ${OpenCV_LIBS})

ADD_EXECUTABLE(demo demo.cpp)
TARGET_LINK_LIBRARIES(demo ${OpenCV_LIBS} dlib)

# display env variable
message(STATUS)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "INCLUDE_DIRECTORIES: ")
foreach(dir ${dirs})
	message(STATUS "\t${dir}")
endforeach()

message(STATUS)
