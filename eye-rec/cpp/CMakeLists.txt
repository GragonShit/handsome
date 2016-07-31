
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(eye-rec)

IF(NOT CMAKE_BUILD_TYPE )
	SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

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

SET(TINY_CNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/tiny-cnn")
include_directories(${TINY_CNN_DIR})

#SET( tiny_cnn_hdrs ../../3rdparty/tiny_cnn/tiny_cnn/activations/activation_function.h  ../../3rdparty/tiny_cnn/tiny_cnn/io/cifar10_parser.h  ../../3rdparty/tiny_cnn/tiny_cnn/layers/convolutional_layer.h  ../../3rdparty/tiny_cnn/tiny_cnn/io/display.h  ../../3rdparty/tiny_cnn/tiny_cnn/util/image.h  ../../3rdparty/tiny_cnn/tiny_cnn/layers/layer.h  ../../3rdparty/tiny_cnn/tiny_cnn/lossfunctions/loss_function.h  ../../3rdparty/tiny_cnn/tiny_cnn/io/mnist_parser.h  ../../3rdparty/tiny_cnn/tiny_cnn/optimizers/optimizer.h  ../../3rdparty/tiny_cnn/tiny_cnn/util/product.h  ../../3rdparty/tiny_cnn/tiny_cnn/util/util.h
#../../3rdparty/tiny_cnn/tiny_cnn/layers/average_pooling_layer.h  ../../3rdparty/tiny_cnn/tiny_cnn/config.h  ../../3rdparty/tiny_cnn/tiny_cnn/util/deform.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/fully_connected_layer.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/input_layer.h  ../../3rdparty/tiny_cnn/tiny_cnn/layers/layers.h  ../../3rdparty/tiny_cnn/tiny_cnn/layers/max_pooling_layer.h  ../../3rdparty/tiny_cnn/tiny_cnn/network.h  ../../3rdparty/tiny_cnn/tiny_cnn/layers/partial_connected_layer.h  ../../3rdparty/tiny_cnn/tiny_cnn/tiny_cnn.h  ../../3rdparty/tiny_cnn/tiny_cnn/util/weight_init.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/arithmetic_layer.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/concat_layer.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/feedforward_layer.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/dropout_layer.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/lrn_layer.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/linear_layer.h ../../3rdparty/tiny_cnn/tiny_cnn/node.h ../../3rdparty/tiny_cnn/tiny_cnn/nodes.h ../../3rdparty/tiny_cnn/tiny_cnn/util/target_cost.h ../../3rdparty/tiny_cnn/tiny_cnn/util/aligned_allocator.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/deconvolutional_layer.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/max_unpooling_layer.h ../../3rdparty/tiny_cnn/tiny_cnn/layers/average_unpooling_layer.h)

ADD_EXECUTABLE(train_lenet train_lenet.cpp)
TARGET_LINK_LIBRARIES(train_lenet ${OpenCV_LIBS})

ADD_EXECUTABLE(test_lenet test_lenet.cpp)
TARGET_LINK_LIBRARIES(test_lenet ${OpenCV_LIBS})

# display env variable
message(STATUS)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "INCLUDE_DIRECTORIES: ")
foreach(dir ${dirs})
	message(STATUS "\t${dir}")
endforeach()

message(STATUS)
