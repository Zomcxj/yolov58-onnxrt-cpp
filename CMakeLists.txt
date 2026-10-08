CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

set(PROJECT_NAME yolov-onnxrt)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(OpenCV_DIR /home/daway/cxj/opencv/compiled/opencv4.7.0/lib/cmake/opencv4)  #opencv root
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/../onnxruntime-win-x64-1.14.0)
set(ONNX_LIB ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll)
include_directories(${ONNXRUNTIME_DIR}/include)

include_directories(${CMAKE_SOURCE_DIR}/include)
aux_source_directory(src DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNX_LIB} -lpthread)
