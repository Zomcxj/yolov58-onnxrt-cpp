CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

set(PROJECT_NAME yolov-onnxrt)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# ONNX Runtime / Eigen：换机器只需改这里的路径
if(WIN32)
    set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/../onnxruntime-win-x64-1.14.0)
    set(ONNX_LIB ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll)
else()
    set(ONNXRUNTIME_DIR /home/cxj/onnxruntime-linux-x64-1.14.0)
    set(ONNX_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
endif()
include_directories(${ONNXRUNTIME_DIR}/include)

include_directories(${CMAKE_SOURCE_DIR}/yolo)
include_directories(${CMAKE_SOURCE_DIR}/bytetrack)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/eigen3)

aux_source_directory(yolo src1)
aux_source_directory(bytetrack src2)
add_executable(${PROJECT_NAME} ${src1} ${src2} main.cpp)

# 对接示例（独立可执行，最小调用流程；同样需要 yolo/bytetrack 源码）
add_executable(api_demo api_demo.cpp ${src1} ${src2})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNX_LIB} pthread)
target_link_libraries(api_demo ${OpenCV_LIBS} ${ONNX_LIB} pthread)

# Linux 运行时免设 LD_LIBRARY_PATH
if(NOT WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")
endif()

# Windows：运行时 DLL 拷到 exe 旁
if(WIN32)
    file(GLOB _DLLS ${ONNX_LIB} ${OpenCV_DIR}/bin/libopencv_*.dll ${OpenCV_DIR}/bin/opencv_videoio_ffmpeg*.dll)
    get_filename_component(_MINGW_BIN ${CMAKE_CXX_COMPILER} DIRECTORY)
    list(APPEND _DLLS ${_MINGW_BIN}/libstdc++-6.dll ${_MINGW_BIN}/libgcc_s_sjlj-1.dll ${_MINGW_BIN}/libwinpthread-1.dll)
    foreach(d ${_DLLS})
        if(EXISTS "${d}")
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${d}" $<TARGET_FILE_DIR:${PROJECT_NAME}>)
        endif()
    endforeach()
endif()
