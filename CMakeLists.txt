CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

set(PROJECT_NAME yolov-onnxrt)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# Eigen 随项目放在 3rdparty/eigen3（不入库）
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/eigen3/Eigen/Core")
    message(FATAL_ERROR "3rdparty/eigen3 not found: copy eigen3 there first")
endif()

# ONNX Runtime：换平台只需改这一个路径（或 -DONNXRUNTIME_DIR=... 覆盖）
if(WIN32)
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/../onnxruntime-win-x64-1.14.0" CACHE PATH "ONNX Runtime 根目录")
    set(ONNX_LIB ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll)
else()
    set(ONNXRUNTIME_DIR "/home/cxj/onnxruntime-linux-x64-1.14.0" CACHE PATH "ONNX Runtime 根目录")
    set(ONNX_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/yolo)
include_directories(${CMAKE_SOURCE_DIR}/bytetrack)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/eigen3)

aux_source_directory(yolo src1)
aux_source_directory(bytetrack src2)

add_executable(${PROJECT_NAME} ${src1} ${src2} ${CMAKE_SOURCE_DIR}/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNX_LIB} pthread)

# 运行时直接找到 libonnxruntime.so，无需设 LD_LIBRARY_PATH
if(NOT WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")
endif()

if(WIN32)
    # 运行时 DLL 拷到 exe 旁（exe 目录优先于 System32/PATH，避免加载到其他版本）
    set(_RUNTIME_DLLS ${ONNX_LIB})

    get_filename_component(_MINGW_BIN ${CMAKE_CXX_COMPILER} DIRECTORY)
    list(APPEND _RUNTIME_DLLS
        ${_MINGW_BIN}/libstdc++-6.dll
        ${_MINGW_BIN}/libgcc_s_sjlj-1.dll
        ${_MINGW_BIN}/libwinpthread-1.dll
    )

    file(GLOB _OPENCV_DLLS ${OpenCV_DIR}/bin/libopencv_*.dll ${OpenCV_DIR}/bin/opencv_videoio_ffmpeg*.dll)
    list(APPEND _RUNTIME_DLLS ${_OPENCV_DLLS})

    foreach(_dll IN LISTS _RUNTIME_DLLS)
        if(EXISTS "${_dll}")
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" $<TARGET_FILE_DIR:${PROJECT_NAME}>)
        endif()
    endforeach()
endif()
