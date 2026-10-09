CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

set(PROJECT_NAME yolov-onnxrt)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/../onnxruntime-win-x64-1.14.0)
set(ONNX_LIB ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll)
include_directories(${ONNXRUNTIME_DIR}/include)

# Eigen（ByteTrack 依赖，纯头文件）：可用 -DEIGEN3_DIR=<path> 覆盖
set(EIGEN3_DIR "D:/VsPro/PTZ/rknn/v8_rknn/3rdparty/eigen3" CACHE PATH "Eigen3 头文件根目录")
if(NOT EXISTS "${EIGEN3_DIR}/Eigen/Core")
    foreach(_eigen_cand
        "D:/software/Qt/vcpkg/packages/eigen3_x64-mingw-dynamic/include/eigen3"
        "C:/vcpkg/installed/x64-windows/include/eigen3")
        if(EXISTS "${_eigen_cand}/Eigen/Core")
            set(EIGEN3_DIR "${_eigen_cand}")
            break()
        endif()
    endforeach()
endif()
if(NOT EXISTS "${EIGEN3_DIR}/Eigen/Core")
    message(FATAL_ERROR "Eigen3 not found, pass -DEIGEN3_DIR=<dir containing Eigen/Core>")
endif()

include_directories(${CMAKE_SOURCE_DIR}/yolo)
include_directories(${CMAKE_SOURCE_DIR}/bytetrack)
include_directories(${EIGEN3_DIR})

aux_source_directory(yolo src1)
aux_source_directory(bytetrack src2)

add_executable(${PROJECT_NAME} ${src1} ${src2} ${CMAKE_SOURCE_DIR}/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNX_LIB} -lpthread)

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
