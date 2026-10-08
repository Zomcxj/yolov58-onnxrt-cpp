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

# ---------------------------------------------------------------------------
# Windows 下把运行时 DLL 复制到 exe 旁（exe 目录在 DLL 搜索顺序中优先于 System32/PATH，
# 避免加载到其他版本的同名 DLL 导致启动失败）：onnxruntime、MinGW 运行时、OpenCV、
# OpenCV 视频后端(opencv_videoio_ffmpeg*.dll，IP 流/视频读取必需)
# ---------------------------------------------------------------------------
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
