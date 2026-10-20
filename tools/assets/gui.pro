# YOLOv58InferGUI：yolov58-onnxrt-cpp 推理测试工具（QProcess 架构，纯 Qt 进程）
#
# 必须用 Qt 配套的 mingw730_64 工具链构建（见 build_gui.cmd）：
# 项目主编译器是 mingw81-sjlj，与 Qt mingw73-seh 运行时混用会随机加载失败。

QT       += widgets multimediawidgets
CONFIG   += c++14
TARGET    = YOLOv58InferGUI
TEMPLATE  = app
DESTDIR   = $$PWD/../build-gui
OBJECTS_DIR = $$PWD/../build-qt/obj          # 中间产物出源码树（Makefile/.obj/moc 归 build-qt）
MOC_DIR     = $$PWD/../build-qt/moc
RC_ICONS  = $$PWD/app.ico                 # 资源管理器/任务栏图标
RESOURCES += icon.qrc                     # 运行时窗口图标（:/app.png）

SOURCES += ../main_qt.cpp ../main_window.cpp
HEADERS += ../main_window.h

# 运行时部署：Qt dll（mingw73 seh 运行时）+ 插件（仅 release，剔除调试版与多余格式）
win32 {
    QTBIN = $$[QT_INSTALL_BINS]
    QTPUGINS = $$[QT_INSTALL_PLUGINS]
    QTDLLS = Qt5Core Qt5Gui Qt5Widgets Qt5Network Qt5OpenGL Qt5Multimedia Qt5MultimediaWidgets \
             libgcc_s_seh-1 libstdc++-6 libwinpthread-1
    for(dll, QTDLLS) {
        QMAKE_POST_LINK += $$escape_expand(\\n\\t) $$QMAKE_COPY \
            $$shell_path($$QTBIN/$${dll}.dll) $$shell_path($$DESTDIR/$${dll}.dll)
    }
    # 插件：仅 release 必需文件（显式拷贝，避免 COPY_DIR 嵌套/调试版混入）
    PLUG = platforms/qwindows.dll mediaservice/dsengine.dll \
           mediaservice/qtmedia_audioengine.dll audio/qtaudio_windows.dll imageformats/qjpeg.dll
    for(p, PLUG) {
        QMAKE_POST_LINK += $$escape_expand(\\n\\t) $$QMAKE_COPY \
            $$shell_path($$QTPUGINS/$${p}) $$shell_path($$DESTDIR/$${p})
    }
}
