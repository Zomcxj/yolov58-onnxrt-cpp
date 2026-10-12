@echo off
rem 构建图形界面：必须用 Qt 自带 mingw730_64 工具链（勿用项目主编译器，ABI 不兼容）
rem 中间产物（Makefile/.obj/moc）在 build-qt\，发布产物在 build-guiset PATH=D:\software\Qt\Qt512\Tools\mingw730_64in;D:\software\Qt\Qt512.12.12\mingw73_64in;%PATH%
cd /d %~dp0
if not exist build-gui\imageformats mkdir build-gui\imageformats
if not exist build-qt mkdir build-qt
cd build-qt
qmake ../assets/gui.pro || goto :err
mingw32-make -j4 || goto :err
cd ..
del /q /s build-gui\*d.dll >nul 2>&1
rem ↑ 递归清理调试版插件，保证 Enigma 封装前目录精简
echo.
echo 构建完成: toolsuild-gui\YOLOInferGUI.exe
echo 运行需同目录 cli\ 子目录（含 yolov-onnxrt.exe 及其 DLL）
goto :eof
:err
echo 构建失败
exit /b 1
