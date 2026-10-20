#!/usr/bin/env bash
# 构建 Qt 图形界面（Linux / macOS / Windows-MSYS-GitBash 通用）
#
# 用法：
#   ./tools/build_gui.sh                 # 自动探测 Qt（qmake 在 PATH 或 QTDIR 指定）
#   QTDIR=/path/to/Qt/5.15.2/gcc_64 ./tools/build_gui.sh
#   MAKE=make ./tools/build_gui.sh       # 指定 make 程序（默认自动探测）
#
# 说明：
#   - 中间产物在 tools/build-qt/，发布产物在 tools/build-gui/
#   - Windows 上必须用 Qt 自带的 mingw 工具链（与 Qt 运行时同 ABI），
#     不要用 MSVC 或其它版本 mingw 编译，否则运行会 0xC0000139 加载失败。
#   - Windows 下的 DLL 部署由 gui.pro 的 win32 块处理；Linux/macOS 需自行
#     设置 LD_LIBRARY_PATH / DYLD_LIBRARY_PATH 或使用系统安装的 Qt。

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
BUILD_QT="$ROOT/build-qt"
BUILD_GUI="$ROOT/build-gui"

# ---------- 探测 qmake ----------
QMAKE_BIN=""
if [ -n "${QTDIR:-}" ]; then
    for c in "$QTDIR/bin/qmake" "$QTDIR/bin/qmake.exe"; do
        [ -x "$c" ] && QMAKE_BIN="$c" && break
    done
fi
if [ -z "$QMAKE_BIN" ] && command -v qmake >/dev/null 2>&1; then
    QMAKE_BIN="$(command -v qmake)"
fi
if [ -z "$QMAKE_BIN" ] && command -v qmake-qt5 >/dev/null 2>&1; then
    QMAKE_BIN="$(command -v qmake-qt5)"
fi
if [ -z "$QMAKE_BIN" ]; then
    echo "错误: 找不到 qmake" >&2
    echo "  请把 Qt 的 bin 目录加入 PATH，或用 QTDIR 指定 Qt 安装路径：" >&2
    echo "    QTDIR=/path/to/Qt/5.15.2/gcc_64 $0" >&2
    exit 1
fi
echo "qmake: $QMAKE_BIN"

# ---------- Windows：自动定位 Qt 自带 mingw 工具链并前置到 PATH ----------
# 原因：Qt 5.12 mingw73 构建使用 seh 异常模型，若 PATH 里的 g++ 是别的 mingw（如 sjlj），
# 链接会报 _Unwind_Resume / __gxx_personality_seh0 未定义。
# 规则：从 qmake 路径向上找 Qt 根目录，探测 Tools/mingw*/bin。
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        QT_ROOT=""
        d="$(cd "$(dirname "$QMAKE_BIN")" && pwd)"
        # 向上最多 4 层，找包含 Tools 或 5.x 的目录
        for _ in 1 2 3 4; do
            d="$(dirname "$d")"
            if [ -d "$d/Tools" ] || ls -d "$d"/[0-9]* >/dev/null 2>&1; then
                QT_ROOT="$d"; break
            fi
        done
        if [ -n "$QT_ROOT" ] && [ -d "$QT_ROOT/Tools" ]; then
            for tc in "$QT_ROOT"/Tools/mingw*/bin; do
                if [ -x "$tc/g++.exe" ] || [ -x "$tc/g++" ]; then
                    export PATH="$tc:$PATH"
                    echo "工具链: $tc"
                    break
                fi
            done
        fi
        ;;
esac

# ---------- 探测 make ----------
if [ -z "${MAKE:-}" ]; then
    for m in mingw32-make make gmake; do
        command -v "$m" >/dev/null 2>&1 && MAKE="$m" && break
    done
fi
if [ -z "${MAKE:-}" ]; then
    echo "错误: 找不到 make（可设 MAKE=... 指定）" >&2
    exit 1
fi
echo "make : $MAKE"

# ---------- 构建 ----------
mkdir -p "$BUILD_QT" "$BUILD_GUI"

# Windows 下插件目录（gui.pro 会拷贝 dll，这里确保目标目录存在）
mkdir -p "$BUILD_GUI/imageformats" "$BUILD_GUI/platforms" "$BUILD_GUI/mediaservice" "$BUILD_GUI/audio"

cd "$BUILD_QT"
"$QMAKE_BIN" "$ROOT/assets/gui.pro"
"$MAKE" -j"$(nproc 2>/dev/null || echo 4)"

cd "$ROOT"

# ---------- 清理调试版插件（仅 Windows 有 *d.dll 问题） ----------
if command -v find >/dev/null 2>&1; then
    find "$BUILD_GUI" -name '*d.dll' -type f -delete 2>/dev/null || true
fi

# 输出文件名随平台（Windows 带 .exe，Linux/macOS 无后缀）
EXE_SUFFIX=""
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) EXE_SUFFIX=".exe" ;;
esac

echo
echo "构建完成: tools/build-gui/YOLOv58InferGUI${EXE_SUFFIX}"
echo "运行：GUI 需与 cli/ 子目录同放（含 yolov58-onnxrt-cpp 及其依赖库）"
