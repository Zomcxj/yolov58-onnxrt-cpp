# yolov58-onnxrt-cpp

YOLOv8 / v5 ONNX Runtime 推理工具：命令行 + Qt 图形界面。

> 要求 ONNX Runtime ≥ 1.13。

## 功能

- 任务：det / obb / pose / seg / cls
- 滑窗切块推理（大图小目标），参数可在 GUI 原图上实时预览切块网格
- 视频抽帧、目标跟踪（ByteTrack）
- 导出 labelme json：图片模式存原图同目录；视频模式逐帧原图+json（断点续跑）
- GPU 推理（CUDA EP），缺环境自动回退 CPU
- GUI 双栏预览（原图/结果）、实时帧、结果视频回放

## 目录

```
main.cpp  yolo/  bytetrack/   CLI 推理（C++ / ONNX Runtime / OpenCV）
tools/                        Qt GUI（源码 + assets + build-qt + build-gui 发布目录）
compose.py                    导出帧的标注回放合成（labelme json -> 视频）
api_demo.cpp                  集成调用示例
```

构建产物名：

| 目标 | 文件名 | 说明 |
|---|---|---|
| CLI | `yolov58-onnxrt-cpp.exe` | CMake 构建，命令行与 api_demo 同源 |
| GUI | `YOLOv58InferGUI.exe` | qmake 构建，QProcess 调 CLI（`cli/` 子目录） |
| 对接示例 | `api_demo.exe` | 最小 C++ 调用流程 |

## 构建

CLI（MinGW，详见 CMakeLists）：

```
cmake -B build -G "MinGW Makefiles" && cmake --build build -j4
```

GUI（需 Qt 5.12+，跨平台脚本）：

```bash
./tools/build_gui.sh                                  # 自动探测 qmake/make
QTDIR=/path/to/Qt/5.12.12/gcc_64 ./tools/build_gui.sh # 指定 Qt 路径
```

产物在 `tools/build-gui/`。Windows 下脚本会自动定位 Qt 自带 mingw 工具链（避免 sjlj/seh ABI 冲突）。

## 推理

```
yolov58-onnxrt-cpp.exe model.onnx <图片/目录/视频> det [options]
```

常用：`--conf` `--nms` `--crop` `--crop-size` `--crop-overlap` `--export-frames` `--cuda`；完整见 `--help`。

关键参数默认值（CLI 与库层便捷重载一致）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--conf` | 0.5 | 置信度阈值，检测 NMS / 跨块合并 / 画框三处贯通 |
| `--nms` | 0.6 | NMS IoU 阈值，越小抑制越狠、保留框越少 |
| `--crop-size` | 640 | 滑窗切块边长（默认正方形，支持 `800x640` 分轴） |
| `--crop-overlap` | 160 | 切块重叠（默认两轴同值，支持 `60x240` 分轴；`-1` = 自动 20% 且 ≥200px） |
| `--threads` | CPU 核数 | 切块并行推理线程数 |

库层用法（`slide_deal.h`）：`OnnxDetectSlideMT(yolo, img, result[, cropSize, overlapW, overlapH, conf, threads])`，
参数顺序与全参版一致（conf 在第 7 位，`-1` = 沿用 `yolo._classThreshold`）；阈值也可直接改 `yolo._classThreshold` / `yolo._nmsThreshold` 成员。

ONNX Runtime 默认用 CPU 包（`-DONNXRUNTIME_DIR=` 可指向 GPU 包）；GPU 加速需 N 卡驱动 + CUDA 11.8 + cuDNN 8.x，缺任何一项自动回退 CPU。

## 依赖与构建

| 依赖 | 版本 | 获取方式 |
|---|---|---|
| CMake | ≥ 3.14 | 系统包管理器 |
| OpenCV | 4.x | 系统安装（`libopencv-dev` / OpenCV 官方包） |
| ONNX Runtime | **≥ 1.13** | 下载 CPU 或 GPU 包，用 `-DONNXRUNTIME_DIR=` 指路 |
| Eigen3 | 3.3+ | **不入库**：本地 `3rdparty/eigen3` 存在则直接用，否则 CMake 自动下载 3.4.0 |
| Qt 5 | 5.12+ | 仅 GUI 需要（`tools/`，qmake 构建） |

最小构建（CLI + api_demo）：

```bash
cmake -S . -B build -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0
cmake --build build -j$(nproc)
```

GUI 构建见上方「构建」一节（`tools/build_gui.sh`）。

## 许可

本项目采用 **MIT License**，详见 [LICENSE](LICENSE)。

### 推理核心来源（Apache 2.0）

`yolo/yolo_onnx.h` 与 `yolo/yolo_onnx.cpp` 的推理逻辑**参考并修改自**
[UNeedCryDear/yolov8-opencv-onnxruntime-cpp](https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp)
（**Apache License 2.0**）。

原项目按任务拆分为 5 组文件（`yolov8_onnx` / `yolov8_seg_onnx` / `yolov8_pose_onnx` /
`yolov8_obb_onnx` / `yolov8_utils`），本项目将其**合并为单一 `YolovOnnx` 类**，并做了以下改造：

- 删除 ONNX Runtime < 1.13 的旧版 API 兼容分支，统一走 OrtApi 函数表（同一份代码兼容 CPU/GPU 包）
- 新增 CUDA Execution Provider 启用与失败回退 CPU（`--cuda` / `--gpu`）
- 新增滑窗切块推理（`yolo/slide_deal.*`，独立实现：分块 + 多线程 + 跨块 IoU 去重）
- 新增标签渲染开关 `drawClass`/`drawConf`/`drawId` 与 `DrawPredTrack`（直接渲染跟踪结果）
- 新增 `LoadClassNames`（按 `<模型名>.txt` 约定加载，缺失回退 `class_0..N`）
- 抽取画图公共函数 `drawScaleThickness` / `boxColor` / `drawLabelBg`
- 修复 seg 掩膜无条件整图克隆（改为仅 `task=="seg"` 时）

本项目中 `yolo/yolo_onnx.h/.cpp` 为该 Apache-2.0 项目的衍生作品，依其原协议随本仓库分发；`main.cpp`、`api_demo.cpp`、`tools/`（Qt GUI）为本项目原创（MIT）。完整归属与修改说明见 [LICENSE](LICENSE) 第 0 节，Apache-2.0 全文见 [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)。

### 其他组件

| 组件 | 许可 | 说明 |
|---|---|---|
| ByteTrack (`bytetrack/`) | MIT | Copyright (c) 2021 Yifu Zhang，全文保留在 LICENSE |
| Eigen3 | MPL2 | **不入库**，构建期获取（本地或 CMake 自动下载） |
| ONNX Runtime | MIT | 运行时动态链接 |
| OpenCV | Apache 2.0 | 运行时动态链接 |
| Qt 5 | LGPL v3 | 运行时动态链接（仅 GUI） |

上述运行时库均不随仓库分发；**分发二进制包时请自行附带其许可文本**。
