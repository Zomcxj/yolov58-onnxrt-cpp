# v58-onnxrt

YOLOv8 / v5 ONNX Runtime 推理工具：命令行 + Qt 图形界面。

> 要求 ONNX Runtime ≥ 1.13（旧版兼容代码已移除）。

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
tools/                        Qt GUI（源码 + assets + build-qt 中间产物 + build-gui 发布目录）
compose.py                    导出帧的标注回放合成（labelme json -> 视频）
api_demo.cpp                  集成调用示例
```

## 构建

CLI（MinGW，详见 CMakeLists）：

```
cmake -B build -G "MinGW Makefiles" && cmake --build build -j4
```

GUI：双击 `tools\build_gui.cmd`（需 Qt 5.12 + mingw730_64 工具链），产物在 `tools\build-gui\`。

## 推理

```
yolov-onnxrt.exe model.onnx <图片/目录/视频> det [options]
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
| `--min-track-frames` | 0（关） | 轨迹去抖：`--track` 时出现不足 N 帧不画不打 |

库层用法（`slide_deal.h`）：`OnnxDetectSlideMT(yolo, img, result[, cropSize, overlapW, overlapH, conf, threads])`，
参数顺序与全参版一致（conf 在第 7 位，`-1` = 沿用 `yolo._classThreshold`）；阈值也可直接改 `yolo._classThreshold` / `yolo._nmsThreshold` 成员。

ONNX Runtime 默认用 CPU 包（`-DONNXRUNTIME_DIR=` 可指向 GPU 包）；GPU 加速需 N 卡驱动 + CUDA 11.8 + cuDNN 8.x，缺任何一项自动回退 CPU。
