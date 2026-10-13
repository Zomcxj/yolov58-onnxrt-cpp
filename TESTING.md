# yolov-onnxrt-v2 测试指令全集

# 终端直接复制执行（Git Bash / Linux 通用）。开头先按平台定义可执行名，后面统一用 $BIN / $API。
`<...>` 占位符按实际替换：`<model.onnx>` 模型、`<img.jpg>` 图片、`<img_dir>` 图片目录、
`<video.mp4>` 视频、`<labels.txt>` 类别名 txt。行尾 `#` 注释 = 预期结果。

```bash
# 可执行名：Windows 用 .exe，Linux 去掉
BIN=./build/yolov-onnxrt.exe       # Linux: BIN=./build/yolov-onnxrt
API=./build/api_demo.exe           # Linux: API=./build/api_demo
# ========== 0. 编译 ==========
cmake -S . -B build && cmake --build build -j 4          # 改名/增删文件后必须重跑 cmake -S . -B build
cmake --build build -j 4                                  # 增量

# ========== 1. 基础推理 ==========
$BIN <model.onnx> <img.jpg> det                    # 单图 → <img>.jpg [N objs] xxx ms，结果 output/<img>.jpg
$BIN <model.onnx> <img_dir> det                    # 目录 → 逐张 + done. N image(s), avg X ms/img, total Y s
$BIN <model.onnx> "<img_dir>/*.jpg" det            # glob
$BIN <model.onnx> <video.mp4> det                  # 视频 → frame N [...] + done. saved: output/<视频名>_result.mp4
$BIN <model.onnx> <broken.jpg> det                 # 坏图 → [warn] cannot read image，不中断
$BIN <model.onnx> <notexist.mp4> det               # → cannot open video，退出码非0

# ========== 2. task 类型 ==========
$BIN <model.onnx> <img.jpg> det                    # 水平框
$BIN <obb模型> <img.jpg> obb --crop                # 旋转框跨块合并 obboxFilter
$BIN <pose模型> <img.jpg> pose                     # pose：crop 自动 off
$BIN <seg模型> <img.jpg> seg                       # seg：同上
$BIN <cls模型> <img.jpg> cls --track               # → [warn] task 'cls' has no boxes，无框
$BIN <model.onnx> <img.jpg> detx                   # → cannot load model

# ========== 3. 滑窗切块（仅 det/obb）==========
$BIN <model.onnx> <video.mp4> det --crop                                              # 默认640 + 重叠160x160 → 打印 crop tiles: img WxH -> tile 640x640, overlap 160x160 -> K tiles
$BIN <model.onnx> <video.mp4> det --crop --crop-overlap -1                            # -1=自动重叠(20%且>=200px)，覆盖默认160
$BIN <model.onnx> <video.mp4> det --crop --crop-size 480x480                          # 分块尺寸
$BIN <model.onnx> <video.mp4> det --crop --crop-size 640                              # 正方形简写
$BIN <model.onnx> <video.mp4> det --crop --crop-size 480x480 --crop-overlap 60x240    # 分轴重叠 → overlap 60x240
$BIN <model.onnx> <video.mp4> det --crop --crop-overlap 200                           # 单值=两轴同值
$BIN <model.onnx> <video.mp4> det --crop --crop-overlap 0                             # 无重叠
$BIN <model.onnx> <video.mp4> det --crop --crop-size 1280                             # 切块≥整图 → 1 tiles=原图直推（小图小目标会全漏，预期）
$BIN <model.onnx> <video.mp4> det --crop --crop-size 480x480 --crop-overlap 480x96    # 重叠=块宽 → 钳制不死循环
$BIN <model.onnx> <video.mp4> det --crop --threads 1                                  # 串行基准
$BIN <model.onnx> <video.mp4> det --crop --threads 4                                  # 并行，ms/帧下降
$BIN <model.onnx> <video.mp4> det --crop --threads 0                                  # 0=CPU核数

# ========== 4. 置信度 / NMS（均贯通：模型 NMS + 跨块合并 + 画框）==========
$BIN <model.onnx> <img.jpg> det --conf 0.3          # 弱检出也画
$BIN <model.onnx> <img.jpg> det --conf 0.8          # 只留高置信
$BIN <model.onnx> <img.jpg> det                     # conf 默认 0.5
$BIN <model.onnx> <img.jpg> det --nms 0.45          # NMS IoU 阈值（越小抑制越狠，保留框越少）
$BIN <model.onnx> <img.jpg> det --nms 0.9           # 几乎不去重（重叠目标如人群保留更多框）
$BIN <model.onnx> <img.jpg> det                     # nms 默认 0.6
$BIN <model.onnx> <img.jpg> det --conf 0.3 --nms 0.45 --track --draw-conf --draw-id   # 组合

# ========== 5. 跟踪 ==========
$BIN <model.onnx> <video.mp4> det --track           # frame N [..., M tracks] + 每轨一行
$BIN <model.onnx> <img_dir> det --track             # 图片序列跨图 id
$BIN <model.onnx> <video.mp4> det --track --no-save # 只看打印不落盘

# ========== 6. 保存 ==========
$BIN <model.onnx> <img.jpg> det                     # 默认 output/（目录不存在自动创建）
$BIN <model.onnx> <img.jpg> det --save-dir <DIR>    # 指定目录（自动创建）
$BIN <model.onnx> <video.mp4> det --no-save         # 不落盘

# ========== 7. 类别 txt ==========
$BIN <model.onnx> <img.jpg> det                                   # 约定：<模型目录>/<模型名>.txt 自动加载，load ... success !
$BIN <model.onnx> <img.jpg> det --labels <labels.txt>             # 显式指定（优先级最高）
$BIN <model.onnx> <img.jpg> det --labels <notexist.txt>           # → label file not found + cannot load model
mv <模型目录>/<模型名>.txt <模型目录>/<模型名>.txt.bak                                  # 都没有 → fallback to class_0..
$BIN <model.onnx> <img.jpg> det                                   # → no label file, fallback to class_0..
mv <模型目录>/<模型名>.txt.bak <模型目录>/<模型名>.txt                                  # 还原

# ========== 画图开关（默认：画图✓ 类别✓ 置信度✗ id✗）==========
$BIN <model.onnx> <img.jpg> det --no-draw                  # 只推理不画（结果图=原图；也不写结果视频的框）
$BIN <model.onnx> <img.jpg> det --no-class                 # 标签去掉类别名（只剩其他开启项）
$BIN <model.onnx> <img.jpg> det --draw-conf                # 标签 = 名字 + 置信度
$BIN <model.onnx> <img.jpg> det --track --draw-id          # 标签 = 名字 + id（id 仅跟踪时有值）
$BIN <model.onnx> <img.jpg> det --draw-conf --draw-id --track --conf 0.3   # 全开示例：名字 置信度 id

# ========== 8. 预览 ==========
$BIN <model.onnx> <video.mp4> det --show            # 窗口默认 800x500 可拖拽缩放；q/ESC 退出

# ========== 9. 逐帧导出标注（LabelMe json，调误检）==========
$BIN <model.onnx> <video.mp4> det --crop --crop-size <WxH> --crop-overlap <列x行> --conf 0.3 --labels <labels.txt> --threads 4 --export-frames --no-save
# → output/<视频名>/ 下每帧 <视频名>_NNNNNN.jpg + 同名 json（干净帧；空帧 shapes=[]；中断重跑自动跳过已导帧）
grep -c "^frame" <进度日志>                                              # 进度
python compose.py output/<视频名> output/<视频名>_composed.mp4 <原视频fps>               # 调整完合成（fps 必须按原视频）

# ========== 10. GPU ==========
./build/yolov-onnxrt <model.onnx> <video.mp4> det --crop --crop-size <WxH> --crop-overlap <列x行> --conf 0.3 --threads 4 --cuda --gpu 0
# → 横幅 Infer model on GPU! + nvidia-smi 有占用 + ms/帧大幅下降
# CPU 包跑 --cuda → Your ORT build without GPU. Change to CPU.（不崩）
# GPU 包但横幅回 CPU → export LD_LIBRARY_PATH=/path/to/onnxruntime-gpu/lib:$LD_LIBRARY_PATH

# ========== 11. api_demo（对接示例）==========
$API <model.onnx> <img.jpg> det                         # 单图原图推理 → N objs, M tracks（滑窗重载默认: 640x640, 重叠160x160, conf=-1=沿用 _classThreshold）
$API <model.onnx> <img.jpg> det <labels.txt>            # 第4参=类别txt（不传走 <模型名>.txt 约定）
$API <model.onnx> <img.jpg>                             # 参数不足 → usage，退出码 -1

# ========== 12. 边界 ==========
$BIN                                                # → usage
$BIN -h                                             # → usage
$BIN <model.onnx> <img.jpg> det --foo               # → unknown option: --foo
$BIN <model.onnx> <img.jpg> det --conf              # → missing value for --conf
$BIN <model.onnx> <img.jpg> det --conf abc          # → stoi 报错退出
$BIN <bad.onnx> <img.jpg> det                       # → cannot load model

# ========== 13. 坏流视频预处理 ==========
ffmpeg -err_detect ignore_err -i <broken.mp4> -c:v libx264 <rec.mp4>    # HEVC POC 错误先洗流再推理
```
