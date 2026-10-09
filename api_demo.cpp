// ============================================================================
// 项目对接示例：模型加载 -> 原图推理 -> MYTrack 跟踪 -> 跟踪后画框存图
// 对接时把 main() 里的调用流程抄进项目即可；本文件独立编译成 api_demo
//
// 用法:   api_demo <model.onnx> <img|dir> [--track]
// 类别名: 模型旁 <模型名>.txt（每行一个，可选，缺省 class_0..N）；或 yolo.labelFile 指定
// ============================================================================
#include "obb.h"
#include "yolo_onnx.h"
#include "MYTrack.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

static std::string pathBasename(const std::string& p) {
    std::string::size_type i = p.find_last_of("/\\");
    return (i == std::string::npos) ? p : p.substr(i + 1);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "usage: " << argv[0] << " <model.onnx> <img|dir> [--track]" << std::endl;
        return 1;
    }
    std::string modelPath = argv[1];
    std::string input     = argv[2];
    bool        track     = (argc > 3 && std::string(argv[3]) == "--track");

    YolovOnnx yolo;
    if (!yolo.ReadModel(modelPath, "det", false)) {          // GPU: 第 3 参传 true
        std::cout << "cannot load model: " << modelPath << std::endl;
        return 1;
    }

    MYTrack tracker;
    std::vector<std::string> files;
    try {
        cv::glob(input, files, false);
    } catch (const cv::Exception&) { files.clear(); }
    if (files.empty()) files.push_back(input);

    for (size_t k = 0; k < files.size(); ++k) {
        const std::string& f = files[k];
        cv::Mat img = cv::imread(f);
        if (img.empty()) { std::cout << "[warn] cannot read image: " << f << std::endl; continue; }

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

        // 原图推理（默认）：result 已映射回原图坐标
        std::vector<YolovOutput> result;
        yolo.OnnxDetect(img, result);

        std::vector<Object> objs, tracks;
        if (track) {
            tracker.resultStructTransfer(result, objs);
            tracker.doTrack(objs, tracks);
            tagTrackIds(tracks, result);                     // 轨迹 id 写回检测框
        }

        yolo.DrawPred(img, result);                          // 跟踪后画图：标签显示 id:N
        cv::imwrite(pathBasename(f), img);

        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::cout << pathBasename(f) << "  [" << result.size() << " objs, "
                  << tracks.size() << " tracks]  " << (int)(ms + 0.5) << " ms" << std::endl;
    }

    // 滑窗推理（小目标增强，需要时用下面这行替换上面的 OnnxDetect）：
    // std::vector<YolovOutput> result = tileDetectMultiThread(yolo, img, "det", cv::Size(800, 640), 200, 0.5f, 4);
    return 0;
}
