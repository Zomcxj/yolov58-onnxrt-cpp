#include "yolo_onnx.h"
#include "MYTrack.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {

    if (argc < 4) {
        std::cout << "usage: " << argv[0] << " <model_path> <img_path> <task>" << std::endl;
        return -1;
    }
    std::string model_path = argv[1];
    std::string img_path = argv[2];
    std::string task = argv[3];    // {"det", "obb", "pose", "seg", "cls"};
    std::string label_file = argc > 4 ? argv[4] : "";   // 类别名 txt；空 = <模型名>.txt

    YolovOnnx yolo;
    yolo.labelFile = label_file;             // 库内加载：labelFile > <模型名>.txt > class_0..N
    if(!yolo.ReadModel(model_path, task, false)) {
        std::cout << "cannot load model" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cout << "cannot read image" << std::endl;
        return -1;
    }

    // 原图推理（默认）：result 已映射回原图坐标
    std::vector<YolovOutput> result;
    yolo.OnnxDetect(img, result);

    // 跟踪：tracks[i].rect 就是匹配上的检测框，且带 track_id，直接用
    MYTrack tracker;
    std::vector<Object> objs, tracks;
    tracker.resultStructTransfer(result, objs);
    tracker.doTrack(objs, tracks);

    std::cout << result.size() << " objs, " << tracks.size() << " tracks" << std::endl;

    // 滑窗推理（小目标增强，需要时替换上面的 OnnxDetect，并补 #include "slide_deal.h"）：
    // 重叠分轴：第 5、6 参 = 列(x)/行(y) 重叠像素；两轴同值时可用 7 参重载 (…, cv::Size(480,480), 200, 0.3f, 4)
    // std::vector<YolovOutput> result = tileDetectMultiThread(yolo, img, task, cv::Size(480, 480), 60, 240, 0.3f, 4);
    return 0;
}
