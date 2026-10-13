#include "yolo_onnx.h"
#include "slide_deal.h"
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
    yolo.labelFile = label_file;             // 加载：labelFile > <模型名>.txt > class_0..N
    if(!yolo.ReadModel(model_path, task, false)) {
        std::cout << "cannot load model" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cout << "cannot read image" << std::endl;
        return -1;
    }

    // ①原图推理（默认）：result 已映射回原图坐标
    std::vector<YolovOutput> result;
    yolo.OnnxDetect(img, result);

    // ②滑窗推理（小目标增强）：全参显式调用
    std::vector<YolovOutput> slideResult;
    OnnxDetectSlideMT(yolo, img, slideResult, 0.3f, cv::Size(640, 640), 200, 200, 4);
    std::cout << "slide: " << slideResult.size() << " objs" << std::endl;

    // 跟踪
    MYTrack tracker;
    std::vector<Object> objs, tracks;
    tracker.resultStructTransfer(result, objs);
    tracker.doTrack(objs, tracks);

    std::cout << result.size() << " objs, " << tracks.size() << " tracks" << std::endl;
    return 0;
}
