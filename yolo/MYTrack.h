#pragma once
#include <vector>
#include "BYTETracker.h"
#include "yolo_onnx.h"

class MYTrack
{
public:

    void resultStructTransfer(const std::vector<YolovOutput>& det_res, std::vector<Object>& object);

    // 纯变换：吃检测框、吐跟踪框，不碰图像，故无 cv::Mat 参数（不需要画框/显示）。
    void doTrack(const std::vector<Object>& res, std::vector<Object>& track_res);


private:

    float trackDetIous(std::vector<float> lbox, cv::Rect rbox);
    int num_frames = 0;
    BYTETracker tracker;
};
