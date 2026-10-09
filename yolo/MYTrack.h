#pragma once
#include <vector>
#include <unordered_map>
#include "BYTETracker.h"
#include "yolo_onnx.h"

class MYTrack
{
public:

    void resultStructTransfer(const std::vector<YolovOutput>& det_res, std::vector<Object>& object);

    // 纯变换：吃检测框、吐跟踪框，不碰图像，故无 cv::Mat 参数（不需要画框/显示）。
    void doTrack(const std::vector<Object>& res, std::vector<Object>& track_res);

    // 把轨迹 id 写回对应检测框（DrawPred 标签即显示 id:N）
    // 匹配依据：Object.rect 就来自对应检测框的 box，坐标一致
    static void tagTrackIds(const std::vector<Object>& tracks, std::vector<YolovOutput>& result) {
        for (size_t i = 0; i < tracks.size(); ++i) {
            for (size_t j = 0; j < result.size(); ++j) {
                YolovOutput& d = result[j];
                if (d.track_id != -1) continue;
                if (d.box.x == (int)tracks[i].rect.x && d.box.y == (int)tracks[i].rect.y &&
                    d.box.width == (int)tracks[i].rect.width && d.box.height == (int)tracks[i].rect.height) {
                    d.track_id = tracks[i].track_id;
                    break;
                }
            }
        }
    }


private:

    float trackDetIous(std::vector<float> lbox, cv::Rect rbox);


    int num_frames = 0;
    BYTETracker tracker;

};

// 便捷封装：以自由函数形式调用（内部转 MYTrack::tagTrackIds）
inline void tagTrackIds(const std::vector<Object>& tracks, std::vector<YolovOutput>& result) {
    MYTrack::tagTrackIds(tracks, result);
}
