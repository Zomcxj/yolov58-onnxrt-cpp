#include "MYTrack.h"

// 数据类型转换
void MYTrack::resultStructTransfer(const std::vector<YolovOutput>& det_res, std::vector<Object>& object) {

	Object obj;
	for (int i = 0; i < det_res.size(); i++) {
        if (det_res[i].confidence > 0.4 && det_res[i].box.width * det_res[i].box.height > 20) {
            float x = max(det_res[i].box.x, 0);
            float y = max(det_res[i].box.y, 0);
            float w = det_res[i].box.width;
            float h = det_res[i].box.height;

            obj.rect = cv::Rect_<float>(x, y, w, h);
            obj.label = det_res[i].id;
            obj.prob = det_res[i].confidence;
            object.push_back(obj);
        }
	}
}

// 跟踪框检测框计算iou
float MYTrack::trackDetIous(std::vector<float> lbox, cv::Rect rbox) {

    float interBox[] = {
        (std::max)(lbox[0], (float)rbox.x), //left
        (std::min)(lbox[0] + lbox[2], (float)(rbox.x + rbox.width)), //right
        (std::max)(lbox[1], (float)rbox.y), //top
        (std::min)(lbox[1] + lbox[3], (float)(rbox.y + rbox.height)), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox.width * rbox.height - interBoxS);
}

// 跟踪
// 纯变换：只吃检测框、只吐跟踪框，不碰图像——所以没有 cv::Mat 参数。
// 输出语义：框来自匹配上的检测框(res)，不是卡尔曼预测框(tlwh)；tlwh 仅用于关联距离。
void MYTrack::doTrack(const std::vector<Object>& res, std::vector<Object>& track_res) {
    
    num_frames ++;
    #if DEBUG
        std::cout << "-------------------- << " << num_frames << " >> --------------------" << std::endl;
    #endif
    std::vector<STrack> output_stracks = tracker.update(res);

    for (int i = 0; i < output_stracks.size(); i++)
    {
        Object min_box;
        int min_distance = 10000;
        std::vector<float> tlwh = output_stracks[i].tlwh;
        float track_cx = tlwh[0] + tlwh[2] / 2;
        float track_cy = tlwh[1] + tlwh[3] / 2;
        for (int j = 0; j < res.size(); j++) {

            float det_cx = res[j].rect.x + res[j].rect.width / 2;
            float det_cy = res[j].rect.y + res[j].rect.height / 2;

            float iou = trackDetIous(tlwh, res[j].rect);
            int distance = std::sqrt(pow(track_cx - det_cx, 2) + pow(track_cy - det_cy, 2));

            if (min_distance > distance && iou > 0.2) {
                min_distance = distance;
                min_box = res[j];
            }
        }

        if (min_distance < 300) {
            min_box.track_id = output_stracks[i].track_id;
            min_box.track_time = output_stracks[i].track_time;
            min_box.vx = 0.0f;   // 速度（px/s）是 PTZ 需求，随 dt_s 一并移除
            min_box.vy = 0.0f;
            track_res.push_back(min_box);
        }
    }

    // for (int i = 0; i < output_stracks.size(); i++)
    // {
    //     vector<float> tlwh = output_stracks[i].tlwh;
    //     Scalar s = tracker.get_color(output_stracks[i].track_id);
    //     // putText(img, format("%d_%d", output_stracks[i].track_id, output_stracks[i].track_time), Point(tlwh[0] - 15, tlwh[1] + 20), 0, 0.7, Scalar(0, 0, 255), 2, cv::LINE_AA);
    //     putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0] - 15, tlwh[1] + 20), 0, 0.7, Scalar(0, 0, 255), 2, cv::LINE_AA);
    //     rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2, cv::LINE_AA);
        
    // }
}
