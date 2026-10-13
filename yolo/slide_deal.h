#pragma once
#include <cmath>
#include <vector>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "yolo_onnx.h"
#define maxn 51
const double eps = 1E-8;

std::vector<std::vector<int>> imgSplit(cv::Mat resImg, int cropSize);

// 带重叠(overlap)参数的矩形切块：cropSize = cv::Size(宽, 高)，overlap 为相邻切块间的
// 重叠像素（0 = 无重叠）。块数按 ceil((边长-overlap)/(块边长-overlap)) 计算，
// 保证覆盖全图且不因重叠多算块。原 imgSplit 固定正方形 20% 重叠（最少 200px）。
std::vector<std::vector<int>> imgSplitWithOverlap(cv::Mat resImg, cv::Size cropSize, int overlap);

// 多线程裁剪推理：切块 -> 线程池并行 OnnxDetect -> 跨块合并去重。
// task: "det" 用 hbboxFilter，"obb" 用 obboxFilter（其余任务请走原图推理）。
// cropSize: cv::Size(宽,高) 切块尺寸；overlapW/H: >=0 指定各轴重叠像素，<0 用自动(20%且最少200px)。
// threads<=0 时自动用 CPU 核数。返回合并后的检测结果（已映射回原图坐标）。
std::vector<YolovOutput> OnnxDetectSlideMT(YolovOnnx& yolo,
                                               const cv::Mat& img,
                                               const std::string& task,
                                               cv::Size cropSize,
                                               int overlapW,
                                               int overlapH,
                                               float conf,
                                               int threads);

// 与 OnnxDetect 同形的便捷重载：不传的参数用默认值（task/阈值取自 yolo 成员、
// 切块 640x640、重叠 160x160、CPU 核数），所有参数都可在调用点覆盖。
inline bool OnnxDetectSlideMT(YolovOnnx& yolo, cv::Mat& img, std::vector<YolovOutput>& output,
                              float conf = -1.f, cv::Size cropSize = cv::Size(640, 640),
                              int overlapW = 160, int overlapH = 160, int threads = 0) {
	const float c = conf < 0 ? yolo._classThreshold : conf;   // -1 = 沿用成员阈值
	output = OnnxDetectSlideMT(yolo, img, yolo.GetTask(), cropSize, overlapW, overlapH, c, threads);
	return true;
}

void obboxFilter(std::vector<YolovOutput> result, std::vector<YolovOutput>& postResult, int lx, int ly, float conf);

void hbboxFilter(std::vector<YolovOutput> result, std::vector<YolovOutput>& postResult, int lx, int ly, float conf);