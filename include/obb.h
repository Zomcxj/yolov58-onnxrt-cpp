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

// 带重叠(overlap)参数的切块：overlap 为相邻切块间的重叠像素（0 = 无重叠）。
// 原 imgSplit 固定 20%（且最少 200px），即 imgSplitWithOverlap(resImg, cropSize, max(cropSize/100*20,200))。
std::vector<std::vector<int>> imgSplitWithOverlap(cv::Mat resImg, int cropSize, int overlap);

// 多线程裁剪推理：切块 -> 线程池并行 OnnxDetect -> 跨块合并去重。
// task: "det" 用 hbboxFilter，"obb" 用 obboxFilter（其余任务请走原图推理）。
// overlap: >=0 指定重叠像素，<0 用自动(20%且最少200px)。threads<=0 时自动用 CPU 核数。
// 返回合并后的检测结果（已映射回原图坐标）。
std::vector<YolovOutput> tileDetectMultiThread(YolovOnnx& yolo,
                                               const cv::Mat& img,
                                               const std::string& task,
                                               int cropSize,
                                               int overlap,
                                               float conf,
                                               int threads);

void obboxFilter(std::vector<YolovOutput> result, std::vector<YolovOutput>& postResult, int lx, int ly, float conf);

void hbboxFilter(std::vector<YolovOutput> result, std::vector<YolovOutput>& postResult, int lx, int ly, float conf);