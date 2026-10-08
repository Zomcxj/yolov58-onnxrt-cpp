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

void obboxFilter(std::vector<YolovOutput> result, std::vector<YolovOutput>& postResult, int lx, int ly, float conf);

void hbboxFilter(std::vector<YolovOutput> result, std::vector<YolovOutput>& postResult, int lx, int ly, float conf);