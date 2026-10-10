#include "slide_deal.h"
#include "yolo_onnx.h"
#include "MYTrack.h"
#include <opencv2/core/utils/filesystem.hpp>
#include <chrono>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>



// 参数配置
struct Options {
    std::string modelPath;   // argv[1]
    std::string input;       // argv[2]：图片/目录 / 视频文件 / RTSP 流
    std::string task;        // argv[3]：det|obb|pose|seg|cls
    bool   crop = false;     // 裁剪推理（仅 det/obb）
    cv::Size cropSize{1280, 1280}; // 切块尺寸（宽×高）
    int    cropOverlapW = -1; // 列(x)重叠像素；-1 = 自动
    int    cropOverlapH = -1; // 行(y)重叠像素；-1 = 自动；0 = 无重叠
    float  conf = 0.5f;      // 跨块合并置信度阈值
    int    threads = 0;      // 裁剪并行线程数；0 = 自动
    bool   track = false;    // --track：检测后接 ByteTrack 跟踪（cls 无框，已在 main 里关闭）
    std::string labels;      // --labels：类别名 txt；空 = <模型名>.txt
    std::string savePath = "output/";
    bool   save = true;      // --no-save 关闭保存
    bool   show = false;     // 视频预览
    bool   draw = true;      // --no-draw：只推理不画
    bool   drawClass = true;  // 标签显示类别名（默认开）
    bool   drawConf = false;  // 标签显示置信度（--draw-conf）
    bool   drawId = false;    // 标签显示跟踪 id（--draw-id，仅跟踪时有值）
    bool   exportFrames = false; // --export-frames：逐帧导出 图片+LabelMe json（便于调整误检）
    bool   isCuda = false;
    int    cudaID = 0;
};

// 按 --crop 分发（原图推理 / 滑窗推理）
static void detectOne(YolovOnnx& yolo, cv::Mat& img, const Options& opt,
                      std::vector<YolovOutput>& result) {
    if (opt.crop && (opt.task == "det" || opt.task == "obb")) {
        result = tileDetectMultiThread(yolo, img, opt.task, opt.cropSize,
                                       opt.cropOverlapW, opt.cropOverlapH,
                                       opt.conf, opt.threads);
    } else {
        yolo.OnnxDetect(img, result);
    }
}

// 输入类型自动判断
enum class InputType { Image, VideoFile };
static std::string toLower(std::string s) {
    for (size_t i = 0; i < s.size(); ++i)
        s[i] = (char)std::tolower((unsigned char)s[i]);
    return s;
}

static InputType detectInputType(const std::string& input) {
    static const char* exts[] = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"};
    std::string l = toLower(input);
    for (const char* e : exts) {
        size_t n = std::strlen(e);
        if (l.size() > n && l.compare(l.size() - n, n, e) == 0)
            return InputType::VideoFile;
    }
    return InputType::Image;
}

static std::string pathBasename(const std::string& p) {
    std::string::size_type i = p.find_last_of("/\\");
    return (i == std::string::npos) ? p : p.substr(i + 1);
}

static std::string pathStem(const std::string& p) {
    std::string b = pathBasename(p);
    std::string::size_type i = b.find_last_of('.');
    return (i == std::string::npos || i == 0) ? b : b.substr(0, i);
}

static std::string withTrailingSep(std::string dir) {
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
        dir += "/";
    return dir;
}

using Clock = std::chrono::steady_clock;
static double elapsedMs(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 打印跟踪结果（每条轨迹一行）
static void printTracks(const std::vector<Object>& tracks) {
    for (size_t i = 0; i < tracks.size(); ++i) {
        const Object& t = tracks[i];
        std::cout << "    track id=" << t.track_id
                  << " class=" << t.label
                  << " conf=" << t.prob
                  << " box=(" << (int)t.rect.x << "," << (int)t.rect.y << ","
                  << (int)t.rect.width << "," << (int)t.rect.height << ")"
                  << " track_time=" << t.track_time << std::endl;
    }
}

static void printInfo(const Options& opt, const char* mode) {
    std::cout << "model: " << opt.modelPath
              << " | task: " << opt.task
              << " | mode: " << mode
              << " | crop: " << (opt.crop ? ("on(size=" + std::to_string(opt.cropSize.width) +
                                             "x" + std::to_string(opt.cropSize.height) +
                                             (opt.cropOverlapW >= 0 || opt.cropOverlapH >= 0
                                                  ? ",ov=" + std::to_string(opt.cropOverlapW) + "x" + std::to_string(opt.cropOverlapH)
                                                  : "") +
                                             ",threads=" + std::to_string(opt.threads) + ")")
                                          : "off")
              << " | track: " << (opt.track ? "on" : "off")
              << std::endl;
}

// ----------------------------------------------------------------------------
// 测试：文件夹遍历推理（glob 图片/目录，逐张存结果图）
// ----------------------------------------------------------------------------
static int runImageMode(YolovOnnx& yolo, const Options& opt) {
    std::vector<std::string> files;
    try {
        cv::glob(opt.input, files, false);
    } catch (const cv::Exception&) {   // 目录打不开等情况（如误传 URL）
        files.clear();
    }
    if (files.empty()) files.push_back(opt.input);   // glob 不匹配时按单文件处理

    std::string savePath = withTrailingSep(opt.savePath);
    if (opt.save && !savePath.empty())      // imwrite 不会自建目录
        cv::utils::fs::createDirectories(savePath);
    MYTrack tracker;                      // --track：跨图片保持轨迹
    std::vector<Object> tracks;
    int nFail = 0;
    double totalMs = 0;
    for (size_t k = 0; k < files.size(); ++k) {
        const std::string& f = files[k];
        cv::Mat img = cv::imread(f);
        if (img.empty()) { std::cerr << "[warn] cannot read image: " << f << std::endl; ++nFail; continue; }

        Clock::time_point t0 = Clock::now();
        std::vector<YolovOutput> result;
        detectOne(yolo, img, opt, result);
        double ms = elapsedMs(t0);
        totalMs += ms;

        if (opt.track) {                       // 检测 -> 跟踪（直接用 MYTrack）
            std::vector<Object> objs;
            tracker.resultStructTransfer(result, objs);
            tracks.clear();
            tracker.doTrack(objs, tracks);
        }

        if (opt.draw && opt.task != "cls") {              // cls 无框可画；--no-draw 跳过
            if (opt.track) MYTrack::tagTrackIds(tracks, result);
            yolo.DrawPred(img, result);
            if (opt.save) cv::imwrite(savePath + pathBasename(f), img);
        }
        std::cout << pathBasename(f) << "  [" << result.size() << " objs";
        if (opt.track) std::cout << ", " << tracks.size() << " tracks";
        std::cout << "]  " << (int)(ms + 0.5) << " ms" << std::endl;
        if (opt.track) printTracks(tracks);
    }
    std::cout << "done. " << files.size() << " image(s), " << nFail << " failed"
              << ", avg " << (files.size() ? (int)(totalMs / files.size() + 0.5) : 0) << " ms/img"
              << ", total " << totalMs / 1000.0 << " s" << std::endl;
    return (nFail > 0 && nFail == (int)files.size()) ? 1 : 0;
}

// ----------------------------------------------------------------------------
// 测试：本地视频文件推理（存结果视频）
// ----------------------------------------------------------------------------
// LabelMe 5.x 格式（rectangle 四点），便于标注工具调整误检；空 shapes 也导（漏检可补标）
static void writeLabelMeJson(const std::string& jsonPath, const std::string& imageName,
                             const cv::Mat& img, const std::vector<YolovOutput>& result,
                             const YolovOnnx& yolo) {
    std::ofstream f(jsonPath);
    f << std::setprecision(15);
    f << "{\n  \"version\": \"5.0.1\",\n  \"flags\": {},\n  \"shapes\": [\n";
    bool first = true;
    for (size_t i = 0; i < result.size(); ++i) {
        const YolovOutput& r = result[i];
        if (r.box.area() <= 0) continue;   // 只导水平框（det）
        double x1 = r.box.x,               y1 = r.box.y;
        double x2 = r.box.x + r.box.width, y2 = r.box.y + r.box.height;
        std::string label = (r.id >= 0 && r.id < (int)yolo._className.size())
                                ? yolo._className[r.id] : ("class_" + std::to_string(r.id));
        if (!first) f << ",\n";
        first = false;
        f << "    {\n"
          << "      \"label\": \"" << label << "\",\n"
          << "      \"points\": [\n"
          << "        [" << x1 << ", " << y1 << "],\n"
          << "        [" << x2 << ", " << y1 << "],\n"
          << "        [" << x2 << ", " << y2 << "],\n"
          << "        [" << x1 << ", " << y2 << "]\n"
          << "      ],\n"
          << "      \"group_id\": null,\n"
          << "      \"description\": \"\",\n"
          << "      \"shape_type\": \"rectangle\",\n"
          << "      \"flags\": {}\n"
          << "    }";
    }
    f << "\n  ],\n"
      << "  \"imagePath\": \"" << imageName << "\",\n"
      << "  \"imageData\": null,\n"
      << "  \"imageHeight\": " << img.rows << ",\n"
      << "  \"imageWidth\": " << img.cols << "\n}\n";
}

static int runVideoMode(YolovOnnx& yolo, const Options& opt) {
    cv::VideoCapture cap;
    cap.open(opt.input);
    if (!cap.isOpened()) { std::cerr << "cannot open video: " << opt.input << std::endl; return 1; }

    std::string savePath = withTrailingSep(opt.savePath);
    if (opt.save && !savePath.empty())      // VideoWriter 不会自建目录
        cv::utils::fs::createDirectories(savePath);

    // 输出：存结果视频
    cv::VideoWriter writer;
    std::string videoOutPath;
    bool saveVideo = false;
    if (opt.save) {
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 1e-3 || fps != fps || fps > 240) fps = 25.0;
        cv::Size size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        videoOutPath = savePath + pathStem(opt.input) + "_result.mp4";
        writer.open(videoOutPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
        if (!writer.isOpened()) {   // 编码器不可用时回退 avi/XVID
            videoOutPath = savePath + pathStem(opt.input) + "_result.avi";
            writer.open(videoOutPath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, size);
        }
        if (!writer.isOpened()) { std::cerr << "[warn] cannot open VideoWriter, skip saving" << std::endl; videoOutPath.clear(); }
        else saveVideo = true;
    }

    cv::Mat frame;
    long long frameIdx = 0;
    double totalMs = 0;
    MYTrack tracker;                      // --track：跨帧保持轨迹
    std::vector<Object> tracks;
    if (opt.show) {                        // 预览窗口：默认 800x500，可拖拽缩放（不再原尺寸）
        cv::namedWindow("yolov-onnxrt", cv::WINDOW_NORMAL);
        cv::resizeWindow("yolov-onnxrt", 800, 500);
    }
    std::string exportDir;
    if (opt.exportFrames) {                // 逐帧图片+json：导干净帧（画框前）
        exportDir = savePath + pathStem(opt.input) + "/";
        cv::utils::fs::createDirectories(exportDir);
    }
    while (cap.read(frame)) {
        Clock::time_point t0 = Clock::now();
        std::string expName;
        if (opt.exportFrames) {
            std::ostringstream nm;
            nm << pathStem(opt.input) << "_" << std::setw(6) << std::setfill('0') << frameIdx;
            expName = nm.str();
            std::ifstream probe(exportDir + expName + ".json");
            if (probe.good()) { ++frameIdx; continue; }   // 断点续跑：已导出帧跳过（不重复检测）
        }
        std::vector<YolovOutput> result;
        detectOne(yolo, frame, opt, result);
        double ms = elapsedMs(t0);

        if (opt.track) {                       // 检测 -> 跟踪（直接用 MYTrack）
            std::vector<Object> objs;
            tracker.resultStructTransfer(result, objs);
            tracks.clear();
            tracker.doTrack(objs, tracks);
        }

        if (opt.exportFrames) {                // 导干净帧 + LabelMe json（在 DrawPred 之前，画的是原帧）
            try {
                cv::imwrite(exportDir + expName + ".jpg", frame);
                writeLabelMeJson(exportDir + expName + ".json", expName + ".jpg", frame, result, yolo);
            } catch (const cv::Exception& e) { // 磁盘/内存异常不崩整跑
                std::cerr << "[warn] export failed @" << expName << ": " << e.what() << std::endl;
            }
        }

        if (opt.draw && opt.task != "cls") {
            if (opt.track) MYTrack::tagTrackIds(tracks, result);
            yolo.DrawPred(frame, result);
        }

        if (saveVideo) writer.write(frame);
        if (opt.show) {
            cv::imshow("yolov-onnxrt", frame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
        }
        totalMs += ms;
        ++frameIdx;
        std::cout << "frame " << frameIdx << "  [" << result.size() << " objs";
        if (opt.track) std::cout << ", " << tracks.size() << " tracks";
        std::cout << "]  " << (int)(ms + 0.5) << " ms" << std::endl;
        if (opt.track) printTracks(tracks);
    }
    std::cout << "done. " << frameIdx << " frame(s), avg "
              << (frameIdx ? (int)(totalMs / frameIdx + 0.5) : 0) << " ms/frame"
              << ", total " << totalMs / 1000.0 << " s";
    if (saveVideo) std::cout << ", saved: " << videoOutPath;
    std::cout << std::endl;
    return 0;
}

// ----------------------------------------------------------------------------
// 命令行解析
// ----------------------------------------------------------------------------
static void printUsage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " <model.onnx> <input> <task> [options]\n"
        "\n"
        "  <input>  自动判断：*.mp4/*.avi 等    -> 本地视频文件(存结果视频)\n"
        "                    其余(图片/目录/glob) -> 文件夹遍历推理(存结果图)\n"
        "  <task>   det | obb | pose | seg | cls\n"
        "\n"
        "Options:\n"
        "  --crop             启用裁剪（切块）推理，仅 det/obb 生效\n"
        "  --crop-size WxH   切块尺寸，默认 1280；支持 800x640（宽x高）或 1280（正方形）\n"
        "  --crop-overlap WxH   切块重叠像素：列x行分轴，如 60x240；单值 N = 两轴同值；默认自动(20%且≥200px)\n"
        "  --threads N        裁剪并行线程数，默认 = CPU 核数\n"
        "  --conf F           置信度阈值（检测 NMS + 跨块合并 + 画框），默认 0.5\n"
        "  --track            检测后接 ByteTrack 跟踪（打印 track id 并画框；cls 忽略）\n"
        "  类别名              模型旁 <模型名>.txt，每行一个类别（可选，缺省 class_0..N）\n"
        "  --labels FILE     类别名 txt（每行一个；缺省 <模型名>.txt，再缺省 class_0..N）\n"
        "  --save-dir DIR     结果输出目录，默认 output/（项目目录下）\n"
        "  --no-save          不保存结果（只推理打印耗时）\n"
        "  --no-draw         只推理不画框（默认画图）\n"
        "  --no-class         标签不显示类别名（默认显示）\n"
        "  --draw-conf        标签显示置信度（默认关）\n"
        "  --draw-id          标签显示跟踪 id（默认关；仅 --track 时有值）\n"
        "  --show             视频弹窗预览（默认 800x500 可拖拽缩放；按 q/ESC 退出）\n"
        "  --export-frames    逐帧导出 图片+LabelMe json 到 <save-dir>/<视频名>/（便于标注调整误检）\n"
        "  --cuda [--gpu N]   使用 GPU（需 CUDA 版 onnxruntime）\n"
        "  -h, --help         显示本帮助\n"
        "\n"
        "Examples:\n"
        "  " << prog << " model.onnx ./imgs det --crop\n"
        "  " << prog << " model.onnx video.mp4 obb --show\n"
        "  " << prog << " model.onnx video.mp4 det --track\n"
        "  " << prog << " model.onnx ./imgs det --track\n";
}

static bool parseArgs(int argc, char** argv, Options& opt) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << std::endl; exit(2); }
            return argv[++i];
        };
        try {
            if      (a == "--crop")         opt.crop = true;
            else if (a == "--crop-size") {
                // 支持 "1280"（正方形）或 "800x640"（宽x高）
                std::string v = need("--crop-size");
                size_t pos = v.find('x');
                if (pos == std::string::npos) pos = v.find('X');
                if (pos != std::string::npos)
                    opt.cropSize = cv::Size(std::stoi(v.substr(0, pos)), std::stoi(v.substr(pos + 1)));
                else
                    opt.cropSize = cv::Size(std::stoi(v), std::stoi(v));
            }
            else if (a == "--crop-overlap") {   // N 或 WxH（列x行分轴重叠）
                std::string v = need("--crop-overlap");
                std::string::size_type x = v.find('x');
                if (x != std::string::npos) {
                    opt.cropOverlapW = std::stoi(v.substr(0, x));
                    opt.cropOverlapH = std::stoi(v.substr(x + 1));
                } else {
                    opt.cropOverlapW = opt.cropOverlapH = std::stoi(v);
                }
            }
            else if (a == "--threads")      opt.threads = std::stoi(need("--threads"));
            else if (a == "--conf")         opt.conf = std::stof(need("--conf"));
            else if (a == "--track")        opt.track = true;
            else if (a == "--labels")       opt.labels = need("--labels");
            else if (a == "--save-dir")     opt.savePath = need("--save-dir");
            else if (a == "--no-save")      opt.save = false;
            else if (a == "--show")         opt.show = true;
            else if (a == "--no-draw")      opt.draw = false;
            else if (a == "--no-class")     opt.drawClass = false;
            else if (a == "--draw-conf")    opt.drawConf = true;
            else if (a == "--draw-id")      opt.drawId = true;
            else if (a == "--export-frames") opt.exportFrames = true;
            else if (a == "--cuda")         opt.isCuda = true;
            else if (a == "--gpu")          opt.cudaID = std::stoi(need("--gpu"));
            else                            pos.push_back(a);
        } catch (const std::exception&) {
            std::cerr << "invalid value for " << a << std::endl;
            return false;
        }
    }
    if (pos.size() != 3) {
        std::cerr << "expected 3 positional args: <model> <input> <task>" << std::endl;
        return false;
    }
    opt.modelPath = pos[0];
    opt.input     = pos[1];
    opt.task      = pos[2];
    if (opt.task != "det" && opt.task != "obb" && opt.task != "pose" &&
        opt.task != "seg" && opt.task != "cls") {
        std::cerr << "unknown task '" << opt.task << "' (expect det|obb|pose|seg|cls)" << std::endl;
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// 入口
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    Options opt;
    if (!parseArgs(argc, argv, opt)) { printUsage(argv[0]); return 1; }
    if (opt.track && opt.task == "cls") {   // cls 无检测框，跟踪无意义
        std::cerr << "[warn] task 'cls' has no boxes, tracking disabled" << std::endl;
        opt.track = false;
    }

    YolovOnnx yolo;
    yolo.labelFile = opt.labels;   // 空 = 用 <模型名>.txt
    if (!yolo.ReadModel(opt.modelPath, opt.task, opt.isCuda, opt.cudaID)) {
        std::cout << "cannot load model: " << opt.modelPath << std::endl;
        return 1;
    }
    yolo._classThreshold = opt.conf;   // 置信度贯通：模型 NMS / 跨块合并 / 画框一致
    yolo.drawClass = opt.drawClass;    // 标签内容开关
    yolo.drawConf  = opt.drawConf;
    yolo.drawId    = opt.drawId;

    // 输入类型自动判断：本地视频 / 图片（文件夹）
    InputType type = detectInputType(opt.input);
    switch (type) {
        case InputType::VideoFile:  printInfo(opt, "video-file"); return runVideoMode(yolo, opt);
        case InputType::Image:      printInfo(opt, "image");      return runImageMode(yolo, opt);
    }
    return 0;
}
