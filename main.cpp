// yolov-onnxrt：YOLO ONNX Runtime 推理工具（det/obb/pose/seg/cls）
// 用法：yolov-onnxrt <model.onnx> <input> <task> [options]，详见 -h
// <input> 按内容自动判断：视频文件 | 图片/目录/glob
// 类别名：模型旁 <模型名>.txt（每行一个，可选，缺省 class_0..N）
#include "obb.h"
#include "yolo_onnx.h"
#include "MYTrack.h"   // ByteTrack 封装（检测 -> 跟踪）

#include <chrono>
#include <cctype>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// 项目对接（本区之外的全部是本地测试代码）
//
//   YolovOnnx yolo;
//   yolo.ReadModel(modelPath, task);        // 启动时一次（GPU: ReadModel(modelPath, task, true)）
//   OnnxDetectv2(yolo, img, task, result);  // 每张图，result 已映射回原图坐标
//   yolo.DrawPred(img, result);             // 可选：画框
//
//   跟踪（可选）：直接用 MYTrack（ByteTrack 封装）
//   MYTrack tracker;
//   std::vector<Object> objs, tracks;
//   tracker.resultStructTransfer(result, objs);
//   tracker.doTrack(objs, tracks);
// ============================================================================
static void OnnxDetectv2(YolovOnnx& yolo, cv::Mat& img, const std::string& task,
                         std::vector<YolovOutput>& result,
                         bool crop = false, const cv::Size& cropSize = cv::Size(1280, 1280),
                         int cropOverlap = -1, float conf = 0.5f, int threads = 0) {
    if (crop && (task == "det" || task == "obb"))
        result = tileDetectMultiThread(yolo, img, task, cropSize, cropOverlap, conf, threads);
    else
        yolo.OnnxDetect(img, result);
}

// ============================================================================
// 以下全部是本地测试代码（Options/解析/文件夹遍历/视频/跟踪），对接不需要
// ============================================================================

// 参数配置
struct Options {
    std::string modelPath;   // argv[1]
    std::string input;       // argv[2]：图片/目录 / 视频文件 / RTSP 流
    std::string task;        // argv[3]：det|obb|pose|seg|cls
    bool   crop = false;     // 裁剪推理（仅 det/obb）
    cv::Size cropSize{1280, 1280}; // 切块尺寸（宽×高）
    int    cropOverlap = -1; // 切块重叠像素；-1 = 自动(20%且最少200px)，0 = 无重叠
    float  conf = 0.5f;      // 跨块合并置信度阈值
    int    threads = 0;      // 裁剪并行线程数；0 = 自动
    bool   track = false;    // --track：检测后接 ByteTrack 跟踪（cls 无框，已在 main 里关闭）
    std::string savePath = "../output/";
    bool   save = true;      // --no-save 关闭保存
    bool   show = false;     // 视频预览
    bool   isCuda = false;
    int    cudaID = 0;
};

// 测试便捷封装：从 Options 取参数调对接接口
static void OnnxDetectv2(YolovOnnx& yolo, cv::Mat& img, const Options& opt,
                      std::vector<YolovOutput>& result) {
    OnnxDetectv2(yolo, img, opt.task, result, opt.crop, opt.cropSize,
                 opt.cropOverlap, opt.conf, opt.threads);
}

// 输入类型自动判断
enum class InputType { Image, VideoFile };

// ----------------------------------------------------------------------------
// 工具函数
// ----------------------------------------------------------------------------
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

// 把跟踪 id 回写到对应检测框（DrawPred 标签即显示 id:N）
// 匹配依据：Object.rect 就来自对应检测框的 box，坐标一致
static void tagTrackIds(const std::vector<Object>& tracks, std::vector<YolovOutput>& result) {
    for (size_t i = 0; i < tracks.size(); ++i) {
        const Object& t = tracks[i];
        for (size_t j = 0; j < result.size(); ++j) {
            YolovOutput& d = result[j];
            if (d.track_id != -1) continue;
            if (d.box.x == (int)t.rect.x && d.box.y == (int)t.rect.y &&
                d.box.width == (int)t.rect.width && d.box.height == (int)t.rect.height) {
                d.track_id = t.track_id;
                break;
            }
        }
    }
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
                                             (opt.cropOverlap >= 0 ? ",ov=" + std::to_string(opt.cropOverlap) : "") +
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
    MYTrack tracker;                      // --track：跨图片保持轨迹
    std::vector<Object> tracks;
    int nFail = 0;
    for (size_t k = 0; k < files.size(); ++k) {
        const std::string& f = files[k];
        cv::Mat img = cv::imread(f);
        if (img.empty()) { std::cerr << "[warn] cannot read image: " << f << std::endl; ++nFail; continue; }

        Clock::time_point t0 = Clock::now();
        std::vector<YolovOutput> result;
        OnnxDetectv2(yolo, img, opt, result);
        double ms = elapsedMs(t0);

        if (opt.track) {                       // 检测 -> 跟踪（直接用 MYTrack）
            std::vector<Object> objs;
            tracker.resultStructTransfer(result, objs);
            tracks.clear();
            tracker.doTrack(objs, tracks);
        }

        if (opt.task != "cls") {              // cls 无框可画，与原行为一致
            if (opt.track) tagTrackIds(tracks, result);
            yolo.DrawPred(img, result);
            if (opt.save) cv::imwrite(savePath + pathBasename(f), img);
        }
        std::cout << pathBasename(f) << "  [" << result.size() << " objs";
        if (opt.track) std::cout << ", " << tracks.size() << " tracks";
        std::cout << "]  " << (int)(ms + 0.5) << " ms" << std::endl;
        if (opt.track) printTracks(tracks);
    }
    std::cout << "done. " << files.size() << " image(s), " << nFail << " failed." << std::endl;
    return (nFail > 0 && nFail == (int)files.size()) ? 1 : 0;
}

// ----------------------------------------------------------------------------
// 测试：本地视频文件推理（存结果视频）
// ----------------------------------------------------------------------------
static int runVideoMode(YolovOnnx& yolo, const Options& opt) {
    cv::VideoCapture cap;
    cap.open(opt.input);
    if (!cap.isOpened()) { std::cerr << "cannot open video: " << opt.input << std::endl; return 1; }

    std::string savePath = withTrailingSep(opt.savePath);

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
    while (cap.read(frame)) {
        Clock::time_point t0 = Clock::now();
        std::vector<YolovOutput> result;
        OnnxDetectv2(yolo, frame, opt, result);
        double ms = elapsedMs(t0);

        if (opt.track) {                       // 检测 -> 跟踪（直接用 MYTrack）
            std::vector<Object> objs;
            tracker.resultStructTransfer(result, objs);
            tracks.clear();
            tracker.doTrack(objs, tracks);
        }

        if (opt.task != "cls") {
            if (opt.track) tagTrackIds(tracks, result);
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
              << (frameIdx ? (int)(totalMs / frameIdx + 0.5) : 0) << " ms/frame";
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
        "  --crop-overlap N   切块重叠像素：默认自动(20%且最少200px)，0=无重叠，>0=指定\n"
        "  --threads N        裁剪并行线程数，默认 = CPU 核数\n"
        "  --conf F           跨块合并置信度阈值，默认 0.5\n"
        "  --track            检测后接 ByteTrack 跟踪（打印 track id 并画框；cls 忽略）\n"
        "  类别名              模型旁 <模型名>.txt，每行一个类别（可选，缺省 class_0..N）\n"
        "  --save-dir DIR     结果输出目录，默认 ../output/\n"
        "  --no-save          不保存结果（只推理打印耗时）\n"
        "  --show             视频模式弹窗预览（按 q/ESC 退出）\n"
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
            else if (a == "--crop-overlap") opt.cropOverlap = std::stoi(need("--crop-overlap"));
            else if (a == "--threads")      opt.threads = std::stoi(need("--threads"));
            else if (a == "--conf")         opt.conf = std::stof(need("--conf"));
            else if (a == "--track")        opt.track = true;
            else if (a == "--save-dir")     opt.savePath = need("--save-dir");
            else if (a == "--no-save")      opt.save = false;
            else if (a == "--show")         opt.show = true;
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
    if (!yolo.ReadModel(opt.modelPath, opt.task, opt.isCuda, opt.cudaID, true)) {
        std::cout << "cannot load model: " << opt.modelPath << std::endl;
        return 1;
    }

    // 输入类型自动判断：本地视频 / 图片（文件夹）
    InputType type = detectInputType(opt.input);
    switch (type) {
        case InputType::VideoFile:  printInfo(opt, "video-file"); return runVideoMode(yolo, opt);
        case InputType::Image:      printInfo(opt, "image");      return runImageMode(yolo, opt);
    }
    return 0;
}
