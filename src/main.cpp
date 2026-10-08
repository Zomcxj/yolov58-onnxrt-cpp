// ============================================================================
// yolov-onnxrt —— YOLO ONNX Runtime 推理工具（det/obb/pose/seg/cls）
//
// 用法（前三个位置参数与旧版一致）：
//   yolov-onnxrt <model.onnx> <input> <task> [options]
//
//   <model.onnx>  模型路径
//   <input>       输入来源，按内容自动判断：
//                   rtsp:// 或 http(s)://  -> IP 视频流接入（逐帧推理，默认存帧图）
//                   *.mp4/*.avi/*.mov 等   -> 本地视频文件（逐帧推理，默认存结果视频）
//                   其余（图片/目录/glob）  -> 图片写入推理（逐帧 imwrite 存结果图）
//   <task>        det | obb | pose | seg | cls
//
// 常用示例：
//   # 图片遍历 + 原图推理（旧版默认行为）
//   yolov-onnxrt model.onnx ./imgs det
//   # 图片遍历 + 裁剪推理（切块 -> 多线程并行 -> 跨块合并）
//   yolov-onnxrt model.onnx ./imgs obb --crop --crop-size 1280 --threads 4
//   # 本地视频文件推理（结果存带框视频）
//   yolov-onnxrt model.onnx video.mp4 det
//   # IP 视频流接入（RTSP，逐帧推理存帧图）
//   yolov-onnxrt model.onnx rtsp://192.168.1.64:554/stream obb --crop
//
// 常用参数：
//   --crop              启用裁剪推理（仅 det/obb 生效，其余自动回退原图）
//   --crop-size N       切块边长，默认 1280
//   --crop-overlap N    相邻切块重叠像素：默认自动(20%且最少200px)，0=无重叠
//   --threads N         裁剪并行线程数，默认 = CPU 核数
//   --conf F            跨块合并置信度阈值，默认 0.5
//   --save-dir DIR      结果输出目录，默认 ../output/
//   --no-save           不保存结果（只推理打印耗时；IP 流不存帧图、视频不存结果视频）
//   --show              IP 流/视频模式弹窗预览（按 q/ESC 退出）
//   --cuda [--gpu N]    使用 GPU（需 CUDA 版 onnxruntime）
//   -h, --help          显示本帮助
//
// 代码组织：本文件直接调用旧接口（YolovOnnx::OnnxDetect / obb.h 的
//   imgSplit + obboxFilter/hbboxFilter + tileDetectMultiThread），无中间抽象层。
// ============================================================================
#include "obb.h"
#include "yolo_onnx.h"

#include <chrono>
#include <cctype>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// 参数配置（旧版散落的配置集中到这里，便于修改）
// ----------------------------------------------------------------------------
struct Options {
    std::string modelPath;   // argv[1]
    std::string input;       // argv[2]：图片/目录 / 视频文件 / RTSP 流
    std::string task;        // argv[3]：det|obb|pose|seg|cls
    bool   crop = false;     // 裁剪推理（仅 det/obb）
    int    cropSize = 1280;  // 切块边长
    int    cropOverlap = -1; // 切块重叠像素；-1 = 自动(20%且最少200px)，0 = 无重叠
    float  conf = 0.5f;      // 跨块合并置信度阈值
    int    threads = 0;      // 裁剪并行线程数；0 = 自动
    std::string savePath = "../output/";
    bool   save = true;      // --no-save 关闭保存
    bool   show = false;     // 视频预览
    bool   isCuda = false;
    int    cudaID = 0;
};

// 输入类型自动判断
enum class InputType { Image, VideoFile, IpStream };

// ----------------------------------------------------------------------------
// 工具函数
// ----------------------------------------------------------------------------
static std::string toLower(std::string s) {
    for (size_t i = 0; i < s.size(); ++i)
        s[i] = (char)std::tolower((unsigned char)s[i]);
    return s;
}

static bool startsWith(const std::string& s, const char* prefix) {
    return s.compare(0, std::strlen(prefix), prefix) == 0;
}

static InputType detectInputType(const std::string& input) {
    if (startsWith(input, "rtsp://") || startsWith(input, "http://") || startsWith(input, "https://"))
        return InputType::IpStream;
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

// ----------------------------------------------------------------------------
// 统一推理入口：原图推理 / 裁剪推理（多线程），按配置 if 判断
// 结果 result 已映射回原图坐标
// ----------------------------------------------------------------------------
static void detectOne(YolovOnnx& yolo, cv::Mat& img, const Options& opt,
                      std::vector<YolovOutput>& result) {
    if (opt.crop && (opt.task == "det" || opt.task == "obb"))
        result = tileDetectMultiThread(yolo, img, opt.task, opt.cropSize,
                                       opt.cropOverlap, opt.conf, opt.threads);
    else
        yolo.OnnxDetect(img, result);
}

static void printInfo(const Options& opt, const char* mode) {
    std::cout << "model: " << opt.modelPath
              << " | task: " << opt.task
              << " | mode: " << mode
              << " | crop: " << (opt.crop ? ("on(size=" + std::to_string(opt.cropSize) +
                                             (opt.cropOverlap >= 0 ? ",ov=" + std::to_string(opt.cropOverlap) : "") +
                                             ",threads=" + std::to_string(opt.threads) + ")")
                                          : "off")
              << std::endl;
}

// ----------------------------------------------------------------------------
// 模式一：图片写入推理（默认，旧版行为）
// ----------------------------------------------------------------------------
static int runImageMode(YolovOnnx& yolo, const Options& opt) {
    std::vector<std::string> files;
    cv::glob(opt.input, files, false);
    if (files.empty()) files.push_back(opt.input);   // glob 不匹配时按单文件处理

    std::string savePath = withTrailingSep(opt.savePath);
    int nFail = 0;
    for (size_t k = 0; k < files.size(); ++k) {
        const std::string& f = files[k];
        cv::Mat img = cv::imread(f);
        if (img.empty()) { std::cerr << "[warn] cannot read image: " << f << std::endl; ++nFail; continue; }

        Clock::time_point t0 = Clock::now();
        std::vector<YolovOutput> result;
        detectOne(yolo, img, opt, result);
        double ms = elapsedMs(t0);

        if (opt.task != "cls") {              // cls 无框可画，与原行为一致
            yolo.DrawPred(img, result);
            if (opt.save) cv::imwrite(savePath + pathBasename(f), img);
        }
        std::cout << pathBasename(f) << "  [" << result.size() << " objs]  "
                  << (int)(ms + 0.5) << " ms" << std::endl;
    }
    std::cout << "done. " << files.size() << " image(s), " << nFail << " failed." << std::endl;
    return (nFail > 0 && nFail == (int)files.size()) ? 1 : 0;
}

// ----------------------------------------------------------------------------
// 模式二/三：视频流（IP 流存帧图 / 本地视频文件存结果视频）
// ----------------------------------------------------------------------------
static int runVideoMode(YolovOnnx& yolo, const Options& opt, InputType type) {
    cv::VideoCapture cap;
    if (type == InputType::IpStream) cap.open(opt.input);   // rtsp://...
    else                            cap.open(opt.input);   // 本地视频文件
    if (!cap.isOpened()) { std::cerr << "cannot open video: " << opt.input << std::endl; return 1; }

    std::string savePath = withTrailingSep(opt.savePath);

    // 输出：IP 流 -> 存帧图；本地视频 -> 存结果视频
    cv::VideoWriter writer;
    std::string videoOutPath;
    bool saveImage = false;      // IP 流逐帧存图
    bool saveVideo = false;      // 本地视频写结果文件
    if (opt.save && type == InputType::IpStream) {
        saveImage = true;
        std::cout << "saving frames to " << savePath << "stream_*.jpg" << std::endl;
    } else if (opt.save && type == InputType::VideoFile) {
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
    while (cap.read(frame)) {
        Clock::time_point t0 = Clock::now();
        std::vector<YolovOutput> result;
        detectOne(yolo, frame, opt, result);
        double ms = elapsedMs(t0);

        if (opt.task != "cls") yolo.DrawPred(frame, result);

        if (saveImage) cv::imwrite(savePath + "stream_" + std::to_string(frameIdx) + ".jpg", frame);
        if (saveVideo) writer.write(frame);
        if (opt.show) {
            cv::imshow("yolov-onnxrt", frame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
        }
        totalMs += ms;
        ++frameIdx;
        std::cout << "frame " << frameIdx << "  [" << result.size() << " objs]  "
                  << (int)(ms + 0.5) << " ms" << std::endl;
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
        "  <input>  自动判断：rtsp:// 或 http(s):// -> IP 视频流(存帧图)\n"
        "                        *.mp4/*.avi 等     -> 本地视频文件(存结果视频)\n"
        "                        其余(图片/目录/glob)-> 图片写入推理(存结果图)\n"
        "  <task>   det | obb | pose | seg | cls\n"
        "\n"
        "Options:\n"
        "  --crop             启用裁剪（切块）推理，仅 det/obb 生效\n"
        "  --crop-size N      切块边长，默认 1280\n"
        "  --crop-overlap N   切块重叠像素：默认自动(20%且最少200px)，0=无重叠，>0=指定\n"
        "  --threads N        裁剪并行线程数，默认 = CPU 核数\n"
        "  --conf F           跨块合并置信度阈值，默认 0.5\n"
        "  --save-dir DIR     结果输出目录，默认 ../output/\n"
        "  --no-save          不保存结果（IP 流不存帧图、视频不存结果视频）\n"
        "  --show             IP 流/视频模式弹窗预览（按 q/ESC 退出）\n"
        "  --cuda [--gpu N]   使用 GPU（需 CUDA 版 onnxruntime）\n"
        "  -h, --help         显示本帮助\n"
        "\n"
        "Examples:\n"
        "  " << prog << " model.onnx ./imgs det --crop\n"
        "  " << prog << " model.onnx video.mp4 obb --show\n"
        "  " << prog << " model.onnx rtsp://192.168.1.64/stream det\n";
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
            else if (a == "--crop-size")    opt.cropSize = std::stoi(need("--crop-size"));
            else if (a == "--crop-overlap") opt.cropOverlap = std::stoi(need("--crop-overlap"));
            else if (a == "--threads")      opt.threads = std::stoi(need("--threads"));
            else if (a == "--conf")         opt.conf = std::stof(need("--conf"));
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

    YolovOnnx yolo;
    if (!yolo.ReadModel(opt.modelPath, opt.task, opt.isCuda, opt.cudaID, true)) {
        std::cout << "cannot load model: " << opt.modelPath << std::endl;
        return 1;
    }

    // 输入类型自动判断：IP 流 / 本地视频 / 图片
    InputType type = detectInputType(opt.input);
    switch (type) {
        case InputType::IpStream:   printInfo(opt, "ip-stream"); return runVideoMode(yolo, opt, type);
        case InputType::VideoFile:  printInfo(opt, "video-file"); return runVideoMode(yolo, opt, type);
        case InputType::Image:      printInfo(opt, "image"); return runImageMode(yolo, opt);
    }
    return 0;
}
