#pragma once
// Qt 图形界面（纯 Qt 进程，无 OpenCV 依赖）：
//   界面收集参数 -> QProcess 启动同目录 yolov58-onnxrt-cpp.exe（命令行与 main.cpp 完全一致）
//   日志/进度来自子进程 stdout；图片模式逐张预览保存的结果图。
// 本进程必须用 Qt 自带 mingw730_64 编译（与 Qt5.12 mingw73_64 运行时同 ABI），
// 不可用项目主编译器（mingw81-sjlj，与 Qt seh 运行时混用会随机加载失败 0xC0000139）。
#include <QMainWindow>
#include <QProcess>
#include <QImage>
#include <QString>
#include <QStringList>
#include <string>

// 界面收集的运行参数（对应 CLI 选项）
struct GuiParams {
    QString modelPath;            // 模型 onnx
    QString labelsPath;           // 类别 txt（空 = 模型同名 txt）
    QString input;                // 图片 / 文件夹 / 视频文件
    QString task = "det";         // det | obb | pose | seg | cls
    double  conf = 0.5;           // 置信度阈值
    double  nms = 0.6;            // NMS IoU 阈值
    bool draw = true;             // 画框
    bool drawClass = true;        // 标签：类别名
    bool drawConf = false;        // 标签：置信度
    bool drawId = false;          // 标签：跟踪 id
    bool track = false;           // 目标跟踪
    int  minTrackFrames = 0;      // 轨迹去抖（0=关）
    bool save = true;             // 保存结果
    QString saveDir = "output";   // 保存目录（子进程工作目录下，自动创建）
    bool preview = true;          // 图片模式：逐张预览结果图
    bool exportJson = false;      // 视频逐帧导出 原图+LabelMe json（同目录，便于标注调整）
    bool crop = false;            // 滑窗切块（仅 det/obb）
    int cropW = 480, cropH = 480; // 切块 宽x高
    int ovW = 256, ovH = 96;      // 重叠 宽x高
    int threads = 4;              // 切块并行线程（0 = CPU 核数）
    int frameStep = 1;            // 视频抽帧：每 N 帧推理一次（1 = 每帧）
    bool isCuda = false;          // GPU（CPU 包自动回退）
    int cudaID = 0;
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void onStart();
    void onStop();
    void onPickModel();
    void onPickLabels();
    void onPickImage();
    void onPickVideo();
    void onPickFolder();
    void onPickSaveDir();
    void onProcStdout();
    void onProcStderr();
    void onProcFinished(int code, QProcess::ExitStatus st);
    void onProcError(QProcess::ProcessError err);

private:
    GuiParams collect();
    QStringList buildArgs(const GuiParams& p) const;
    QString resolveCli() const;               // 定位 yolov58-onnxrt-cpp.exe
    QString absSaveDir(const GuiParams& p) const;
    void previewInput();                      // 选中输入后立刻在左栏预览（不等推理）
    void setBusy(bool busy);
    void log(const QString& s);               // 日志统一入口：北京时间前缀（UTC+8，毫秒）
    QString stamp() const;
    void showImages();
    void overlayTiles(QImage& im) const;      // 原图叠画滑窗切块网格（参数调整即时预览）                        // 双栏预览按窗口尺寸自适应缩放
    void resizeEvent(QResizeEvent* e) override;
    void playVideo(class QMediaPlayer*& pl, class QVideoWidget* vw,
                   class QStackedWidget* st, const QString& file);   // 切到视频页并循环播放

    // 模型/任务
    class QLineEdit* m_model = nullptr;
    class QLineEdit* m_labels = nullptr;
    class QComboBox* m_task = nullptr;
    // 推理参数
    class QDoubleSpinBox* m_conf = nullptr;
    QDoubleSpinBox* m_nms = nullptr;
    class QSpinBox* m_minTrackFrames = nullptr;
    class QCheckBox* m_draw = nullptr;
    class QCheckBox* m_drawClass = nullptr;
    class QCheckBox* m_drawConf = nullptr;
    class QCheckBox* m_drawId = nullptr;
    class QCheckBox* m_track = nullptr;
    class QCheckBox* m_crop = nullptr;
    class QSpinBox* m_cropW = nullptr;
    class QSpinBox* m_cropH = nullptr;
    class QSpinBox* m_ovW = nullptr;
    class QSpinBox* m_ovH = nullptr;
    class QSpinBox* m_threads = nullptr;
    class QSpinBox* m_frameStep = nullptr;
    class QCheckBox* m_save = nullptr;
    class QLineEdit* m_saveDir = nullptr;
    class QCheckBox* m_preview = nullptr;
    class QCheckBox* m_exportJson = nullptr;
    class QCheckBox* m_cuda = nullptr;
    class QSpinBox* m_cudaId = nullptr;
    // 输入/控制
    class QLineEdit* m_input = nullptr;
    class QPushButton* m_start = nullptr;
    class QPushButton* m_stop = nullptr;
    // 展示（左：原图/原视频；右：结果图/结果视频）
    class QSplitter* m_split = nullptr;
    class QStackedWidget* m_stackL = nullptr;
    class QStackedWidget* m_stackR = nullptr;
    class QLabel* m_origInfo = nullptr;      // 左栏下方：原图尺寸标注
    class QLabel* m_viewOrig = nullptr;       // 页0：图片
    class QLabel* m_viewRes = nullptr;
    class QVideoWidget* m_vwR = nullptr;      // 页1：视频（右栏结果视频）
    class QMediaPlayer* m_plR = nullptr;      // 懒创建（仅右栏结果视频用）
    class QPlainTextEdit* m_log = nullptr;
    class QProgressBar* m_bar = nullptr;
    QImage m_lastOrig, m_lastRes;             // 左右栏当前原图（缩放前）
    bool m_streamSeen = false;                // 视频实时帧是否已连上（诊断用）
    // 子进程
    class QProcess* m_proc = nullptr;
    bool m_busy = false;
    bool m_videoMode = false;
};
