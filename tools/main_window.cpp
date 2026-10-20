// Qt 图形界面实现（QProcess 架构）：窗口/参数控件 + 子进程编排 + 结果图预览
#include "main_window.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImageReader>
#include <QPainter>
#include <QLabel>
#include <QLineEdit>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QVideoWidget>
#include <QMediaPlayer>

#ifdef Q_OS_WIN
#include <windows.h>    // CREATE_NO_WINDOW（隐藏子进程控制台窗口）
#endif

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("YOLO Infer GUI");
    resize(1280, 860);

    auto* central = new QWidget(this);
    auto* root = new QVBoxLayout(central);

    // ---------- 模型 / 任务 ----------
    auto* gbModel = new QGroupBox("模型", central);
    auto* gModel = new QGridLayout(gbModel);
    m_model = new QLineEdit(gbModel);        m_model->setPlaceholderText("onnx 模型路径");
    m_labels = new QLineEdit(gbModel);       m_labels->setPlaceholderText("留空 = 模型同名 txt");
    m_task = new QComboBox(gbModel);
    m_task->addItems({"det", "obb", "pose", "seg", "cls"});
    auto* bModel = new QPushButton("浏览...", gbModel);
    auto* bLabels = new QPushButton("浏览...", gbModel);
    gModel->addWidget(new QLabel("模型:", gbModel), 0, 0);
    gModel->addWidget(m_model, 0, 1);        gModel->addWidget(bModel, 0, 2);
    gModel->addWidget(new QLabel("类别txt:", gbModel), 1, 0);
    gModel->addWidget(m_labels, 1, 1);       gModel->addWidget(bLabels, 1, 2);
    gModel->addWidget(new QLabel("任务:", gbModel), 0, 3);
    gModel->addWidget(m_task, 0, 4);
    root->addWidget(gbModel);

    // ---------- 推理参数 ----------
    auto* gbInf = new QGroupBox("推理参数", central);
    auto* gInf = new QGridLayout(gbInf);
    m_conf = new QDoubleSpinBox(gbInf);
    m_conf->setRange(0.05, 0.95); m_conf->setSingleStep(0.05); m_conf->setValue(0.5);
    m_conf->setToolTip("置信度阈值：模型 NMS + 跨块合并 + 画框 三处贯通");
    m_nms = new QDoubleSpinBox(gbInf);
    m_nms->setRange(0.1, 0.95); m_nms->setSingleStep(0.05); m_nms->setValue(0.6);
    m_nms->setToolTip("NMS IoU 阈值：越小抑制越狠，保留框越少");
    m_draw      = new QCheckBox("画框", gbInf);       m_draw->setChecked(true);
    m_drawClass = new QCheckBox("类别名", gbInf);     m_drawClass->setChecked(true);
    m_drawConf  = new QCheckBox("置信度", gbInf);     m_drawConf->setToolTip("框上显示置信度");
    m_drawId    = new QCheckBox("跟踪id", gbInf);     m_drawId->setToolTip("仅跟踪时有值");
    m_drawId->setEnabled(false);                      // 依赖目标跟踪
    m_track     = new QCheckBox("目标跟踪", gbInf);
    connect(m_track, &QCheckBox::toggled, m_drawId, &QCheckBox::setEnabled);
    m_minTrackFrames = new QSpinBox(gbInf); m_minTrackFrames->setRange(0, 100); m_minTrackFrames->setValue(0);
    m_minTrackFrames->setToolTip("轨迹去抖：出现不足 N 帧不画不打，0=关");
    m_minTrackFrames->setEnabled(false);              // 依赖目标跟踪
    connect(m_track, &QCheckBox::toggled, m_minTrackFrames, &QSpinBox::setEnabled);
    m_crop = new QCheckBox("滑窗切块", gbInf);
    m_cropW = new QSpinBox(gbInf);  m_cropW->setRange(64, 4096);  m_cropW->setSingleStep(32); m_cropW->setValue(640);
    m_cropH = new QSpinBox(gbInf);  m_cropH->setRange(64, 4096);  m_cropH->setSingleStep(32); m_cropH->setValue(640);
    m_ovW   = new QSpinBox(gbInf);  m_ovW->setRange(0, 2048);     m_ovW->setValue(160);
    m_ovH   = new QSpinBox(gbInf);  m_ovH->setRange(0, 2048);     m_ovH->setValue(160);
    m_threads = new QSpinBox(gbInf); m_threads->setRange(0, 64);  m_threads->setValue(4);
    m_threads->setToolTip("0 = CPU 核数");
    m_frameStep = new QSpinBox(gbInf); m_frameStep->setRange(1, 60); m_frameStep->setValue(1);
    m_frameStep->setToolTip("视频抽帧：每 N 帧推理一次，其余帧沿用上次结果；1=每帧");
    m_save = new QCheckBox("保存结果", gbInf); m_save->setChecked(true);
    m_saveDir = new QLineEdit(QDir::toNativeSeparators(QCoreApplication::applicationDirPath() + "/output"), gbInf);
    m_saveDir->setToolTip("结果输出目录（绝对路径）；默认为程序所在目录下的 output");
    auto* bDir = new QPushButton("目录...", gbInf);
    m_preview = new QCheckBox("图片预览", gbInf); m_preview->setChecked(true);
    m_preview->setToolTip("图片模式逐张显示结果图；视频仅日志/进度");
    m_cuda = new QCheckBox("GPU", gbInf);
    m_cudaId = new QSpinBox(gbInf); m_cudaId->setRange(0, 7); m_cudaId->setEnabled(false);
    m_exportJson = new QCheckBox("导出json", gbInf);
    m_exportJson->setToolTip("视频逐帧导出 原图+LabelMe json（同目录，便于标注调整误检）");
    gInf->addWidget(m_exportJson, 3, 7);
    // 滑窗参数调整 → 左栏原图实时叠画切块网格（所见即所得，无需试推理）
    auto tileSync = [this] { previewInput(); };
    connect(m_crop,  &QCheckBox::toggled, this, tileSync);
    connect(m_cropW, qOverload<int>(&QSpinBox::valueChanged), this, tileSync);
    connect(m_cropH, qOverload<int>(&QSpinBox::valueChanged), this, tileSync);
    connect(m_ovW,   qOverload<int>(&QSpinBox::valueChanged), this, tileSync);
    connect(m_ovH,   qOverload<int>(&QSpinBox::valueChanged), this, tileSync);
    m_origInfo = new QLabel(central);                     // 原图尺寸（显示在线程右侧）
    m_origInfo->setStyleSheet("color:#8a919c; background:transparent;");
    // 参数行：水平流式聚簇（相关参数紧挨，不再按列拉开）
    auto* row0 = new QHBoxLayout;
    row0->addWidget(new QLabel("置信度:", gbInf));
    row0->addWidget(m_conf);
    row0->addSpacing(14);
    row0->addWidget(new QLabel("NMS:", gbInf));
    row0->addWidget(m_nms);
    row0->addSpacing(14);
    row0->addWidget(m_draw);      row0->addWidget(m_drawClass);
    row0->addWidget(m_drawConf);  row0->addWidget(m_drawId);
    row0->addWidget(m_track);
    row0->addWidget(new QLabel("去抖N帧:", gbInf));
    row0->addWidget(m_minTrackFrames);
    row0->addStretch(1);
    gInf->addLayout(row0, 0, 0, 1, 2);

    auto* row1 = new QHBoxLayout;
    row1->addWidget(m_crop);
    row1->addWidget(new QLabel("块宽:", gbInf));   row1->addWidget(m_cropW);
    row1->addWidget(new QLabel("块高:", gbInf));   row1->addWidget(m_cropH);
    row1->addSpacing(14);
    row1->addWidget(new QLabel("重叠宽:", gbInf)); row1->addWidget(m_ovW);
    row1->addWidget(new QLabel("重叠高:", gbInf)); row1->addWidget(m_ovH);
    row1->addSpacing(14);
    row1->addWidget(new QLabel("抽帧:", gbInf));   row1->addWidget(m_frameStep);
    row1->addSpacing(14);
    row1->addWidget(new QLabel("线程:", gbInf));   row1->addWidget(m_threads);
    row1->addSpacing(14);
    row1->addWidget(m_origInfo);                  // 原图: 1920 × 1080
    row1->addStretch(1);
    gInf->addLayout(row1, 1, 0, 1, 2);

    auto* row2 = new QHBoxLayout;
    row2->addWidget(m_save);
    row2->addWidget(new QLabel("目录:", gbInf));
    row2->addWidget(m_saveDir, 1);
    row2->addWidget(bDir);
    row2->addSpacing(14);
    row2->addWidget(m_preview);
    row2->addWidget(m_cuda);      row2->addWidget(m_cudaId);
    row2->addWidget(m_exportJson);
    gInf->addLayout(row2, 2, 0, 1, 2);
    // 输入框统一宽度（块/重叠/线程/抽帧/GPU编号）
    for (QSpinBox* s : {m_cropW, m_cropH, m_ovW, m_ovH, m_threads, m_frameStep, m_cudaId, m_minTrackFrames})
        s->setFixedWidth(72);
    for (QDoubleSpinBox* s : {m_conf, m_nms})
        s->setFixedWidth(90);
    root->addWidget(gbInf);

    // ---------- 输入 / 控制 ----------
    auto* hIn = new QHBoxLayout();
    m_input = new QLineEdit(central); m_input->setPlaceholderText("图片 / 文件夹 / 视频文件");
    auto* bImg = new QPushButton("选图片", central);
    auto* bVideo = new QPushButton("选视频", central);
    auto* bFold = new QPushButton("选文件夹", central);
    m_start = new QPushButton("开始推理", central);
    m_stop = new QPushButton("停止", central);
    m_stop->setEnabled(false);
    QFont bf = m_start->font(); bf.setBold(true); m_start->setFont(bf);
    m_start->setStyleSheet(
        "QPushButton { background:#3a7fc1; color:white; font-weight:bold; border:none; border-radius:6px; padding:5px 18px; }"
        "QPushButton:hover { background:#2f6cab; }"
        "QPushButton:pressed { background:#275d92; }"
        "QPushButton:disabled { background:#a9c7e8; color:#eef3f8; }");
    hIn->addWidget(new QLabel("输入:", central));
    hIn->addWidget(m_input, 1);
    hIn->addWidget(bImg); hIn->addWidget(bVideo); hIn->addWidget(bFold);
    hIn->addWidget(m_start); hIn->addWidget(m_stop);
    root->addLayout(hIn);

    m_bar = new QProgressBar(central);
    m_bar->setRange(0, 1); m_bar->setValue(0);
    root->addWidget(m_bar);

    // ---------- 预览：左=原图/原视频，右=结果图/结果视频（随窗口自适应） ----------
    m_viewOrig = new QLabel(central);
    m_viewRes  = new QLabel(central);
    for (QLabel* v : {m_viewOrig, m_viewRes}) {
        v->setAlignment(Qt::AlignCenter);
        v->setFrameShape(QFrame::Box);
        v->setStyleSheet("background:#202020; color:#909090;");
        v->setText(v == m_viewOrig ? "原图 / 原视频" : "结果图 / 结果视频");
        v->setMinimumSize(1, 1);
        v->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    }
    m_vwR = new QVideoWidget(central);   m_vwR->setStyleSheet("background:black;");
    m_stackL = new QStackedWidget(central); m_stackL->addWidget(m_viewOrig);
    m_stackR = new QStackedWidget(central); m_stackR->addWidget(m_viewRes);  m_stackR->addWidget(m_vwR);
    m_split = new QSplitter(Qt::Horizontal, central);
    m_split->addWidget(m_stackL); m_split->addWidget(m_stackR);
    m_split->setStretchFactor(0, 1); m_split->setStretchFactor(1, 1);
    m_split->setChildrenCollapsible(false);        // 拖拽不允许把任一栏折到 0
    m_split->setSizes({1, 1});                     // 初始等分（避免 sizeHint 差异导致右栏被压扁）
    root->addWidget(m_split, 1);

    // ---------- 日志 ----------
    m_log = new QPlainTextEdit(central);
    m_log->setReadOnly(true);
    m_log->setMaximumBlockCount(2000);
    m_log->setMinimumHeight(120);
    root->addWidget(m_log);

    setCentralWidget(central);

    // ---------- 圆角现代风样式 ----------
    central->setStyleSheet(
        "QGroupBox { border:1px solid #c8ccd4; border-radius:8px; margin-top:10px; background:#fafbfc; }"
        "QGroupBox::title { subcontrol-origin: margin; left:12px; padding:0 4px; color:#33475b; }"
        "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { border:1px solid #c8ccd4; border-radius:6px;"
        "  padding:2px 6px; background:white; selection-background-color:#4a90d9; }"
        "QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color:#4a90d9; }"
        "QPushButton { border:1px solid #c8ccd4; border-radius:6px; padding:4px 14px; background:white; }"
        "QPushButton:hover { background:#e8f0fe; border-color:#4a90d9; }"
        "QPushButton:pressed { background:#d7e5fa; }"
        "QPushButton:disabled { color:#9aa3ad; background:#f2f3f5; }"
        "QPlainTextEdit { border:1px solid #c8ccd4; border-radius:6px; background:white; }"
        "QProgressBar { border:1px solid #c8ccd4; border-radius:6px; background:#eef1f5; text-align:center; }"
        "QProgressBar::chunk { border-radius:5px; background:#4a90d9; }"
        "QCheckBox { spacing:4px; }"
        "QSplitter::handle { background:#dfe3e8; width:3px; }"
    );

    // ---------- 信号 ----------
    connect(bModel,  &QPushButton::clicked, this, &MainWindow::onPickModel);
    connect(bLabels, &QPushButton::clicked, this, &MainWindow::onPickLabels);
    connect(bImg,    &QPushButton::clicked, this, &MainWindow::onPickImage);
    connect(bVideo,  &QPushButton::clicked, this, &MainWindow::onPickVideo);
    connect(bFold,   &QPushButton::clicked, this, &MainWindow::onPickFolder);
    connect(bDir,    &QPushButton::clicked, this, &MainWindow::onPickSaveDir);
    connect(m_start, &QPushButton::clicked, this, &MainWindow::onStart);
    connect(m_stop,  &QPushButton::clicked, this, &MainWindow::onStop);
    connect(m_cuda,  &QCheckBox::toggled, m_cudaId, &QSpinBox::setEnabled);
    connect(m_task, &QComboBox::currentTextChanged, this, [this](const QString& t) {
        m_crop->setEnabled(t == "det" || t == "obb");   // 滑窗仅 det/obb 生效
    });

    m_proc = new QProcess(this);
    // 隐藏子进程控制台窗口
#ifdef Q_OS_WIN
    m_proc->setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* a) {
        a->flags |= CREATE_NO_WINDOW;
    });
#endif
    connect(m_proc, &QProcess::readyReadStandardOutput, this, &MainWindow::onProcStdout);
    connect(m_proc, &QProcess::readyReadStandardError,  this, &MainWindow::onProcStderr);
    connect(m_proc, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(onProcFinished(int,QProcess::ExitStatus)));
    connect(m_proc, SIGNAL(errorOccurred(QProcess::ProcessError)), this, SLOT(onProcError(QProcess::ProcessError)));
}

MainWindow::~MainWindow() {
    if (m_plR) m_plR->stop();
    if (m_proc && m_proc->state() != QProcess::NotRunning) {
        m_proc->kill();
        m_proc->waitForFinished(2000);
    }
}

QString MainWindow::resolveCli() const {
    QString appDir = QCoreApplication::applicationDirPath();
    // 部署形态：cli/ 子目录（CLI 自带 sjlj 运行时，与 GUI 的 Qt/seh 运行时隔离）
    for (const QString& c : {appDir + "/cli/yolov-onnxrt.exe", appDir + "/yolov-onnxrt.exe",
                             appDir + "/../yolov-onnxrt.exe"})
        if (QFileInfo::exists(c)) return QFileInfo(c).absoluteFilePath();
    return QString();
}

QString MainWindow::absSaveDir(const GuiParams& p) const {
    QDir d(p.saveDir);
    if (d.isAbsolute()) return QDir::toNativeSeparators(p.saveDir);
    return QDir::toNativeSeparators(QCoreApplication::applicationDirPath() + "/" + p.saveDir);
}

// 选中输入后立刻预览：图片→左栏显示；视频→左栏立即循环播放；文件夹→预览第一张
void MainWindow::previewInput() {
    const QString in = m_input->text().trimmed();
    if (in.isEmpty() || !QFileInfo::exists(in)) return;
    QFileInfo fi(in);
    if (fi.isDir()) {
        if (m_plR) { m_plR->stop(); m_stackR->setCurrentIndex(0); }
        QStringList fs = QDir(in).entryList({"*.jpg", "*.jpeg", "*.png", "*.bmp"}, QDir::Files, QDir::Name);
        m_lastOrig = fs.isEmpty() ? QImage() : QImageReader(in + "/" + fs.first()).read();
        overlayTiles(m_lastOrig);
        m_origInfo->setText(m_lastOrig.isNull() ? QString()
                            : QString("原图: %1 × %2").arg(m_lastOrig.width()).arg(m_lastOrig.height()));
        m_lastRes = QImage();
        m_viewRes->setText("结果图（推理后逐张更新）");
    } else if (QString("mp4 avi mov mkv wmv flv webm m4v").contains(fi.suffix().toLower())) {
        // 视频不预播放（不依赖系统解码器）；点开始推理后左栏显示实时原始帧
        if (m_plR) { m_plR->stop(); m_stackR->setCurrentIndex(0); }
        m_lastOrig = QImage();
        m_viewOrig->setText("原视频（点开始推理后实时显示）");
        m_origInfo->clear();
        m_stackL->setCurrentIndex(0);
        m_lastRes = QImage();
        m_viewRes->setText("结果视频（实时帧，完成后播放）");
    } else {
        if (m_plR) { m_plR->stop(); m_stackR->setCurrentIndex(0); }
        m_lastOrig = QImageReader(in).read();
        overlayTiles(m_lastOrig);
        m_origInfo->setText(m_lastOrig.isNull() ? QString()
                            : QString("原图: %1 × %2").arg(m_lastOrig.width()).arg(m_lastOrig.height()));
        m_lastRes = QImage();
        m_viewRes->setText("结果图（推理后显示）");
    }
    showImages();
}

QStringList MainWindow::buildArgs(const GuiParams& p) const {
    QStringList a;
    a << p.modelPath << p.input << p.task;
    a << "--conf" << QString::number(p.conf, 'f', 2);
    a << "--nms"  << QString::number(p.nms,  'f', 2);
    if (!p.draw)      a << "--no-draw";
    if (!p.drawClass) a << "--no-class";
    if (p.drawConf)   a << "--draw-conf";
    if (p.drawId)     a << "--draw-id";
    if (p.track)      a << "--track";
    if (p.track && p.minTrackFrames > 0)
        a << "--min-track-frames" << QString::number(p.minTrackFrames);
    if (!p.save)      a << "--no-save";
    if (p.save && !p.saveDir.trimmed().isEmpty())
        a << "--save-dir" << p.saveDir.trimmed();
    if (p.crop) {
        a << "--crop"
          << "--crop-size" << QString("%1x%2").arg(p.cropW).arg(p.cropH)
          << "--crop-overlap" << QString("%1x%2").arg(p.ovW).arg(p.ovH)
          << "--threads" << QString::number(p.threads);
    }
    if (p.frameStep > 1) a << "--frame-step" << QString::number(p.frameStep);
    if (p.preview && QFileInfo(p.input).isFile() &&
        QString("mp4 avi mov mkv wmv flv webm m4v").contains(QFileInfo(p.input).suffix().toLower())) {
        a << "--preview-frames" << "--frame-dir"
          << QDir::toNativeSeparators(QCoreApplication::applicationDirPath() + "/preview_frames");
    }
    if (p.exportJson) a << "--export-frames";   // 逐帧 原图+LabelMe json（save-dir/<视频名>/）
    if (p.isCuda) a << "--cuda" << "--gpu" << QString::number(p.cudaID);
    if (!p.labelsPath.trimmed().isEmpty())
        a << "--labels" << p.labelsPath.trimmed();
    return a;
}

GuiParams MainWindow::collect() {
    GuiParams p;
    p.modelPath  = QDir::toNativeSeparators(m_model->text().trimmed());
    p.labelsPath = QDir::toNativeSeparators(m_labels->text().trimmed());
    p.input      = QDir::toNativeSeparators(m_input->text().trimmed());
    p.task       = m_task->currentText();
    p.conf       = m_conf->value();
    p.nms        = m_nms->value();
    p.draw = m_draw->isChecked();         p.drawClass = m_drawClass->isChecked();
    p.drawConf = m_drawConf->isChecked(); p.drawId = m_drawId->isChecked();
    p.track = m_track->isChecked();
    p.minTrackFrames = m_minTrackFrames->value();
    p.save = m_save->isChecked();
    p.saveDir = m_saveDir->text().trimmed();
    if (!p.saveDir.isEmpty() && QDir::isRelativePath(p.saveDir))   // 统一绝对路径（相对时按程序目录）
        p.saveDir = QDir::toNativeSeparators(QCoreApplication::applicationDirPath() + "/" + p.saveDir);
    p.preview = m_preview->isChecked();
    p.exportJson = m_exportJson->isChecked();
    p.crop = m_crop->isChecked() && m_crop->isEnabled();
    p.cropW = m_cropW->value();  p.cropH = m_cropH->value();
    p.ovW = m_ovW->value();      p.ovH = m_ovH->value();
    p.threads = m_threads->value();
    p.frameStep = m_frameStep->value();
    p.isCuda = m_cuda->isChecked();
    p.cudaID = m_cudaId->value();
    return p;
}

void MainWindow::onStart() {
    if (m_busy) return;
    if (m_model->text().trimmed().isEmpty() || !QFileInfo::exists(m_model->text())) {
        log("[error] 请先选择模型 onnx"); return;
    }
    if (m_input->text().trimmed().isEmpty() || !QFileInfo::exists(m_input->text())) {
        log("[error] 请先选择输入（图片/文件夹/视频）"); return;
    }
    QString cli = resolveCli();
    if (cli.isEmpty()) {
        log("[error] 找不到 yolov-onnxrt.exe（请放到本程序同目录或其上级目录）"); return;
    }
    GuiParams p = collect();
    m_videoMode = QFileInfo(p.input).isFile() &&
                  QString("mp4 avi mov mkv wmv flv webm m4v").contains(QFileInfo(p.input).suffix().toLower());
    if (m_videoMode) {
        m_streamSeen = false;
        m_viewOrig->setText("原视频（实时帧）");
        m_stackL->setCurrentIndex(0);                        // 左：CLI 原始帧流（不依赖系统解码器）
        if (m_plR) m_plR->stop();
        m_stackR->setCurrentIndex(0);                        // 右：实时帧 → 完成后结果视频
        m_lastOrig = QImage();
        m_lastRes = QImage();
        m_viewRes->setText("结果视频（实时帧，完成后播放）");
    } else {
        if (m_plR) { m_plR->stop(); m_stackR->setCurrentIndex(0); }
        if (QFileInfo(p.input).isFile()) {                   // 单图：左=原图
            QImageReader ro(p.input);
            m_lastOrig = ro.read();
            m_lastRes = QImage();
            m_viewRes->setText("结果图");
        } else {                                             // 文件夹：左右逐张更新
            m_lastOrig = QImage(); m_lastRes = QImage();
            m_viewOrig->setText("原图（逐张更新）");
            m_viewRes->setText("结果图（逐张更新）");
        }
        showImages();
    }
    m_proc->setWorkingDirectory(QCoreApplication::applicationDirPath());
    m_proc->start(cli, buildArgs(p));
    if (!m_proc->waitForStarted(5000)) {
        log("[error] 子进程启动失败"); return;
    }
    setBusy(true);
    m_bar->setRange(0, 0);                       // 先转圈；图片模式下面按张数设定
    if (!m_videoMode && QDir(p.input).exists()) {
        int n = QDir(p.input).entryList(QStringList{"*.jpg", "*.jpeg", "*.png", "*.bmp"}, QDir::Files).size();
        if (n > 0) m_bar->setRange(0, n);
    }
    log(QString("=== 开始: %1 ===").arg(m_input->text()));
    log("结果保存目录: " + absSaveDir(p));   // 明示实际落盘位置
    if (p.exportJson) {
        QFileInfo fiIn(p.input);
        const bool isVid = fiIn.isFile() &&
            QString("mp4 avi mov mkv wmv flv webm m4v").contains(fiIn.suffix().toLower());
        log(isVid ? QString("json导出: %1（逐帧原图+json）")
                        .arg(QDir::toNativeSeparators(absSaveDir(p) + "/" + fiIn.completeBaseName()))
                  : "json导出: 原图同目录（<原图名>.json，原图不动）");
    }
}

void MainWindow::onStop() {
    if (m_proc && m_proc->state() != QProcess::NotRunning) {
        log("... 停止：终止子进程");
        m_proc->kill();
    }
}

// 解析 CLI stdout：图片行 "<name>  [N objs...]"、视频行 "frame N [...]"、结尾 done.
void MainWindow::onProcStdout() {
    static QString pending;
    const GuiParams p = collect();               // 左栏原图路径/右栏结果目录所需
    pending += QString::fromLocal8Bit(m_proc->readAllStandardOutput());
    QStringList lines = pending.split('\n');
    pending = lines.takeLast();                  // 末段可能是半行，留到下次
    int done = 0;
    for (const QString& raw : lines) {
        QString line = raw.trimmed();
        if (line.isEmpty()) continue;
        int so = line.indexOf(" so:");
        int sf = line.indexOf(" sf:");
        if (so >= 0 || sf >= 0) {                      // 双流：左=原始帧，右=处理帧
            auto absCli = [](QString fp) {
                if (QDir::isRelativePath(fp)) fp = QCoreApplication::applicationDirPath() + "/" + fp;
                return fp;
            };
            if (so >= 0) {
                QImageReader ro(absCli(line.mid(so + 4, (sf >= 0 ? sf - so - 4 : -1)).trimmed()));
                QImage imo = ro.read();
                if (!imo.isNull()) m_lastOrig = imo;
            }
            if (sf >= 0) {
                QImageReader rf(absCli(line.mid(sf + 4).trimmed()));
                QImage imf = rf.read();
                if (!imf.isNull()) m_lastRes = imf;
            }
            if (!m_streamSeen && !m_lastOrig.isNull()) {
                m_streamSeen = true;
                log("实时帧已连接");
            }
            showImages();
            line = line.left((so >= 0 ? so : sf)).trimmed();
        }
        log(line);
        if (m_preview->isChecked() && !m_videoMode && line.contains("[") && line.contains("objs")) {
            QString name = line.section("[", 0, 0).trimmed();   // 文件名在行首
            if (!name.isEmpty() && !name.contains(" ")) {
                if (QDir(p.input).exists()) {                   // 文件夹模式：左=原图
                    QImageReader ro(p.input + "/" + name);
                    QImage imo = ro.read();
                    if (!imo.isNull()) m_lastOrig = imo;
                }
                QImageReader r(absSaveDir(p) + "\\" + name);    // 右=结果图
                QImage im = r.read();
                if (!im.isNull()) m_lastRes = im;
                showImages();
            }
            ++done;
            if (m_bar->maximum() > 0) m_bar->setValue(done);
        }
        if (line.contains("保存: ")) {                           // 视频完成：右=结果视频循环播放
            QString vp = line.section("保存: ", 1).trimmed();
            if (!vp.isEmpty()) playVideo(m_plR, m_vwR, m_stackR, vp);
        }
    }
}

void MainWindow::onProcStderr() {
    for (const QString& l : QString::fromLocal8Bit(m_proc->readAllStandardError()).split('\n'))
        if (!l.trimmed().isEmpty()) log(l.trimmed());
}

void MainWindow::onProcFinished(int code, QProcess::ExitStatus st) {
    if (st == QProcess::CrashExit)
        log(QString("[stopped] 子进程被终止 (code=%1)").arg(code));
    else
        log(QString("=== 结束 (code=%1) ===").arg(code));
    setBusy(false);
    m_bar->setRange(0, 1); m_bar->setValue(1);
}

void MainWindow::onProcError(QProcess::ProcessError err) {
    if (err == QProcess::FailedToStart)
        log("[error] 子进程启动失败（缺 DLL？把 yolov-onnxrt 全套放同目录）");
}

void MainWindow::onPickModel() {
    QString f = QFileDialog::getOpenFileName(this, "选择模型", QString(), "ONNX 模型 (*.onnx)");
    if (!f.isEmpty()) {
        m_model->setText(QDir::toNativeSeparators(f));
        QFileInfo info(f);                       // 默认填同名 txt（约定路径）
        QString txt = info.absolutePath() + "/" + info.completeBaseName() + ".txt";
        if (QFileInfo::exists(txt) && m_labels->text().isEmpty())
            m_labels->setText(QDir::toNativeSeparators(txt));
    }
}
void MainWindow::onPickLabels() {
    QString f = QFileDialog::getOpenFileName(this, "选择类别 txt", QString(), "文本 (*.txt)");
    if (!f.isEmpty()) m_labels->setText(QDir::toNativeSeparators(f));
}
void MainWindow::onPickImage() {
    QString f = QFileDialog::getOpenFileName(this, "选择图片", QString(),
                                             "图片 (*.jpg *.jpeg *.png *.bmp)");
    if (!f.isEmpty()) { m_input->setText(QDir::toNativeSeparators(f)); previewInput(); }
}
void MainWindow::onPickVideo() {
    QString f = QFileDialog::getOpenFileName(this, "选择视频", QString(),
                                             "视频 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v)");
    if (!f.isEmpty()) { m_input->setText(QDir::toNativeSeparators(f)); previewInput(); }
}
void MainWindow::onPickFolder() {
    QString d = QFileDialog::getExistingDirectory(this, "选择图片文件夹");
    if (!d.isEmpty()) { m_input->setText(QDir::toNativeSeparators(d)); previewInput(); }
}
void MainWindow::onPickSaveDir() {
    QString d = QFileDialog::getExistingDirectory(this, "选择保存目录");
    if (!d.isEmpty()) m_saveDir->setText(QDir::toNativeSeparators(d));
}

void MainWindow::showImages() {
    std::pair<QImage*, QLabel*> panes[2] = { {&m_lastOrig, m_viewOrig}, {&m_lastRes, m_viewRes} };
    for (auto& s : panes) {
        if (s.first->isNull()) continue;
        QSize dst = s.second->size();
        if (dst.width() < 16 || dst.height() < 16) dst = QSize(480, 360);
        s.second->setPixmap(QPixmap::fromImage(*s.first).scaled(
            dst, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void MainWindow::playVideo(QMediaPlayer*& pl, QVideoWidget* vw,
                           QStackedWidget* st, const QString& file) {
    if (!pl) {
        pl = new QMediaPlayer(this);
        pl->setVideoOutput(vw);
        QMediaPlayer* p = pl;                                // 播完从头：循环
        connect(pl, &QMediaPlayer::mediaStatusChanged, this, [p](QMediaPlayer::MediaStatus s) {
            if (s == QMediaPlayer::EndOfMedia) { p->setPosition(0); p->play(); }
        });
        connect(pl, QOverload<QMediaPlayer::Error>::of(&QMediaPlayer::error), this,
                [this, p](QMediaPlayer::Error) {     // 缺解码器等：回退到实时帧画面
            log("[warn] 结果视频播放失败（缺解码器），保留实时帧画面: " + p->errorString());
            m_stackR->setCurrentIndex(0);
        });
    }
    st->setCurrentWidget(vw);
    pl->setMedia(QUrl::fromLocalFile(file));
    pl->play();
}

void MainWindow::resizeEvent(QResizeEvent* e) {
    QMainWindow::resizeEvent(e);
    showImages();                                // 窗口变化 → 双栏预览自适应
}

// 在原图上叠画滑窗切块网格（与 CLI 切块一致：最后一块贴边截短）；仅边框+左下角 r行c列 标签，不加蒙版
void MainWindow::overlayTiles(QImage& im) const {
    if (!m_crop || !m_crop->isChecked() || im.isNull()) return;
    const int bw = m_cropW->value(), bh = m_cropH->value();
    const int W = im.width(), H = im.height();
    if (bw <= 0 || bh <= 0) return;
    QPainter p(&im);
    QFont f = p.font(); f.setPointSize(10); f.setBold(true); p.setFont(f);
    const int sx = qMax(1, bw - m_ovW->value()), sy = qMax(1, bh - m_ovH->value());
    int idx = 0, row = 0;
    for (int y = 0;; y += sy) {
        const int ty = qMin(y, qMax(0, H - bh));
        int col = 0;
        for (int x = 0;; x += sx) {
            const int tx = qMin(x, qMax(0, W - bw));
            const QRect r(tx, ty, qMin(bw, W), qMin(bh, H));
            ++col; ++idx;
            const QColor c = QColor::fromHsv((idx * 47) % 360, 210, 255, 235);   // 黄金角取色：相邻块色相错开
            p.setPen(QPen(c, 2));
            p.drawRect(r);
            const QRect  tr = r.adjusted(5, 0, -5, -3);
            const QString tag = QString("r%1c%2").arg(row + 1).arg(col);
            p.setPen(Qt::black); p.drawText(tr.translated(1, 1), Qt::AlignBottom | Qt::AlignLeft, tag);
            p.setPen(Qt::white); p.drawText(tr,                   Qt::AlignBottom | Qt::AlignLeft, tag);
            if (tx + bw >= W) break;
        }
        ++row;
        if (ty + bh >= H) break;
    }
}

QString MainWindow::stamp() const {
    // 北京时间（UTC+8 固定，无夏令时），毫秒精度
    return QDateTime::currentDateTimeUtc().addSecs(8 * 3600).toString("[HH:mm:ss.zzz] ");
}

void MainWindow::log(const QString& s) {
    m_log->appendPlainText(stamp() + s);
}

void MainWindow::setBusy(bool busy) {
    m_busy = busy;
    m_start->setEnabled(!busy);
    m_stop->setEnabled(busy);
}
