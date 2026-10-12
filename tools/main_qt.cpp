// Qt 图形界面入口
#include <QApplication>
#include <QIcon>
#include "main_window.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setWindowIcon(QIcon(":/app.png"));   // 标题栏/任务栏图标（Y）
    MainWindow w;
    w.show();
    return app.exec();
}
