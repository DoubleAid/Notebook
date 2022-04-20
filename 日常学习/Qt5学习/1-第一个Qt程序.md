新建的项目包括三个文件
+ main.cpp
+ mainwindow.h
+ mainwindow.cpp

### <font color=coral>main.cpp</font>
```cpp
#include "mainwindow.h"
#include <QApplication>
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    // 默认情况下，Qt 提供的所有组件（控件、部件）都是隐藏的，不会自动显示。通过调用 MainWindow 类提供的 show() 方法，w 窗口就可以在程序运行后显示出来。
    w.show();
    return a.exec();
}
```
### <font color=coral>mainWindow.h 和 mainWindow.cpp</font>
```cpp
//mainwindow.h
#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QLabel>      // 引入 QLable 文件框组件的头文件
class MainWindow : public QMainWindow
{
    Q_OBJECT
    public:
        MainWindow(QWidget *parent = 0);
        ~MainWindow();
    private:
        QLabel *lab;        // 定义一个私有的 QLabel 指针对象
};
#endif // MAINWINDOW_H


//mainwindow.cpp
#include "mainwindow.h"
MainWindow::MainWindow(QWidget *parent): QMainWindow(parent)
{
    // 创建一个 QLable 对象
    this->lab = new QLabel("Hello,World!",this);
}
MainWindow::~MainWindow()
{}
```
