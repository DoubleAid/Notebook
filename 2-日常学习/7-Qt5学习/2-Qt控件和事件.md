Qt 中的每个控件都由特定的类表示，每个控件类都包含一些常用的属性和方法，所有的控件类都直接或者间接继承自 QWidget 类。实际开发中，我们可以使用 Qt 提供的这些控件，也可以通过继承某个控件类的方式自定义一个新的控件。

前面说过，Qt 中所有可视化的元素都称为控件，我们习惯将带有标题栏、关闭按钮的控件称为窗口。例如，下图展示了两种常用的窗口，实现它们的类分别是 QMainWindow 和 QDialog。 
+ QMainWindow 类生成的窗口自带菜单栏、工具栏和状态栏，中央区域还可以添加多个控件，常用来作为应用程序的主窗口；
+ QDialog 类生成的窗口非常简单，没有菜单栏、工具栏和状态栏，但可以添加多个控件，常用来制作对话框。

除了 QMainWindow 和 QDialog 之外，还可以使用 QWidget 类，它的用法非常灵活，既可以用来制作窗口，也可以作为某个窗口上的控件。

Qt 程序可以接收的事件种类有很多，例如鼠标点击事件、鼠标滚轮事件、键盘输入事件、定时事件等。每接收一个事件，Qt 会分派给相应的事件处理函数来处理。所谓事件处理函数，本质就是一个普通的类成员函数，以用户按下某个 QPushButton 按钮为例，Qt 会分派给 QPushButton 类中的 mousePressEvent() 函数处理。

Qt 中的所有控件都具有接收信号的能力，一个控件还可以接收多个不同的信号。对于接收到的每个信号，控件都会做出相应的响应动作。例如，按钮所在的窗口接收到“按钮被点击”的信号后，会做出“关闭自己”的响应动作；再比如输入框自己接收到“输入框被点击”的信号后，会做出“显示闪烁的光标，等待用户输入数据”的响应动作。在 Qt 中，对信号做出的响应动作就称为槽。

信号函数和槽函数通常位于某个类中，和普通的成员函数相比，它们的特别之处在于：
+ 信号函数用  signals 关键字修饰，槽函数用 public slots、protected slots 或者 private slots 修饰。signals 和 slots 是 Qt 在 C++ 的基础上扩展的关键字，专门用来指明信号函数和槽函数；
+ 信号函数只需要声明，不需要定义（实现），而槽函数需要定义（实现）。

connect() 是 QObject 类中的一个静态成员函数，专门用来关联指定的信号函数和槽函数。
关联某个信号函数和槽函数，需要搞清楚以下 4 个问题， 也是connect 函数的四个变量：
+ 信号发送者是谁？
+ 哪个是信号函数？
+ 信号的接收者是谁？
+ 哪个是接收信号的槽函数？

在 Qt5 版本之前，connect() 函数最常用的语法格式是：
```
QObject::connect(const QObject *sender, const char *signal, const QObject *receiver, const char *method, Qt::ConnectionType type = Qt::AutoConnection)
```
各个参数的含义分别是：
+ sender：指定信号的发送者；
+ signal：指定信号函数，信号函数必须用 SIGNAL() 宏括起来；
+ reveiver：指定信号的接收者；
+ method：指定接收信号的槽函数，槽函数必须用 SLOT() 宏括起来；
+ type 用于指定关联方式，默认的关联方式为 Qt::AutoConnection，通常不需要手动设定。

用 connect() 函数将 But 按钮的 clicked() 信号函数和 widget 窗口的 close() 槽函数关联起来，实现代码如下：
```cpp
connect(&But, SIGNAL(clicked()), &widget, SLOT(close()));
```

