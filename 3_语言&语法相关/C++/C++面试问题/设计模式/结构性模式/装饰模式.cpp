// 正常情况下磁盘中的数据文件可以直接读取， 但是对于敏感数据需要进行压缩和加密。
// 我们需要实现两个装饰器， 它们都改变了从磁盘读写数据的方式
// 加密：对数据进行脱敏处理
// 压缩：对数据进行压缩处理

#include <iostream>
#include <string>

using namespace std;

class DataSource {
public:
    virtual void writeData(string data) = 0;
};

class FileDataSource : public DataSource {
public:
    explicit FileDataSource(string file_name) : file_name_(file_name) {}
    void writeData(string data) override {
        cout << "写入文件" << file_name_ << "中：" << data << endl;
    }
private:
    string file_name_;
};

class Decorator : public DataSource {
public:
    explicit Decorator(DataSource* ds) : data_source_(ds) {}
    void writeData(string data) override {
        data_source_->writeData(data);
    }
protected:
    DataSource* data_source_;
};

// 加密装饰器
class EncryDecorator : public Decorator {
public:
    using Decorator::Decorator;
    void writeData(string data) override {
        data = "加密(" + data + ")";
        data_source_->writeData(data);
    }
};

class CompressDecorator : public Decorator {
public:
    using Decorator::Decorator; // 指明使用基类的构造函数 Decorator::Decorator(DataSource*)
    void writeData(string data) override {
        data = "压缩(" + data + ")";
        data_source_->writeData(data);
    }
};

int main() {
    FileDataSource* source1 = new FileDataSource("stdout");
    source1->writeData("tomocat");

    CompressDecorator* source2 = new CompressDecorator(source1);
    source2->writeData("tomocat");

    EncryDecorator* source3 = new EncryDecorator(source2);
    source3->writeData("tomocat");

    delete source1;
    delete source2;
    delete source3;
}