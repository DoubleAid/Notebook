// 比如在构建进度条时， 把进度的变化提取出来
#include <iostream>
#include <list>

using namespace std;

class IProgress {
public:
    virtual void DoProgress(float val) = 0;
    virtual ~IProgress() {}
};

class Download {
private:
    list<IProgress*> iprogress_list;
public:
    void addIProgress(IProgress* iprogress);
    void removeIProgress(IProgress* iprogress);
    void get_some_files() {
        float value; // 进度信息
        // do some thing
        
    }
    virtual void onProgress(float value) {
        auto itor = iprogress_list.begin();
        while(itor != iprogress_list.end()) {
            (*itor) -> DoProgress(value);
            itor++;
        }
    }
};

class MainProcess : public IProgress {
public:
    void DownLoadButtonClick() {
        Download dld;
        dld.addIProgress(this);
        dld.get_some_files();
    }

    virtual void DoProgress(float val) override {
        // display
    }
};