// 在代理模式（Proxy Pattern）中，一个类代表另一个类的功能。这种类型的设计模式属于结构型模式。
// 在代理模式中，我们创建具有现有对象的对象，以便向外界提供功能接口。

// 意图：为其他对象提供一种代理以控制对这个对象的访问。 a->代理->b

// 主要解决：在直接访问对象时带来的问题，比如说：要访问的对象在远程的机器上。
// 在面向对象系统中，有些对象由于某些原因（比如对象创建开销很大，或者某些操作需要安全控制，或者需要进程外的访问）
// ，直接访问会给使用者或者系统结构带来很多麻烦，我们可以在访问此对象时加上一个对此对象的访问层。

// 何时使用：想在访问一个类时做一些控制。
// 如何解决：增加中间层。
// 关键代码：实现与被代理类组合。
// 应用实例： 1、Windows 里面的快捷方式。 

// 优点： 1、职责清晰。 2、高扩展性。 3、智能化。
// 缺点： 1、由于在客户端和真实主题之间增加了代理对象，因此有些类型的代理模式可能会造成请求的处理速度变慢。 
// 2、实现代理模式需要额外的工作，有些代理模式的实现非常复杂。

// 使用场景：按职责来划分，通常有以下使用场景： 
// 1、远程代理。 2、虚拟代理。 
// 3、Copy-on-Write 代理。 
// 4、保护（Protect or Access）代理。 
// 5、Cache代理。 6、防火墙（Firewall）代理。 7、同步化（Synchronization）代理。 
// 8、智能引用（Smart Reference）代理。

// 注意事项： 
// 1、和适配器模式的区别：适配器模式主要改变所考虑对象的接口，而代理模式不能改变所代理类的接口。 
// 2、和装饰器模式的区别：装饰器模式为了增强功能，而代理模式是为了加以控制。
#include <iostream>
#include <string>

using namespace std;

class OnlineVideoLib {
public:
    virtual string listVideos() = 0;
    virtual string getVideoInfo(int id) = 0;
};

// 视频下载类
class OnlineVideoService : public OnlineVideoLib {
public:
    string listVideos() override {
        // 向远程视频后端服务发送一个api请求获取视频信息， 这里忽略实现
        return "video list";
    }
    string getVideoInfo(int id) override {
        // 向远程视频后端服务发送一个api获取某个视频的元数据， 这里忽略现实
        return "video info";
    }
};

// proxy.h
class CachedTVClass : public OnlineVideoLib {
public:
    explicit CachedTVClass(OnlineVideoLib* service) : 
            service_(service), need_reset_(false), list_cache_(""), video_cache_("") {}
    void reset() {
        need_reset_ = true;
    }
    string getVideoInfo(int id) override {
        if(video_cache_ == "" || need_reset_) {
            video_cache_ = service_->getVideoInfo(id);
        }
        return video_cache_;
    }
    string listVideos() override {
        if (list_cache_ == "" || need_reset_) {
            list_cache_ = service_->listVideos();
        }
        return list_cache_;
    }
private:
    OnlineVideoLib* service_;
    string list_cache_;
    string video_cache_;
    bool need_reset_;
};

class TVManager {
public:
    explicit TVManager(OnlineVideoLib* s) : service_(s) {}
    void renderVideoPage(int id) {
        string video_info = service_->getVideoInfo(id);
        cout << "渲染视频页面" << video_info << endl;
    }
    void renderListPanel() {
        string videos = service_->listVideos();
        cout << "渲染视频缩略图列表" << videos << endl;
    }
private:
    OnlineVideoLib* service_;
};

int main() {
    OnlineVideoService* service = new OnlineVideoService();
    CachedTVClass* proxy = new CachedTVClass(service);
    TVManager* manager = new TVManager(proxy);

    manager->renderVideoPage(1);
    manager->renderListPanel();

    delete service;
    delete proxy;
    delete manager;
}