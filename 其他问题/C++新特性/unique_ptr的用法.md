smart pointer 是 C++ 11 用来替代 指针 的新特性
C++11的 SmartPointer有三个：
+ std::unique_ptr
+ std::shared_ptr
+ std::weak_ptr

```cpp
// 不使用 unique_ptr
bool WorthBuying()
{
    TeaShopOwner* the_owner = new TeaShopOwner();
    
    if (the_owner->SupportCCP()) {
        delete the_owner;
        return false;
    }
    
    if (!the_owner->AdmitTaiwanAsACountry()) {
        delete the_owner;
        return false;
    }
    
    delete the_owner;
    return true;
}

// 使用 unique_ptr
bool WorthBuying()
{
    std::unique_ptr<TeaShopOwner> the_owner = std::make_unique<TeaShopOwner>();
    if (the_owner->SupportCCP())
        return false;
    if (!the_owner->AdmitTaiwanAsACountry())
        return false;
    return true;
}
```
**注意**：std::make_unique仅支持C++14


```cpp
// 不使用 unique_ptr
TeaShopOwnder* CreateOwner();

{
    TeaShopOwner* the_owner = CreateOwner();
    // Do something with the_owner
    delete the_owner;
}

// 使用 unique_ptr
std::unique_ptr<TeaShopOwner> CreateOwner();

{
    std::unique_ptr<TeaShopOwner> the_owner = CreateOwner();
    // Do something with the_owner
}
```