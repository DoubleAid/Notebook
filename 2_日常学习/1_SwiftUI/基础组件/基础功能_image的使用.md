# 基础功能：Image的使用

#### 默认图片
```swift
extension Image {
    init(_ name: String, defaultImage: String) {
        if let img = UIImage(named: name) {
            self.init(uiImage: img)
        } else {
            self.init(defaultImage)
        }
    }
    
    init(_ name: String, defaultSystemImage: String) {
        if let img = UIImage(named: name) {
            self.init(uiImage: img)
        } else {
            self.init(systemName: defaultSystemImage)
        }
    }
    
}
```
#### SF图片的使用
##### 导入图片
```swift
Image(systemName:"repeat")
```
##### 修改图片的颜色
```swift
Image(systemName:"repeat")
    .foregroundColor(.green)
```
##### 修改图片的大小