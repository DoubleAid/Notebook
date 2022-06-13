# 功能实现： 父子 Scrollview 滑动
实现子滑动和父滑动相互区别，在子滑动上只进行子滑动
设置父滑动的`DragGesture` 的最小滑动距离

```swift
DragGesture(minimumDistance: 50, coordinateSpace: .local)
```