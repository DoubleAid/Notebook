# 基础功能：NavigationView 的使用

```swift
NavigationView {
    VStack {
    }
    .navigationBarTitle(Text("Detail View"), displayMode: .inline)
}
```

##### NavigationView 报错
在使用navigationBarTitle 之后，会出现束缚界面错误
```swift
NavigationView {
    ...
}
.navigationViewStyle(StackNavigationViewStyle())
```