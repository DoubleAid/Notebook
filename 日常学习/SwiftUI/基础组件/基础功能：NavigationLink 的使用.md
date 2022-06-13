# 基础功能：NavigationLink 的使用
##### 自定义返回

```swift
struct l3: View {
    @Environment(\.presentationMode) var presentationMode
    var body: some View {
        VStack {
            Text("l3 result")
            Button(action: {
                presentationMode.wrappedValue.dismiss()
            }, label: {
                Text("返回")
            })
        }
        .navigationBarBackButtonHidden(true)
        .navigationBarItems(leading: Button(action: {presentationMode.wrappedValue.dismiss()}, label: {
            Text("返回")
        }), trailing: Button(action: {}, label: {
            Text("保存")
        }))
    }
}
```
##### link 链接页隐去 tabview
只需要在 TabView 外添加一个总的 NavigationView