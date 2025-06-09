# 基础功能：TabView的使用

```swift
struct ContentView: View {
    @State private var selection = 1
    
    var body: some View {
        TabView(selection: $selection){
            FirstTab()
                .tabItem {
                    VStack {
                        Image("first")
                        Text("FirstTab")
                    }
                }
                .tag(0)
            SecondTab()
                .tabItem {
                    VStack {
                        Image("second")
                        Text("SecondTab")
                    }
                }
                .tag(1)
        }
    }
}
```

