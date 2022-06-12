# 功能实现:带显示滑动位置的 scrollview

```swift
import SwiftUI

struct CustomScrollView<Content>: View where Content: View {
    
    let axes: Axis.Set
    let showIndicators: Bool
    let content: Content
    @Binding var contentOffset: CGFloat
    
    init(_ axes:Axis.Set = .vertical, showIndicators: Bool = true, contentOffset: Binding<CGFloat>, @ViewBuilder content: () -> Content ) {
        self.axes = axes
        self.showIndicators = showIndicators
        self.content = content()
        self._contentOffset = contentOffset
    }
    
    var body: some View {
        GeometryReader { outsideProxy in
            ScrollView(self.axes, showsIndicators: self.showIndicators) {
                ZStack(alignment: self.axes == .vertical ? .top : .leading) {
                    GeometryReader { insideProxy in
                        Color.clear
                            .preference(key: ScrollOffsetPreferenceKey.self, value: [self.calculateOffset(outsideProxy: outsideProxy, insideProxy: insideProxy)])
//                        self.contentOffset = calculateOffset(outsideProxy: outsideProxy, insideProxy: insideProxy)
                    }
                    VStack {
                        self.content
                    }
                }
            }
            .onPreferenceChange(ScrollOffsetPreferenceKey.self) { value in
                        self.contentOffset = value[0]
                    }
        }
    }
    
    private func calculateOffset(outsideProxy: GeometryProxy, insideProxy: GeometryProxy) -> CGFloat {
        if self.axes == .vertical {
            return outsideProxy.frame(in: .global).minY - insideProxy.frame(in: .global).minY
        }
        else {
            return outsideProxy.frame(in: .global).minX - insideProxy.frame(in: .global).minX
        }
    }
}

struct ScrollOffsetPreferenceKey: PreferenceKey {
    typealias Value = [CGFloat]
    
    static var defaultValue: [CGFloat] = [0]
    
    static func reduce(value: inout [CGFloat], nextValue: () -> [CGFloat]) {
        value.append(contentsOf: nextValue())
    }
}

struct Ttest: View {
    @State var offset: CGFloat = 0
    
    var body: some View {
        VStack {
            Text("scroll offset: \(offset)")
            CustomScrollView(contentOffset: $offset) {
                Group {
                    Text("222")
                    Text("333")
                    Text("123")
                    Text("222")
                    Text("333")
                    Text("123")
                    Text("222")
                    Text("333")
                    Text("222")
                    Text("333")
                }
            }
            .disabled(offset > 100)
        }
    }
}

struct TTest_Previews: PreviewProvider {
    static var previews: some View {
        Ttest()
    }
}
```