# 语法基础: 重载 <,使用sorted排序

```swift
struct TT {
    var name: String
    var age: Int
}

extension TT: Comparable {
    static func < (lhs:TT, rhs: TT) -> Bool {
        return lhs.age < rhs.age
    }
}
//// Present the view controller in the Live View window
//PlaygroundPage.current.liveView = MyViewController()
var list = [TT(name: "bbb", age: 8), TT(name: "aaa", age: 2), TT(name: "ccc", age: 9)]
```