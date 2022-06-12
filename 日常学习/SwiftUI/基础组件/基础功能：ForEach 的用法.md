# 基础功能：ForEach 的用法

```swift
ForEach([2, 4, 6, 8, 10], id: \.self) {
    Text("\($0) is even")
}
```

```swift
VStack {
    ForEach(Array(some_string.enumerated()), id: \.offset) { character in
        Text(String(character.element))
    }
}
```

```swift
ForEach(0 ..< liJuCount, id: \.self) { i in
    VStack(alignment:.leading, spacing: 5) {
        Text(enLiJu[i])
            .font(.callout)
            .fontWeight(.light)
        Text(cnLiJu[i])
            .font(.callout)
            .fontWeight(.light)
    }
}
```