# 功能实现: ForEach 从字符串输出字符
`ForEach`要求传递给它的值符合`RandomAccessCollection`。String是Sequence，可以通过将其变成`RandomAccessCollection`来将其转换为Array。
使用Array()转串入[Character]：
```swift
VStack {
    ForEach(Array(some_string), id: \.self) { character in
        Text(String(character))
    }
} 
```
通常，要谨慎选择id的唯一性。由于string可能包含重复的字符，因此您可以通过.enumerated()将每个字符Character变成一(offset, element)对（offset它在中的位置String）然后.offset将id和用作从元组对中.element检索Character：
```swift
VStack {
    ForEach(Array(some_string.enumerated()), id: \.offset) { character in
        Text(String(character.element))
    }
}
```