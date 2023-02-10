# vector
```cpp
reserve() // 修改容器的容量
resize() // 修改实际使用的元素的个数
push_back()
pop_back()
clear()

iterator erase(pos) // 删除指定元素
iterator erase(first, last) // //删除指定范围内[first, last)的元素

iterator emplace(pos, args) // emplace() 每次只能插入一个元素，而不是多个, pos 为指定插入位置的迭代器；args... 表示与新插入元素的构造函数相对应的多个参数；该函数会返回表示新插入元素位置的迭代器。
emplace_back() 
swap(vector<T>)

iterator insert(pos, elem) // 在迭代器 pos 指定的位置之前插入一个新元素elem，并返回表示新插入元素位置的迭代器。
iterator insert(pos, n, elem) //
iterator insert(pos, start, last) // 在迭代器 pos 指定的位置之前，插入其他容器（不仅限于vector）中位于 [start,last) 区域的所有元素，并返回表示第一个新插入元素位置的迭代器。
iterator insert(pos, initlist) // 在迭代器 pos 指定的位置之前，插入初始化列表（用大括号{}括起来的多个元素，中间有逗号隔开）中所有的元素，并返回表示第一个新插入元素位置的迭代器。

front() // 返回第一个元素的引用。
back() // 返回最后一个元素的引用。
begin() // 返回指向容器中第一个元素的迭代器。
end() // 返回指向容器最后一个元素所在位置后一个位置的迭代器，通常和 begin() 结合使用。
```

# queue
```cpp
front()
back()

push() // 在 queue 的尾部添加一个元素的副本。这是通过调用底层容器的成员函数 push_back() 来完成的。
emplace() // 用传给 emplace() 的参数调用 T 的构造函数，在 queue 的尾部生成对象。
pop() // 删除 queue 中的第一个元素。

size()
swap()
```

# unordered_map


# set


# stack

