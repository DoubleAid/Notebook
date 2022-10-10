# cannot convert argument 1 from int to int && error
代码如下
```cpp
std::unordered_map<GeoTileId, std::pair<std::string, int32_t>> RoadMapUpdateCall::getTileTable(int version) {
    std::unordered_map<GeoTileId, std::pair<std::string, int32_t>> tile_tables;
    tile_tables.emplace(tile_id, std::make_pair<std::string, int32_t>(tile.md5, 0));
}
```
只需要把代码修改一下
```
tile_tables.emplace(tile_id, std::make_pair<std::string, int32_t>(tile.md5, 0));
```
修改成
```
tile_tables.emplace(tile_id, std::make_pair(tile.md5, 0));
```

This is not how `std::make_pair` is intended to be used; you are not supposed to explicitly specify the template arguments.

The C++11 `std::make_pair` takes two arguments, of type `T&&` and `U&&`, where `T` and `U` are template type parameters. Effectively, it looks like this  
```cpp
template <typename T, typename U>
[return type] make_pair(T&& argT, U&& argU);
```

When you call `std::make_pair` and explicitly specify the template type arguments, no argument deduction takes place.(没有参数会被释放) Instead, the type arguments are substituted（替代） directly into the template declaration, yielding:
```
[return type] make_pair(std::string&& argT, int&& argU);
```

Note that both of these parameter types are rvalue references. Thus, they can only bind to rvalues. This isn't a problem for the second argument that you pass `7`, because that is an rvalue expression. however, `tile.md5()` is an lvalue expression (it isn't a temporary and it isn't being moved). This means the function template is not a match for you arguments, which is why you get the error.

So, why does it work when you don't explicitly specify what `T` and `U` are in the template argument list? In short, rvalue reference parameters are special in templates. Due in part to a language feature called reference collapsing, an rvalue reference parameter of type `A&&`, where `A` is a template type parameter, can bind to any kind of `A`.

It doesn't matter whether the `A` is a lvalue, an rvalue, const-qualified, volatile-qualified, or unqualified, an `A&&` can bind to that object (again, if and only if `A` if itself a template parameter)

We can make the call:
```
make_pair(tile.md5(), 0);
```
Here, '0' is an lvalue, to `T&&`, the compiler deduces `T` to be `std::string&`, yielding an argument of type `std::string& &&`. There are no references to references, though, so this "double reference" collapses to become `std:;string&`, `tile.md5()` if a match.

It's simple to bind `0` to `U&&`: the compiler can deduce `U` to be `int`, yielding a parameter of type `int&&`, which binds successfully to `7` because it is an rvalue.

There are lots of subtleties with these new language features, but if you follow one simple rule, it's pretty easy:
```
If a template argument can be deduced from the function arguments, let it be deduced. Don't explicitly provide the argument unless you absolutely must.
  
Let the compiler do the hard work, and 99.9% of the time it'll be exactly what you wanted anyway. When it isn't what you wanted, you'll usually get a compilation error which is easy to identify and fix.
```
[参考文档](https://stackoverflow.com/questions/9641960/)