std::unique_lock为锁管理模板类，是对通用mutex的封装。
std::unique_lock对象以独占所有权的方式(unique owership)管理mutex对象的上锁和解锁操作，即在unique_lock对象的声明周期内，它所管理的锁对象会一直保持上锁状态；
而unique_lock的生命周期结束之后，它所管理的锁对象会被解锁。
unique_lock具有lock_guard的所有功能，而且更为灵活。虽然二者的对象都不能复制，但是unique_lock可以移动(movable)，因此用unique_lock管理互斥对象，可以作为函数的返回值，也可以放到STL的容器中。

### lock_guard 的用法
