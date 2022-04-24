本章介绍 利用 replication controller 来扩展和管理pod
+ 介绍 stateful app 和 stateless app
+ 什么是 replication controller
+ 通过 kubectl 操作 replication controller
 
## <font color=coral>什么是 stateless 和 stateful</font>
stateless 是指 应用不受时间，设备等外部条件限制， 不会影响响应返回的数据
```js
function int sum(int a, int b) {
    return a + b;
}
```
stateful 是指会记录每一步操作的状态，即使重启服务，仍然会保留， 比如 mysql
```js
int count = 0;
function int counter() {
	count++;
	return count;
}
```