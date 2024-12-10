# Autosar的基础概念

公司使用的是Vector的方案，以下只针对其方案和工具使用做讲解

## 1. 什么是Tire1， Tire2，OEM，ECU

+ Tire1 意为车厂一级供货商，给设备厂商供货，也就是车厂零部件的供应商
+ Tire2 就是二级供应商，可以理解为Tire1的供应商
+ OEM 是 Original Equipment Manufacturer 的缩写，通常指设备厂商，主机厂，整车厂，例如宝马，丰田，byd等
+ ECU 就是Electronic Control Unit，也就是开发的那个项目器件，例如雷达，空调控制器等

## 2. 什么是SIP

SIP 或者 SIP包，即Software Integration Package，是 Tire1在做Autosar项目之前，向Vector购买集成了AUTOSAR方案的软件包，Vector最终交付给Tire1时的软件包就是SIP包

Tire1开发者就是基于这个SIP包来做项目上的应用开发

除了SIP，还会遇到SLP，HLP等概念

Software License Package (SLP)：软件许可包，是Vector提供的软件许可协议，Tire1需要根据这个协议来使用Vector提供的软件

Hardware License Package (HLP)：硬件许可包，是Vector提供的硬件许可协议，Tire1需要根据这个协议来使用Vector提供的硬件

后面还会遇到 Beta SIP， Production SIP， QM Approval SIP 等，后面会慢慢介绍

## 3. SIP里有什么

SIP包里包含以下内容：

+ Applications：应用程序，是Vector对这个软件包，做了一个应用工程，可以理解为一个Demo，你可以根据这个案例来构建你的工程
+ BSW：Basic Software，基础软件，是AUTOSAR的核心，包含BSW层的源码，在通过Configuration 添加模块生成代码的时候，工具会将这些代码拷贝到你的工程里去
+ BSWMD：Basic Software Module Description，基础软件模块描述，是AUTOSAR的核心，包含BSW层的配置文件，通过这些配置文件，可以生成BSW层的代码，存放可生成BSW配置的一些策略和关联
+ DaVinciConfigurator: 就是Vecotr的第二个工具，另一个就是Developer，这个Configurator是一个运行软件，和SIP集成在一起，用来配置BSWMD，生成代码，生成ECU描述文件等
+ Doc: SIP包的一些参考文档
+ Generators：就是一些组建的配置生成器，相当于Configurator的插件，通常是写exe等文件
+ Misc：一些不好分类的杂项
+ ThirdParty：就是Vector以外的第三方的内容，一般就是MCAL，即Microcontroller Abstraction Layer，微控制器抽象层，是AUTOSAR的底层，是MCU芯片厂商提供的，Vector也会提供一些MCAL的支持，但是不是所有的MCAL都支持AUTOSAR
+ 