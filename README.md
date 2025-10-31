# exvllm

| [快速开始](#快速开始) | [版本日志](docs/version.md) | [English Document](README_EN.md)

## 介绍

exvllm是外挂的vllm插件，可以扩展vllm使用moe混合推理功能

部署交流QQ群： 903418132

微信群：![二维码](docs/wechat_group0.jpg)

## 亮点功能

- 🚀 安装使用简单方便，一条命令就能成功安装，一条命令就能成功运行。
- 🚀 支持CPU + GPU混合推理MOE大参数模型（单显卡即可推理DEEPSEEK 671B）。

## 快速开始

### 安装

- `pip`安装速度慢时，可使用镜像加速

```
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

#### Linux系统 + Nvidia GPU:

建议在Python虚拟环境中安装，防止破坏其它环境

首先安装vllm。一般可以使用pip安装，若不成功则参照vllm文档用其它方式安装

```
pip install vllm 
```

然后安装exvllm插件

```
pip install exvllm -U
```

### 运行api server

```
exvllm serve Qwen/Qwen3-30B-A3B
```

## 使用指南

### 0. 支持模型格式

目前支持原始模型, FP8模型, AWQ模型

### 1. 运行参数

使用`vllm --help`可以查看vllm原本的参数

`exvllm`可以通过下列环境变量设置运行参数

需要注意的是，速度和参数设置并不一定正相关，如果对性能要求高，可以多方向尝试一下

- `FT_THREADS`:
  - **描述**: 设置使用的CPU线程数。
  - **示例**: `FT_THREADS=30 exvllm serve Qwen/Qwen3-30B-A3B`

### 源码安装

若pip安装失败或有其它特殊需求，可以用源码编译安装
源码安装后如果需要卸载，方法和PIP安装一样
```
pip uninstall ftllm
```

建议使用cmake编译，需要提前安装gcc，g++ (建议9.4以上), make, cmake (建议3.23以上)

GPU编译需要提前安装好CUDA编译环境，建议使用尽可能新的CUDA版本

需要安装依赖

``` sh
apt-get install libnuma-dev
```

使用如下命令编译

``` sh
bash install.sh
```
