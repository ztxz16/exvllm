# exvllm

| [快速开始](#快速开始) | [版本日志](docs/version.md) | [English Document](README_EN.md)

## 介绍

exvllm是外挂的vllm插件，可以扩展vllm使用moe混合推理功能

本项目使用了下列项目的项目的部分代码，并参考了一部分优化方法：

https://github.com/kvcache-ai/ktransformers/ 作者 kvcache-ai, 趋境科技，开源了最早的在transformers中进行混合推理的思路、代码，以及早期的numa优化代码

https://github.com/guqiong96/Lvllm 作者 guqiong96 (B站： 爱跳绳的乃龙)，开源了在vllm中进行混合推理的代码，以及改进的numa优化代码

https://github.com/ikawrakow/ik_llama.cpp/ 作者 ikawrakow，开源了很多高效的AVX512算子

感谢以上项目的贡献，具体方法和相关文章请参考 [参考代码和文章](#参考代码和文章)

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

- `FT_THREADS_START` (新增):
  - **描述**: 指定从哪个CPU核心开始分配线程。
  - **默认值**: 0 (从第一个核心开始)
  - **示例**: `FT_THREADS=30 FT_THREADS_START=30 exvllm serve Qwen/Qwen3-30B-A3B`

- `FT_NUMAS`:
  - **描述**: 设置使用的NUMA节点数量。
  - **示例**: `FT_NUMAS=2 exvllm serve Qwen/Qwen3-30B-A3B`

- `FT_NUMAS_START` (新增):
  - **描述**: 指定从哪个NUMA节点开始分配。
  - **默认值**: 0 (从第一个NUMA节点开始)
  - **示例**: `FT_NUMAS=2 FT_NUMAS_START=1 exvllm serve Qwen/Qwen3-30B-A3B`

### 多实例使用示例

```bash
# 实例1：使用核心0-29
FT_THREADS=30 exvllm serve Qwen/Qwen3-30B-A3B --port 8000 &

# 实例2：使用核心30-59
FT_THREADS=30 FT_THREADS_START=30 exvllm serve Qwen/Qwen3-30B-A3B --port 8001 &

# 实例3：使用核心60-89
FT_THREADS=30 FT_THREADS_START=60 exvllm serve Qwen/Qwen3-30B-A3B --port 8002 &

# 等待所有实例完成
wait
```

更多详细信息请参考 [CPU绑定指南](CPU_BINDING_GUIDE.md)。

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

## 参考代码和文章

1、在大模型框架中使用算子替代的方法实现混合推理的思路

[灵活可配的 CPU/GPU 异构大模型推理策略 - KTransformers](https://zhuanlan.zhihu.com/p/714877271)

2、混合推理中cuda graph的使用

[CUDA Graph 在 Transformers 中的使用和进一步改进 - KTransformers](https://zhuanlan.zhihu.com/p/714877271)

[KT在transformers中的实现](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/cpuinfer.h)

3、具体在vllm中MOE算子的替代算子

[lvllm中用于vllm推理的代码](https://github.com/guqiong96/Lvllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)

4、具体在vllm中cuda_graph的使用

[lvllm中用于vllm推理的代码](https://github.com/guqiong96/Lvllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)

[lvllm中可用于cuda graph的c++算子](https://github.com/guqiong96/Lvllm/blob/main/csrc/lk/lk_bindings.cpp)

5、MOE算子线程不平衡时动态调度的思路

[KTransformers 0.3 思路介绍](https://zhuanlan.zhihu.com/p/1900318746402329329)

[KT中关于线程调度的相关代码](https://github.com/kvcache-ai/ktransformers/blob/main/csrc/ktransformers_ext/cpu_backend/backend.cpp)

6、基于numa改进的MOE动态调度算子

[lvllm中的实现](https://github.com/guqiong96/Lvllm/blob/main/csrc/lk/moe.cpp)

感谢大佬对开源社区的贡献！如发现未标明的引用代码可在issue中提出