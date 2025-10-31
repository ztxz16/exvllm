# exvllm

| [Quick Start ](#Quick_Start) | [version log](docs/version.md) | [English Document](README_EN.md)

## Introduction

exvllm is an external plugin for vllm that extends its capabilities to support hybrid inference for MoE models.

Deployment & Discussion QQ Group: 903418132

WeChat Group:![qr](docs/wechat_group0.jpg)

## Highlights

- 🚀 Simple and easy installation and usage. Get started with just one command for installation and another for running.
- 🚀 Supports hybrid CPU + GPU inference for large MoE models (e.g., run DEEPSEEK 671B with a single GPU).

## Quick Start

### Installation

- Use mirror sources if pip is slow:

```
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

#### Linux System + Nvidia GPU:

It is recommended to install within a Python virtual environment to avoid conflicts with other environments.

First, install vllm. Usually, it can be installed via pip. If unsuccessful, refer to the vllm documentation for alternative installation methods.

```
pip install vllm 
```

Then, install the exvllm plugin:

```
pip install exvllm -U
```

### Run API Server

```
exvllm serve Qwen/Qwen3-30B-A3B
```

## User Guide

### 0. Supported Model Formats

Currently supports original models, FP8 models, and AWQ models.

### 1. Runtime Parameters

Use `vllm --help` to view the original vllm parameters.

`exvllm` runtime parameters can be configured using the following environment variables.

Note: Performance is not always directly proportional to these settings. For optimal performance, experimentation is recommended.

- `FT_THREADS`:
  - **Description**: Sets the number of CPU threads to use.
  - **Example**: `FT_THREADS=30 exvllm serve Qwen/Qwen3-30B-A3B`

### Install from Source

If pip installation fails or for other specific needs, you can compile and install from source.
To uninstall after installing from source, use the same method as for a PIP installation.

```
pip uninstall ftllm
```

Compilation with cmake is recommended. Ensure you have gcc, g++ (version 9.4 or above recommended), make, and cmake (version 3.23 or above recommended) installed beforehand.

For GPU compilation, a CUDA compilation environment is required. It is advised to use the latest possible CUDA version.

First, install libnuma-dev:

``` sh
apt-get install libnuma-dev
```

Use the following command to compile:

``` sh
bash install.sh
```
