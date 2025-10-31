#!/bin/bash
folder="build-exvllm"

# 创建工作文件夹
if [ ! -d "$folder" ]; then
    mkdir "$folder"
fi

cd $folder
cmake .. "$@" -DCUDA_ARCH="52;53;70;89" -D CMAKE_CXX_COMPILER=g++-11 -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc
make -j$(nproc)

# 编译失败停止执行
if [ $? != 0 ]; then
    exit -1
fi

cd python
ldd exvllm/libft_kernel.so | grep '=>' | awk '{print $3}' | grep 'libnuma' | xargs -I {} cp -n {} exvllm/.
python3 setup.py sdist build
python3 setup.py bdist_wheel --plat-name manylinux2014_$(uname -m)
#python3 setup.py install --all