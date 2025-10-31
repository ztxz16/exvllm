import ctypes
import numpy as np
import os
from typing import Optional
import sys

def find_cuda_runtime():
    """在当前Python环境中查找CUDA运行时库"""
    cuda_lib_names = [
        "libcudart.so.12"
    ]
    
    # 可能的CUDA库路径
    possible_paths = []
    
    # 1. 从当前Python环境的site-packages查找
    for path in sys.path:
        if 'site-packages' in path or 'dist-packages' in path:
            # 检查nvidia相关路径
            possible_paths.extend([
                os.path.join(path, 'nvidia', 'cuda_runtime', 'lib'),
                os.path.join(path, 'nvidia', 'cuda_runtime', 'lib64'),
                os.path.join(path, 'nvidia', 'cuda_runtime'),
                os.path.join(path, 'nvidia', 'cuda', 'lib'),
                os.path.join(path, 'nvidia', 'cuda', 'lib64'),
            ])
    
    # 2. 从环境变量LD_LIBRARY_PATH查找
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_library_path:
        possible_paths.extend(ld_library_path.split(':'))
    
    # 3. 从CUDA_HOME环境变量查找
    cuda_home = os.environ.get('CUDA_HOME', '')
    if cuda_home:
        possible_paths.extend([
            os.path.join(cuda_home, 'lib64'),
            os.path.join(cuda_home, 'lib'),
        ])
    
    # 4. 常见的系统路径
    possible_paths.extend([
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib64',
        '/usr/lib',
    ])
    
    # 查找CUDA库
    for path in possible_paths:
        if os.path.exists(path):
            for lib_name in cuda_lib_names:
                lib_path = os.path.join(path, lib_name)
                if os.path.exists(lib_path):
                    return lib_path
    
    return None

extra_libs = ["libnuma.so.1"]
cuda_runtime_path = find_cuda_runtime()
if cuda_runtime_path:
    extra_libs.insert(0, cuda_runtime_path)  # 优先加载CUDA库
# 加载动态库
for extraLibName in extra_libs:
    try:
        ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], extraLibName))
        print("Load", extraLibName)
    except:
        continue

_lib_path = os.path.join(os.path.dirname(__file__), "libft_kernel.so")  # Linux
# _lib_path = os.path.join(os.path.dirname(__file__), "moe.dll")  # Windows
_lib = ctypes.CDLL(_lib_path)

# 定义句柄类型
FastllmLinearWeightHandle = ctypes.c_void_p
FastllmMoeHandle = ctypes.c_void_p

# 定义 C 函数原型
# FastllmLinearWeight 相关函数
_lib.fastllm_linear_weight_create.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
_lib.fastllm_linear_weight_create.restype = FastllmLinearWeightHandle

_lib.fastllm_linear_weight_create_quantized.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.fastllm_linear_weight_create_quantized.restype = FastllmLinearWeightHandle

_lib.fastllm_linear_weight_destroy.argtypes = [FastllmLinearWeightHandle]
_lib.fastllm_linear_weight_destroy.restype = None

# Getter functions
_lib.fastllm_linear_weight_get_batch.argtypes = [FastllmLinearWeightHandle]
_lib.fastllm_linear_weight_get_batch.restype = ctypes.c_int

_lib.fastllm_linear_weight_get_k.argtypes = [FastllmLinearWeightHandle]
_lib.fastllm_linear_weight_get_k.restype = ctypes.c_int

_lib.fastllm_linear_weight_get_m.argtypes = [FastllmLinearWeightHandle]
_lib.fastllm_linear_weight_get_m.restype = ctypes.c_int

_lib.fastllm_linear_weight_get_block.argtypes = [FastllmLinearWeightHandle]
_lib.fastllm_linear_weight_get_block.restype = ctypes.c_int

_lib.fastllm_linear_weight_get_dataType.argtypes = [FastllmLinearWeightHandle]
_lib.fastllm_linear_weight_get_dataType.restype = ctypes.c_int

# Setter functions
_lib.fastllm_linear_weight_set_batch.argtypes = [FastllmLinearWeightHandle, ctypes.c_int]
_lib.fastllm_linear_weight_set_batch.restype = None

_lib.fastllm_linear_weight_set_k.argtypes = [FastllmLinearWeightHandle, ctypes.c_int]
_lib.fastllm_linear_weight_set_k.restype = None

_lib.fastllm_linear_weight_set_m.argtypes = [FastllmLinearWeightHandle, ctypes.c_int]
_lib.fastllm_linear_weight_set_m.restype = None

_lib.fastllm_linear_weight_set_block.argtypes = [FastllmLinearWeightHandle, ctypes.c_int]
_lib.fastllm_linear_weight_set_block.restype = None

# FastllmMoe 相关函数
_lib.fastllm_moe_create.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    FastllmLinearWeightHandle, FastllmLinearWeightHandle, FastllmLinearWeightHandle]
_lib.fastllm_moe_create.restype = FastllmMoeHandle

_lib.fastllm_moe_destroy.argtypes = [FastllmMoeHandle]
_lib.fastllm_moe_destroy.restype = None

_lib.fastllm_moe_get_expertNum.argtypes = [FastllmMoeHandle]
_lib.fastllm_moe_get_expertNum.restype = ctypes.c_int

_lib.fastllm_moe_get_routedExpertNum.argtypes = [FastllmMoeHandle]
_lib.fastllm_moe_get_routedExpertNum.restype = ctypes.c_int

_lib.fastllm_moe_get_hiddenSize.argtypes = [FastllmMoeHandle]
_lib.fastllm_moe_get_hiddenSize.restype = ctypes.c_int

_lib.fastllm_moe_get_intermediateSize.argtypes = [FastllmMoeHandle]
_lib.fastllm_moe_get_intermediateSize.restype = ctypes.c_int

_lib.fastllm_moe_get_hiddenType.argtypes = [FastllmMoeHandle]
_lib.fastllm_moe_get_hiddenType.restype = ctypes.c_int

_lib.fastllm_moe_warm_up.argtypes = [FastllmMoeHandle]
_lib.fastllm_moe_warm_up.restype = None

_lib.fastllm_moe_sync_with_cuda_stream.argtypes = [FastllmMoeHandle, ctypes.c_size_t]
_lib.fastllm_moe_sync_with_cuda_stream.restype = None

_lib.fastllm_moe_submit_with_cuda_stream.argtypes = [
    FastllmMoeHandle, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_lib.fastllm_moe_submit_with_cuda_stream.restype = None

_lib.fastllm_moe_forward.argtypes = [
    FastllmMoeHandle, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p, ctypes.c_void_p]
_lib.fastllm_moe_forward.restype = None


class FastllmLinearWeight:
    def __init__(self, batch, k, m, data, dataType, 
                 scales=None, zeros=None, blockK=None, blockM=None):
        """
        初始化 FastllmLinearWeight
        
        Args:
            batch: 批次大小
            k: 矩阵维度 k
            m: 矩阵维度 m
            data: 数据指针 (整数地址)
            dataType: 数据类型
            scales: 缩放因子指针 (可选)
            zeros: 零点指针 (可选)
            blockK: 块大小 K (可选)
            blockM: 块大小 M (可选)
        """
        data_ptr = ctypes.c_void_p(data)
        
        if scales is not None and zeros is not None:
            scales_ptr = ctypes.c_void_p(scales)
            zeros_ptr = ctypes.c_void_p(zeros)
            self._handle = _lib.fastllm_linear_weight_create_quantized(
                batch, k, m, data_ptr, dataType,
                scales_ptr, zeros_ptr, blockK, blockM)
        else:
            self._handle = _lib.fastllm_linear_weight_create(
                batch, k, m, data_ptr, dataType)
            
        if not self._handle:
            raise RuntimeError("Failed to create FastllmLinearWeight")
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.fastllm_linear_weight_destroy(self._handle)
    
    @property
    def batch(self):
        return _lib.fastllm_linear_weight_get_batch(self._handle)
    
    @batch.setter
    def batch(self, value):
        _lib.fastllm_linear_weight_set_batch(self._handle, value)
    
    @property
    def k(self):
        return _lib.fastllm_linear_weight_get_k(self._handle)
    
    @k.setter
    def k(self, value):
        _lib.fastllm_linear_weight_set_k(self._handle, value)
    
    @property
    def m(self):
        return _lib.fastllm_linear_weight_get_m(self._handle)
    
    @m.setter
    def m(self, value):
        _lib.fastllm_linear_weight_set_m(self._handle, value)
    
    @property
    def block(self):
        return _lib.fastllm_linear_weight_get_block(self._handle)
    
    @block.setter
    def block(self, value):
        _lib.fastllm_linear_weight_set_block(self._handle, value)
    
    @property
    def dataType(self):
        return _lib.fastllm_linear_weight_get_dataType(self._handle)


class FastllmMoe:
    def __init__(self, expertNum, routedExpertNum, hiddenSize, 
                 intermediateSize, hiddenType,
                 gate: FastllmLinearWeight, 
                 up: FastllmLinearWeight, 
                 down: FastllmLinearWeight):
        """
        初始化 FastllmMoe
        
        Args:
            expertNum: 专家数量
            routedExpertNum: 路由专家数量
            hiddenSize: 隐藏层大小
            intermediateSize: 中间层大小
            hiddenType: 隐藏层类型
            gate: FastllmLinearWeight 对象
            up: FastllmLinearWeight 对象
            down: FastllmLinearWeight 对象
        """
        self._handle = _lib.fastllm_moe_create(
            expertNum, routedExpertNum, hiddenSize, intermediateSize, hiddenType,
            gate._handle, up._handle, down._handle)
        
        if not self._handle:
            raise RuntimeError("Failed to create FastllmMoe")
        
        # 保持对权重对象的引用，防止被垃圾回收
        self._gate = gate
        self._up = up
        self._down = down
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.fastllm_moe_destroy(self._handle)
    
    @property
    def expertNum(self):
        return _lib.fastllm_moe_get_expertNum(self._handle)
    
    @property
    def routedExpertNum(self):
        return _lib.fastllm_moe_get_routedExpertNum(self._handle)
    
    @property
    def hiddenSize(self):
        return _lib.fastllm_moe_get_hiddenSize(self._handle)
    
    @property
    def intermediateSize(self):
        return _lib.fastllm_moe_get_intermediateSize(self._handle)
    
    @property
    def hiddenType(self):
        return _lib.fastllm_moe_get_hiddenType(self._handle)
    
    def warm_up(self):
        """预热"""
        _lib.fastllm_moe_warm_up(self._handle)
    
    def sync_with_cuda_stream(self, user_cuda_stream):
        """与 CUDA 流同步"""
        _lib.fastllm_moe_sync_with_cuda_stream(self._handle, user_cuda_stream)
    
    def submit_with_cuda_stream(self, user_cuda_stream, qlen, k, 
                               expert_ids, weights, input, output, 
                               batch_size_tensor):
        """使用 CUDA 流提交任务"""
        expert_ids_ptr = ctypes.cast(expert_ids, ctypes.POINTER(ctypes.c_uint64))
        weights_ptr = ctypes.cast(weights, ctypes.POINTER(ctypes.c_float))
        input_ptr = ctypes.c_void_p(input)
        output_ptr = ctypes.c_void_p(output)
        batch_size_ptr = ctypes.cast(batch_size_tensor, ctypes.POINTER(ctypes.c_int))
        
        _lib.fastllm_moe_submit_with_cuda_stream(
            self._handle, user_cuda_stream, qlen, k,
            expert_ids_ptr, weights_ptr, input_ptr, output_ptr, batch_size_ptr)
    
    def forward(self, qlen, k, expert_ids, weights, input, output, 
                batch_size_tensor=None):
        """
        前向传播
        
        Args:
            qlen: 序列长度
            k: k 值
            expert_ids: 专家 ID 数组的地址
            weights: 权重数组的地址
            input: 输入数据的地址
            output: 输出数据的地址
            batch_size_tensor: 批次大小张量的地址（未使用，保持接口兼容）
        """
        expert_ids_ptr = ctypes.cast(expert_ids, ctypes.POINTER(ctypes.c_uint64))
        weights_ptr = ctypes.cast(weights, ctypes.POINTER(ctypes.c_float))
        input_ptr = ctypes.c_void_p(input)
        output_ptr = ctypes.c_void_p(output)
        
        _lib.fastllm_moe_forward(
            self._handle, qlen, k,
            expert_ids_ptr, weights_ptr, input_ptr, output_ptr)
# 辅助函数，用于将 numpy 数组或 Python 列表转换为指针
def get_pointer(data):
    """
    获取数据的指针地址
    
    Args:
        data: numpy 数组、ctypes 数组或整数地址
    
    Returns:
        整数形式的指针地址
    """
    if isinstance(data, np.ndarray):
        return data.ctypes.data
    elif isinstance(data, int):
        return data
    elif hasattr(data, 'ctypes'):
        return data.ctypes.data
    elif hasattr(data, '_as_parameter_'):
        return ctypes.cast(data, ctypes.c_void_p).value
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
# 数据类型枚举（根据实际的 fastllm::DataType 定义）
class DataType:
    FLOAT32 = 0
    FLOAT16 = 1
    INT8 = 2
    INT4 = 3
    INT2 = 4
    BIT = 5
    INT32 = 6
    # 根据实际需要添加更多类型
# 便捷的创建函数
def create_linear_weight(batch, k, m, data, dataType, 
                         scales=None, zeros=None, blockK=None, blockM=None):
    """
    创建 FastllmLinearWeight 对象的便捷函数
    
    Args:
        batch: 批次大小
        k: 矩阵维度 k
        m: 矩阵维度 m
        data: 数据（numpy 数组或指针地址）
        dataType: 数据类型（使用 DataType 枚举）
        scales: 缩放因子（可选）
        zeros: 零点（可选）
        blockK: 块大小 K（可选）
        blockM: 块大小 M（可选）
    
    Returns:
        FastllmLinearWeight 对象
    """
    data_ptr = get_pointer(data)
    
    if scales is not None and zeros is not None:
        scales_ptr = get_pointer(scales)
        zeros_ptr = get_pointer(zeros)
        return FastllmLinearWeight(
            batch, k, m, data_ptr, dataType,
            scales_ptr, zeros_ptr, blockK, blockM)
    else:
        return FastllmLinearWeight(batch, k, m, data_ptr, dataType)
def create_moe(expertNum, routedExpertNum, hiddenSize, 
               intermediateSize, hiddenType,
               gate, up, down):
    """
    创建 FastllmMoe 对象的便捷函数
    
    Args:
        expertNum: 专家数量
        routedExpertNum: 路由专家数量
        hiddenSize: 隐藏层大小
        intermediateSize: 中间层大小
        hiddenType: 隐藏层类型（使用 DataType 枚举）
        gate: FastllmLinearWeight 对象或参数元组
        up: FastllmLinearWeight 对象或参数元组
        down: FastllmLinearWeight 对象或参数元组
    
    Returns:
        FastllmMoe 对象
    """
    # 如果传入的是元组，则创建 FastllmLinearWeight 对象
    if not isinstance(gate, FastllmLinearWeight):
        gate = create_linear_weight(*gate)
    if not isinstance(up, FastllmLinearWeight):
        up = create_linear_weight(*up)
    if not isinstance(down, FastllmLinearWeight):
        down = create_linear_weight(*down)
    
    return FastllmMoe(
        expertNum, routedExpertNum, hiddenSize, 
        intermediateSize, hiddenType,
        gate, up, down)
# 模块文档字符串
__doc__ = "MOE (Mixture of Experts) bindings"
# 导出的符号
__all__ = [
    'FastllmLinearWeight',
    'FastllmMoe',
    'DataType',
    'create_linear_weight',
    'create_moe',
    'get_pointer'
]
