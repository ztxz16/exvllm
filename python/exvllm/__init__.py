__all__ = [""]

from importlib.metadata import version

try:
    __version__ = version("exvllm")  # 从安装的元数据读取
except:
    __version__ = version("exvllm-rocm")  # 从安装的元数据读取

__all__ = ["ft_kernel"]