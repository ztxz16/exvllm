import sys
import importlib
import site
import os
import shutil

def main():
    replace_vllm_file()

def replace_vllm_file():
    """
    替换vllm库中的fused_moe/layer.py文件
    """
    
    # 获取site-packages路径
    site_packages = site.getsitepackages()
    
    # 查找正确的site-packages目录
    target_site_packages = None
    for sp in site_packages:
        if os.path.exists(sp):
            vllm_path = os.path.join(sp, 'vllm')
            if os.path.exists(vllm_path):
                target_site_packages = sp
                break
    
    if not target_site_packages:
        print("错误：未找到vllm库的安装路径")
        return False
    
    replace_dict = [
        (
            os.path.join (target_site_packages, 'exvllm', 'vllm', 'fused_moe_layers.py'),
            os.path.join (target_site_packages, 'vllm', 'model_executor', 'layers', 'fused_moe', 'layer.py')
        ), 
        (
            os.path.join (target_site_packages, 'exvllm', 'vllm', 'model_loader_utils.py'),
            os.path.join (target_site_packages, 'vllm', 'model_executor', 'model_loader', 'utils.py')
        )
    ]

    for it in replace_dict:
        source_file = it[0]
        target_file = it[1]
    
        # 检查源文件是否存在
        if not os.path.exists(source_file):
            print(f"错误：源文件不存在: {source_file}")
            return False
    
        # 检查目标文件是否存在
        if not os.path.exists(target_file):
            print(f"错误：目标文件不存在: {target_file}")
            return False
    
        try:
            # 执行替换
            # print(f"正在替换文件...")
            # print(f"源文件: {source_file}")
            # print(f"目标文件: {target_file}")
            shutil.copy2(source_file, target_file)
        
            # print("文件替换成功！")
        
        except PermissionError:
            print("错误：权限不足。请使用管理员权限运行此脚本。")
            return False
    
        except Exception as e:
            print(f"错误：替换文件时发生异常: {e}")
            return False
    return

def init():
    main()