"""
资源管理器 - 自动清理显存、内存和 CPU 资源
"""
import gc
import torch
import psutil
import os
from typing import Optional
from contextlib import contextmanager


class ResourceManager:
    """
    资源管理器
    自动监控和清理 GPU 显存、CPU 内存等资源
    """
    
    def __init__(self, auto_cleanup: bool = True, verbose: bool = True):
        """
        初始化资源管理器
        
        Args:
            auto_cleanup: 是否自动清理资源
            verbose: 是否输出详细信息
        """
        self.auto_cleanup = auto_cleanup
        self.verbose = verbose
        self.process = psutil.Process(os.getpid())
    
    def get_memory_info(self) -> dict:
        """获取当前内存使用情况"""
        info = {
            "cpu_memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "cpu_memory_percent": self.process.memory_percent(),
        }
        
        if torch.cuda.is_available():
            info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            info["gpu_memory_percent"] = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
        
        return info
    
    def print_memory_info(self, prefix: str = ""):
        """打印内存使用情况"""
        if not self.verbose:
            return
        
        info = self.get_memory_info()
        print(f"[ResourceManager] {prefix}")
        print(f"  CPU 内存: {info['cpu_memory_mb']:.1f} MB ({info['cpu_memory_percent']:.1f}%)")
        
        if torch.cuda.is_available():
            print(f"  GPU 显存分配: {info['gpu_memory_allocated_mb']:.1f} MB")
            print(f"  GPU 显存保留: {info['gpu_memory_reserved_mb']:.1f} MB")
            print(f"  GPU 使用率: {info['gpu_memory_percent']:.1f}%")
    
    def cleanup_gpu(self):
        """清理 GPU 显存"""
        if not torch.cuda.is_available():
            return
        
        if self.verbose:
            before = torch.cuda.memory_allocated() / 1024 / 1024
        
        # 清理 PyTorch CUDA 缓存
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        if self.verbose:
            after = torch.cuda.memory_allocated() / 1024 / 1024
            freed = before - after
            print(f"[ResourceManager] GPU 显存清理: 释放 {freed:.1f} MB")
    
    def cleanup_cpu(self):
        """清理 CPU 内存"""
        if self.verbose:
            before = self.process.memory_info().rss / 1024 / 1024
        
        # 强制垃圾回收
        gc.collect()
        
        if self.verbose:
            after = self.process.memory_info().rss / 1024 / 1024
            freed = before - after
            if freed > 0:
                print(f"[ResourceManager] CPU 内存清理: 释放 {freed:.1f} MB")
    
    def cleanup_all(self):
        """清理所有资源"""
        if self.verbose:
            print("[ResourceManager] 开始清理资源...")
        
        self.cleanup_cpu()
        self.cleanup_gpu()
        
        if self.verbose:
            self.print_memory_info("清理后内存状态:")
    
    @contextmanager
    def track(self, operation: str = "操作"):
        """
        上下文管理器：跟踪操作的资源使用
        
        用法:
            with resource_manager.track("模型加载"):
                # 执行操作
                pass
        """
        if self.verbose:
            print(f"[ResourceManager] 开始 {operation}")
            self.print_memory_info("操作前内存状态:")
        
        try:
            yield
        finally:
            if self.auto_cleanup:
                self.cleanup_all()
            elif self.verbose:
                self.print_memory_info(f"{operation} 后内存状态:")


# 全局资源管理器实例
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager(auto_cleanup: bool = True, verbose: bool = True) -> ResourceManager:
    """获取全局资源管理器实例"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager(auto_cleanup=auto_cleanup, verbose=verbose)
    return _global_resource_manager


def cleanup_resources(verbose: bool = True):
    """快捷清理函数"""
    manager = get_resource_manager(verbose=verbose)
    manager.cleanup_all()


def print_memory_info():
    """快捷打印内存信息"""
    manager = get_resource_manager(verbose=True)
    manager.print_memory_info("当前内存状态:")


# 装饰器：自动清理资源
def auto_cleanup(func):
    """
    装饰器：函数执行后自动清理资源
    
    用法:
        @auto_cleanup
        def my_function():
            # 执行操作
            pass
    """
    def wrapper(*args, **kwargs):
        manager = get_resource_manager()
        try:
            return func(*args, **kwargs)
        finally:
            manager.cleanup_all()
    return wrapper


if __name__ == "__main__":
    # 测试资源管理器
    print("=" * 60)
    print("测试资源管理器")
    print("=" * 60)
    
    manager = ResourceManager(auto_cleanup=True, verbose=True)
    
    # 测试内存跟踪
    with manager.track("测试操作"):
        # 模拟一些内存分配
        data = [i for i in range(1000000)]
        
        if torch.cuda.is_available():
            tensor = torch.randn(1000, 1000).cuda()
    
    print("\n测试完成！")

