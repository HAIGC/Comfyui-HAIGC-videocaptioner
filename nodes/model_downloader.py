"""
FasterWhisper 模型路径管理
"""

import os
from pathlib import Path
from typing import Optional

def get_model_path(model_name: str, models_dir: str) -> Path:
    """
    获取模型路径
    
    Args:
        model_name: 模型名称（如 tiny, base, small, medium, large-v2, large-v3）
        models_dir: 模型目录
    
    Returns:
        模型文件路径
    """
    # FasterWhisper 模型目录结构
    # models/whisper/faster-whisper-{model_name}/
    # 特殊模型名称映射
    if model_name == "belle-large-v3-zh-punct":
        folder_name = "Belle-whisper-large-v3-zh-punct-ct2-float32"
    else:
        folder_name = f"faster-whisper-{model_name}"
        
    model_dir = Path(models_dir) / folder_name
    return model_dir


def is_model_downloaded(model_name: str, models_dir: str) -> bool:
    """
    检查模型是否已存在
    
    Args:
        model_name: 模型名称
        models_dir: 模型目录
    
    Returns:
        是否已存在
    """
    model_path = get_model_path(model_name, models_dir)
    
    # 检查模型目录是否存在且包含必要文件
    if not model_path.exists():
        return False
    
    # FasterWhisper 模型至少需要这些文件
    # 检查是否有模型文件
    has_model_file = (
        (model_path / "model.bin").exists() or
        (model_path / "model.pt").exists() or
        any(model_path.glob("model*.bin"))
    )
    
    has_config = (model_path / "config.json").exists()
    
    return has_model_file and has_config


def ensure_model_available(model_name: str, models_dir: str, auto_download: bool = False, callback=None) -> bool:
    """
    检查模型是否可用 (不进行自动下载)
    
    Args:
        model_name: 模型名称
        models_dir: 模型目录
        auto_download: (已弃用，强制为False)
        callback: (已弃用)
    
    Returns:
        模型是否可用
    """
    # 检查模型是否已存在
    if is_model_downloaded(model_name, models_dir):
        print(f"[VideoCaptioner] 模型已存在: {model_name}")
        return True
    
    print(f"[VideoCaptioner] 模型不存在: {model_name}")
    print(f"[VideoCaptioner] 请将模型放置在: {get_model_path(model_name, models_dir)}")
    return False


def get_available_models(models_dir: str) -> list:
    """
    获取已存在的模型列表
    
    Args:
        models_dir: 模型目录
    
    Returns:
        已存在的模型名称列表
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    available = []
    # 检查标准 faster-whisper- 前缀文件夹
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            if model_dir.name.startswith("faster-whisper-"):
                model_name = model_dir.name.replace("faster-whisper-", "")
                if is_model_downloaded(model_name, models_dir):
                    available.append(model_name)
            # 检查特殊模型文件夹
            elif model_dir.name == "Belle-whisper-large-v3-zh-punct-ct2-float32":
                if is_model_downloaded("belle-large-v3-zh-punct", models_dir):
                    available.append("belle-large-v3-zh-punct")
    
    return available
