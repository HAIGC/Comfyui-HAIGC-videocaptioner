"""
工具节点
提供视频加载等实用功能
"""

import os
import sys
from pathlib import Path
from typing import Tuple

# 导入基础模块
try:
    from .base_node import setup_videocaptioner_path
    setup_videocaptioner_path()
    
    DEPENDENCIES_OK = True
except Exception as e:
    print(f"[VideoCaptioner] UtilityNodes import error: {e}")
    DEPENDENCIES_OK = False


class LoadVideoNode:
    """
    加载视频节点
    从文件路径加载视频
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频路径": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("视频路径", "视频信息",)
    FUNCTION = "load_video"
    CATEGORY = "video/subtitle/utility"
    
    def load_video(self, 视频路径: str) -> Tuple[str, str]:
        """
        加载视频并返回视频信息
        
        Args:
            视频路径: 视频文件路径
            
        Returns:
            (video_path, video_info): 视频路径和视频信息
        """
        if not os.path.exists(视频路径):
            raise FileNotFoundError(f"视频文件不存在: {视频路径}")
        
        # 获取视频信息
        file_size = os.path.getsize(视频路径) / (1024 * 1024)  # MB
        video_info = f"视频路径: {视频路径}\n文件大小: {file_size:.2f} MB"
        
        print(f"[VideoCaptioner] 加载视频: {视频路径}")
        print(video_info)
        
        return (视频路径, video_info)


NODE_CLASS_MAPPINGS = {
    "LoadVideoNode": LoadVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoNode": "加载视频",
}

