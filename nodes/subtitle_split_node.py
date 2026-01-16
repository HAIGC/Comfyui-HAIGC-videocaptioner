"""
字幕智能断句节点
使用 LLM 大模型进行智能断句，提升字幕观看体验
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Any, Dict

# 导入基础模块
try:
    from .base_node import setup_videocaptioner_path
    setup_videocaptioner_path()
    
    from app.core.subtitle_processor.split import SubtitleSplitter
    
    DEPENDENCIES_OK = True
except Exception as e:
    print(f"[VideoCaptioner] SubtitleSplit import error: {e}")
    DEPENDENCIES_OK = False
    SubtitleSplitter = None


class SubtitleSplitNode:
    """
    字幕智能断句节点
    使用 LLM 对逐字字幕进行智能断句，使其符合自然语言习惯
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "字幕数据": ("SUBTITLE_DATA",),
                "分段类型": (["语义分段", "句子分段"], {
                    "default": "语义分段"
                }),
                "中日韩最大字数": ("INT", {
                    "default": 25,
                    "min": 10,
                    "max": 50,
                    "step": 1,
                }),
                "英文最大单词数": ("INT", {
                    "default": 18,
                    "min": 8,
                    "max": 30,
                    "step": 1,
                }),
                "使用缓存": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "LLM配置": ("LLM_CONFIG",),
            }
        }
    
    RETURN_TYPES = ("SUBTITLE_DATA", "STRING",)
    RETURN_NAMES = ("断句后字幕数据", "断句后文本",)
    FUNCTION = "split_subtitle"
    CATEGORY = "video/subtitle"
    
    def split_subtitle(
        self,
        字幕数据: Any,
        分段类型: str,
        中日韩最大字数: int,
        英文最大单词数: int,
        使用缓存: bool,
        LLM配置: Dict[str, Any] = None,
    ) -> Tuple[Any, str]:
        """
        智能断句处理
        
        Args:
            subtitle_data: 字幕数据对象（ASRData）
            split_type: 分段类型（语义分段/句子分段）
            max_word_count_cjk: 中日韩文本最大字数
            max_word_count_english: 英文最大单词数
            use_cache: 是否使用缓存
            llm_config: LLM 配置
            
        Returns:
            (split_subtitle_data, split_subtitle_text): 断句后的字幕数据和文本
        """
        try:
            # 确保环境变量已设置
            self._check_llm_env()
            
            # 获取 LLM 配置
            if LLM配置:
                model = LLM配置.get("model", "gpt-4o-mini")
                temperature = LLM配置.get("temperature", 0.4)
                thread_num = LLM配置.get("thread_num", 5)
            else:
                model = "gpt-4o-mini"
                temperature = 0.4
                thread_num = 5
            
            # 映射分段类型
            split_type_mapping = {
                "语义分段": "semantic",
                "句子分段": "sentence",
            }
            
            # 创建字幕分割器
            if SubtitleSplitter is None:
                raise RuntimeError("缺少 openai 依赖或相关模块未加载，请安装 openai 并重试")
            splitter = SubtitleSplitter(
                thread_num=thread_num,
                model=model,
                temperature=temperature,
                timeout=60,
                retry_times=2,
                split_type=split_type_mapping.get(分段类型, "semantic"),
                max_word_count_cjk=中日韩最大字数,
                max_word_count_english=英文最大单词数,
                use_cache=使用缓存,
            )
            
            print(f"[VideoCaptioner] 开始智能断句，分段类型: {分段类型}")
            
            # 执行断句
            split_data = splitter.split_subtitle(字幕数据)
            
            # 转换为文本
            split_text = split_data.to_txt()
            
            print(f"[VideoCaptioner] 断句完成，共 {len(split_data.segments)} 个字幕段")
            
            # 清理资源
            splitter.stop()
            
            return (split_data, split_text)
            
        except Exception as e:
            print(f"[VideoCaptioner] 断句失败: {str(e)}")
            raise RuntimeError(f"字幕断句失败: {str(e)}")
    
    def _check_llm_env(self):
        """检查 LLM 环境变量是否设置"""
        if not os.getenv("OPENAI_BASE_URL"):
            raise ValueError(
                "未设置 OPENAI_BASE_URL 环境变量\n"
                "请设置环境变量: OPENAI_BASE_URL 和 OPENAI_API_KEY"
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "未设置 OPENAI_API_KEY 环境变量\n"
                "请设置环境变量: OPENAI_BASE_URL 和 OPENAI_API_KEY"
            )


NODE_CLASS_MAPPINGS = {
    "SubtitleSplitNode": SubtitleSplitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleSplitNode": "字幕智能断句"
}

