"""
字幕翻译节点
支持多种翻译引擎：LLM 翻译、DeepLX、微软翻译、谷歌翻译
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Any, Dict

# 导入基础模块
try:
    from .base_node import setup_videocaptioner_path
    setup_videocaptioner_path()
    
    from app.core.subtitle_processor.translate import (
        TranslatorFactory,
        TranslatorType,
    )
    
    DEPENDENCIES_OK = True
except Exception as e:
    print(f"[VideoCaptioner] SubtitleTranslate import error: {e}")
    DEPENDENCIES_OK = False
    TranslatorFactory = None
    TranslatorType = None


class SubtitleTranslateNode:
    """
    字幕翻译节点
    支持多种翻译引擎和反思翻译功能
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "字幕数据": ("SUBTITLE_DATA",),
                "翻译器类型": ([
                    "LLM 大模型翻译",
                    "DeepLx 翻译",
                    "微软翻译",
                    "谷歌翻译",
                ], {
                    "default": "LLM 大模型翻译"
                }),
                "目标语言": ([
                    "简体中文", "繁体中文", "英语", "日本語", "韩语",
                    "粤语", "法语", "德语", "西班牙语", "俄语", 
                    "葡萄牙语", "土耳其语"
                ], {
                    "default": "简体中文"
                }),
                "线程数": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                }),
                "批处理数量": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                }),
            },
            "optional": {
                "LLM配置": ("LLM_CONFIG",),
                "自定义提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "反思翻译": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("SUBTITLE_DATA", "STRING",)
    RETURN_NAMES = ("翻译后字幕数据", "翻译后文本",)
    FUNCTION = "translate_subtitle"
    CATEGORY = "video/subtitle"
    
    def translate_subtitle(
        self,
        字幕数据: Any,
        翻译器类型: str,
        目标语言: str,
        线程数: int,
        批处理数量: int,
        LLM配置: Dict[str, Any] = None,
        自定义提示词: str = "",
        反思翻译: bool = False,
    ) -> Tuple[Any, str]:
        """
        翻译字幕
        
        Args:
            subtitle_data: 字幕数据对象（ASRData）
            translator_type: 翻译器类型
            target_language: 目标语言
            thread_num: 线程数
            batch_num: 批处理数量
            llm_config: LLM 配置（用于 LLM 翻译）
            custom_prompt: 自定义提示词
            is_reflect: 是否使用反思翻译
            
        Returns:
            (translated_subtitle_data, translated_text): 翻译后的字幕数据和文本
        """
        try:
            # 映射翻译器类型
            translator_mapping = {
                "LLM 大模型翻译": TranslatorType.OPENAI,
                "DeepLx 翻译": TranslatorType.DEEPLX,
                "微软翻译": TranslatorType.BING,
                "谷歌翻译": TranslatorType.GOOGLE,
            }
            
            translator_enum = translator_mapping.get(翻译器类型, TranslatorType.OPENAI)
            
            # 如果使用 LLM 翻译，检查环境变量
            if translator_enum == TranslatorType.OPENAI:
                self._check_llm_env()
            
            # 获取 LLM 配置
            if LLM配置 and translator_enum == TranslatorType.OPENAI:
                model = LLM配置.get("model", "gpt-4o-mini")
                temperature = LLM配置.get("temperature", 0.7)
            else:
                model = "gpt-4o-mini"
                temperature = 0.7
            
            # 创建翻译器
            print(f"[VideoCaptioner] 开始翻译，翻译器: {翻译器类型}, 目标语言: {目标语言}")
            
            translator = TranslatorFactory.create_translator(
                translator_type=translator_enum,
                thread_num=线程数,
                batch_num=批处理数量,
                target_language=目标语言,
                model=model,
                custom_prompt=自定义提示词,
                temperature=temperature,
                is_reflect=反思翻译,
            )
            
            # 执行翻译
            translated_data = translator.translate_subtitle(字幕数据)
            
            # 提取翻译后的文本
            translated_text = "\n".join([
                seg.translated_text if seg.translated_text else seg.text
                for seg in translated_data.segments
            ])
            
            print(f"[VideoCaptioner] 翻译完成，共 {len(translated_data.segments)} 个字幕段")
            
            # 清理资源
            translator.stop()
            
            return (translated_data, translated_text)
            
        except Exception as e:
            print(f"[VideoCaptioner] 翻译失败: {str(e)}")
            raise RuntimeError(f"字幕翻译失败: {str(e)}")
    
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
    "SubtitleTranslateNode": SubtitleTranslateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleTranslateNode": "字幕翻译"
}

