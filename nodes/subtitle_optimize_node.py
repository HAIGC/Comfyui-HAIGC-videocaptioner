"""
字幕优化节点
使用 LLM 对字幕内容进行优化校正（错别字、标点符号、格式等）
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Any, Dict

# 导入基础模块
try:
    from .base_node import setup_videocaptioner_path
    setup_videocaptioner_path()
    
    from app.core.subtitle_processor.optimize import SubtitleOptimizer
    
    DEPENDENCIES_OK = True
except Exception as e:
    print(f"[VideoCaptioner] SubtitleOptimize import error: {e}")
    DEPENDENCIES_OK = False
    SubtitleOptimizer = None


class SubtitleOptimizeNode:
    """
    字幕优化节点
    使用 LLM 对字幕进行校正优化，包括错别字、标点符号、专业术语等
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "优化类型": ([
                    "全面优化",
                    "仅校正错误",
                    "仅优化格式",
                ], {
                    "default": "全面优化"
                }),
                "线程数": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                }),
                "批处理数量": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                }),
            },
            "optional": {
                "字幕数据": ("SUBTITLE_DATA",),
                "文本内容": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "在此输入要优化的文本内容...",
                }),
                "LLM配置": ("LLM_CONFIG",),
                "自定义提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "术语表": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }
    
    RETURN_TYPES = ("SUBTITLE_DATA", "STRING",)
    RETURN_NAMES = ("优化后字幕数据", "优化后文本",)
    FUNCTION = "optimize_subtitle"
    CATEGORY = "video/subtitle"
    
    def optimize_subtitle(
        self,
        优化类型: str,
        线程数: int,
        批处理数量: int,
        字幕数据: Any = None,
        文本内容: str = "",
        LLM配置: Dict[str, Any] = None,
        自定义提示词: str = "",
        术语表: str = "",
    ) -> Tuple[Any, str]:
        """
        优化字幕内容
        
        Args:
            optimize_type: 优化类型
            thread_num: 线程数
            batch_num: 批处理数量
            subtitle_data: 字幕数据对象（ASRData）- 可选
            text_content: 文本内容 - 可选，如果未提供字幕数据则使用此文本
            llm_config: LLM 配置
            custom_prompt: 自定义优化提示词
            terminology: 术语表（每行一个术语或 错误->正确 的格式）
            
        Returns:
            (optimized_subtitle_data, optimized_text): 优化后的字幕数据和文本
        """
        try:
            # 检查输入
            if not 字幕数据 and not 文本内容.strip():
                raise ValueError("请提供字幕数据或文本内容")
            
            # 如果只提供了文本内容，创建临时字幕数据对象
            if not 字幕数据 and 文本内容.strip():
                print(f"[VideoCaptioner] 使用文本输入模式")
                字幕数据 = self._text_to_subtitle_data(文本内容)
            
            # 检查 LLM 环境变量
            self._check_llm_env()
            
            # 获取 LLM 配置
            if LLM配置:
                model = LLM配置.get("model", "gpt-4o-mini")
                temperature = LLM配置.get("temperature", 0.3)
            else:
                model = "gpt-4o-mini"
                temperature = 0.3
            
            # 构建优化提示词
            optimize_prompts = {
                "全面优化": "请全面优化字幕内容：修正错别字、优化标点符号、统一专业术语、改善格式。",
                "仅校正错误": "请仅修正字幕中的错别字和明显错误，保持原文风格。",
                "仅优化格式": "请优化字幕的标点符号和格式，不改变文字内容。",
            }
            
            base_prompt = optimize_prompts.get(优化类型, optimize_prompts["全面优化"])
            
            # 添加术语表到提示词
            if 术语表:
                base_prompt += f"\n\n术语表：\n{术语表}"
            
            # 添加自定义提示词
            if 自定义提示词:
                base_prompt += f"\n\n{自定义提示词}"
            
            print(f"[VideoCaptioner] 开始优化字幕，优化类型: {优化类型}")
            
            # 创建优化器
            optimizer = SubtitleOptimizer(
                thread_num=线程数,
                batch_num=批处理数量,
                model=model,
                temperature=temperature,
                custom_prompt=base_prompt,
            )
            
            # 执行优化
            optimized_data = optimizer.optimize_subtitle(字幕数据)
            
            # 转换为文本
            optimized_text = optimized_data.to_txt()
            
            print(f"[VideoCaptioner] 优化完成，共 {len(optimized_data.segments)} 个字幕段")
            
            # 清理资源
            optimizer.stop()
            
            return (optimized_data, optimized_text)
            
        except Exception as e:
            print(f"[VideoCaptioner] 优化失败: {str(e)}")
            raise RuntimeError(f"字幕优化失败: {str(e)}")
    
    def _text_to_subtitle_data(self, text: str) -> Any:
        """
        将纯文本转换为字幕数据对象
        
        Args:
            text: 输入文本
            
        Returns:
            ASRData 对象
        """
        from app.core.bk_asr.asr_data import ASRData, ASRDataSeg
        
        # 按行分割文本
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 创建字幕段（每行作为一个段）
        segments = []
        current_time = 0
        
        for line in lines:
            # 估算时长：每个字符约 200ms（中文）或 100ms（英文）
            char_count = len(line)
            # 简单判断：如果有中文字符，使用中文时长
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in line)
            duration = char_count * (200 if has_chinese else 100)
            
            seg = ASRDataSeg(
                text=line,
                start_time=current_time,
                end_time=current_time + duration
            )
            segments.append(seg)
            current_time += duration + 500  # 段间间隔 500ms
        
        # 创建 ASRData 对象
        asr_data = ASRData(segments=segments)
        
        print(f"[VideoCaptioner] 文本转换完成: {len(segments)} 段")
        return asr_data
    
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
    "SubtitleOptimizeNode": SubtitleOptimizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleOptimizeNode": "字幕优化"
}

