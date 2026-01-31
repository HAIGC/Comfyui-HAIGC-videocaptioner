"""
配置节点
用于配置转录、LLM、翻译等参数
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

# 设置路径并导入
current_dir = Path(__file__).parent.parent
videocaptioner_path = current_dir / "VideoCaptioner"
if str(videocaptioner_path) not in sys.path:
    sys.path.insert(0, str(videocaptioner_path))

# 导入基础模块
try:
    from app.core.entities import (
        TranscribeConfig,
        TranscribeModelEnum,
        FasterWhisperModelEnum,
        VadMethodEnum,
    )
    
    DEPENDENCIES_OK = True
    print("[VideoCaptioner] ConfigNodes dependencies loaded successfully")
except Exception as e:
    print(f"[VideoCaptioner] ConfigNodes import error: {e}")
    import traceback
    traceback.print_exc()
    DEPENDENCIES_OK = False
    TranscribeConfig = None
    TranscribeModelEnum = None
    FasterWhisperModelEnum = None
    VadMethodEnum = None

# 获取 ComfyUI 的 models 目录
def get_comfyui_models_dir():
    """获取 ComfyUI 标准 models 目录"""
    comfyui_root = Path(__file__).parent.parent.parent.parent
    models_dir = comfyui_root / "models" / "whisper"
    models_dir.mkdir(parents=True, exist_ok=True)
    return str(models_dir)


class TranscribeConfigNode:
    """
    转录配置节点
    配置语音识别的详细参数
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "转录模型": ([
                    "Whisper",
                    "J接口",
                    "B接口",
                ], {
                    "default": "Whisper"
                }),
                "语言": ([
                    "auto", "zh", "en", "ja", "ko", "fr", "de", "es", "ru",
                ], {
                    "default": "zh"
                }),
                "模型大小": ([
                    "belle-large-v3-zh-punct"
                ], {
                    "default": "belle-large-v3-zh-punct"
                }),
                "使用缓存": ("BOOLEAN", {"default": True}),
                "词级时间戳": ("BOOLEAN", {"default": False}),
                "语音检测过滤": ("BOOLEAN", {"default": False}),
                "语音检测阈值": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "提示模式": ([
                    "自动优化 🎯",
                    "中英混合 🌐", 
                    "纯中文 🇨🇳",
                    "纯英文 🇺🇸",
                    "关闭 ❌"
                ], {
                    "default": "关闭 ❌"
                }),
            },
            "optional": {
                "设备": (["cpu", "cuda"], {"default": "cpu"}),  # 默认 CPU 避免需要 faster-whisper-xxl
                "语音检测方法": ([
                    "silero_v4_fw",
                    "silero_v3",
                    "pyannote_v3",
                ], {
                    "default": "silero_v4_fw"
                }),
            }
        }
    
    RETURN_TYPES = ("TRANSCRIBE_CONFIG",)
    RETURN_NAMES = ("转录配置",)
    FUNCTION = "create_config"
    CATEGORY = "video/subtitle/config"
    
    def create_config(
        self,
        转录模型: str,
        语言: str,
        模型大小: str,
        使用缓存: bool,
        词级时间戳: bool,
        语音检测过滤: bool,
        语音检测阈值: float,
        提示模式: str,
        设备: str = "cpu",  # 默认 CPU
        语音检测方法: str = "silero_v4_fw",
        **kwargs,
    ) -> Tuple[Dict[str, Any]]:
        """
        创建转录配置
        
        Returns:
            (transcribe_config,): 转录配置字典
        """
        # 检查依赖是否加载成功
        if not DEPENDENCIES_OK or TranscribeModelEnum is None:
            error_msg = "依赖加载失败！请检查 VideoCaptioner 安装是否完整"
            print(f"[VideoCaptioner] {error_msg}")
            return ({"error": error_msg},)
        
        # 映射模型名称
        model_mapping = {
            "Whisper": TranscribeModelEnum.FASTER_WHISPER,
            "J接口": TranscribeModelEnum.JIANYING,
            "B接口": TranscribeModelEnum.BIJIAN,
        }
        
        # Whisper 原版模型映射
        from app.core.entities import WhisperModelEnum
        whisper_model_mapping = {
            "tiny": WhisperModelEnum.TINY,
            "base": WhisperModelEnum.BASE,
            "small": WhisperModelEnum.SMALL,
            "medium": WhisperModelEnum.MEDIUM,
            "turbo": WhisperModelEnum.TURBO,
            "large-v2": WhisperModelEnum.LARGE_V2,
            "large-v3": WhisperModelEnum.LARGE_V3,
        }
        
        # FasterWhisper 模型映射
        faster_whisper_model_mapping = {
            "tiny": FasterWhisperModelEnum.TINY,
            "base": FasterWhisperModelEnum.BASE,
            "small": FasterWhisperModelEnum.SMALL,
            "medium": FasterWhisperModelEnum.MEDIUM,
            "turbo": FasterWhisperModelEnum.LARGE_V3_TURBO,
            "large-v2": FasterWhisperModelEnum.LARGE_V2,
            "large-v3": FasterWhisperModelEnum.LARGE_V3,
            "belle-large-v3-zh-punct": FasterWhisperModelEnum.BELLE_LARGE_V3_ZH_PUNCT,
        }
        
        vad_method_mapping = {
            "silero_v4_fw": VadMethodEnum.SILERO_V4_FW,
            "silero_v3": VadMethodEnum.SILERO_V3,
            "pyannote_v3": VadMethodEnum.PYANNOTE_V3,
        }
        
        # 获取 ComfyUI models 目录
        models_dir = get_comfyui_models_dir()
        
        # 创建配置对象
        config = TranscribeConfig(
            transcribe_model=model_mapping.get(转录模型),
            transcribe_language=语言,
            whisper_model=whisper_model_mapping.get(模型大小),
            faster_whisper_model=faster_whisper_model_mapping.get(模型大小),
            use_asr_cache=使用缓存,
            need_word_time_stamp=词级时间戳,
            faster_whisper_device=设备,
            faster_whisper_model_dir=models_dir,  # 复用统一模型目录
            faster_whisper_vad_filter=语音检测过滤,
            faster_whisper_vad_threshold=语音检测阈值,
            faster_whisper_vad_method=vad_method_mapping.get(语音检测方法),
            faster_whisper_prompt=None,
        )
        
        # 生成 initial_prompt（提示词）
        initial_prompt = self._generate_prompt(提示模式, 语言)
        
        # 将提示词保存到配置对象，便于通用转录路径使用
        try:
            setattr(config, "faster_whisper_prompt", initial_prompt or None)
        except Exception:
            pass
        
        # 转换为字典以便传递
        config_dict = {
            "config_object": config,
            "transcribe_model": 转录模型,
            "language": 语言,
            "whisper_model": 模型大小,
            "prompt_mode": 提示模式,
            "initial_prompt": initial_prompt,
        }
        
        print(f"[TranscribeConfig] 识别提示模式: {提示模式}")
        if initial_prompt:
            print(f"[TranscribeConfig] 提示词: {initial_prompt[:100]}...")
        
        return (config_dict,)
    
    def _generate_prompt(self, prompt_mode: str, language: str) -> str:
        """
        根据提示模式生成 initial_prompt
        
        initial_prompt 是 Whisper 的一个重要参数，可以：
        1. 提示模型保留专有名词（如 ComfyUI, OpenAI 等）
        2. 指定输出格式和风格
        3. 提高识别准确率
        
        Args:
            prompt_mode: 提示模式
            language: 语言代码
            
        Returns:
            initial_prompt 字符串
        """
        if "关闭" in prompt_mode:
            return ""
        
        if "自动优化" in prompt_mode:
            # 根据语言自动选择
            if language == "zh":
                return "以下是普通话的句子，请保留英文专有名词原文，如 ComfyUI, Stable Diffusion, Python, AI, GPU, CPU, API 等技术术语。"
            elif language == "en":
                return "The following is a transcript in English. Preserve proper nouns and technical terms."
            elif language == "auto":
                return "以下是中英文混合内容。对于中文部分使用简体中文，对于英文专有名词和技术术语保留原文，如 ComfyUI, API, GPU, Python, Stable Diffusion, OpenAI, ChatGPT 等。"
            else:
                return ""
        
        if "中英混合" in prompt_mode:
            return "以下是中英文混合内容。对于中文部分使用简体中文，对于英文专有名词和技术术语保留原文，如 ComfyUI, API, GPU, Python, Stable Diffusion, OpenAI, ChatGPT, CUDA, PyTorch, TensorFlow, Node, Workflow 等。不要将英文翻译成中文同音字。"
        
        if "纯中文" in prompt_mode:
            return "以下是普通话的句子。"
        
        if "纯英文" in prompt_mode:
            return "The following is a transcript in English."
        
        return ""


class LLMConfigNode:
    """
    LLM 配置节点
    配置大语言模型的参数
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("STRING", {
                    "default": "gpt-4o-mini",
                }),
                "API地址": ("STRING", {
                    "default": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                }),
                "API密钥": ("STRING", {
                    "default": os.getenv("OPENAI_API_KEY", ""),
                }),
                "温度": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
                "线程数": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                }),
            },
        }
    
    RETURN_TYPES = ("LLM_CONFIG",)
    RETURN_NAMES = ("LLM配置",)
    FUNCTION = "create_config"
    CATEGORY = "video/subtitle/config"
    
    def create_config(
        self,
        模型: str,
        API地址: str,
        API密钥: str,
        温度: float,
        线程数: int,
    ) -> Tuple[Dict[str, Any]]:
        """
        创建 LLM 配置
        
        Returns:
            (llm_config,): LLM 配置字典
        """
        # 设置环境变量
        if API地址:
            os.environ["OPENAI_BASE_URL"] = API地址
        if API密钥:
            os.environ["OPENAI_API_KEY"] = API密钥
        
        config = {
            "model": 模型,
            "base_url": API地址,
            "api_key": API密钥,
            "temperature": 温度,
            "thread_num": 线程数,
        }
        
        return (config,)


class TranslateConfigNode:
    """
    翻译配置节点
    配置翻译相关参数
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                "反思翻译": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "LLM配置": ("LLM_CONFIG",),
            }
        }
    
    RETURN_TYPES = ("TRANSLATE_CONFIG",)
    RETURN_NAMES = ("翻译配置",)
    FUNCTION = "create_config"
    CATEGORY = "video/subtitle/config"
    
    def create_config(
        self,
        翻译器类型: str,
        目标语言: str,
        线程数: int,
        批处理数量: int,
        反思翻译: bool,
        LLM配置: Dict[str, Any] = None,
    ) -> Tuple[Dict[str, Any]]:
        """
        创建翻译配置
        
        Returns:
            (translate_config,): 翻译配置字典
        """
        config = {
            "translator_type": 翻译器类型,
            "target_language": 目标语言,
            "thread_num": 线程数,
            "batch_num": 批处理数量,
            "is_reflect": 反思翻译,
            "llm_config": LLM配置,
        }
        
        return (config,)


NODE_CLASS_MAPPINGS = {
    "TranscribeConfigNode": TranscribeConfigNode,
    "LLMConfigNode": LLMConfigNode,
    "TranslateConfigNode": TranslateConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TranscribeConfigNode": "转录配置",
    "LLMConfigNode": "LLM 配置",
    "TranslateConfigNode": "翻译配置",
}

