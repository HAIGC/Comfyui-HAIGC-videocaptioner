"""
视频转录节点 - 语音识别
支持多种 ASR 引擎：FasterWhisper、WhisperAPI、剪映等
"""

import os
import gc
import torch
import numpy as np
from typing import Tuple, Dict, Any
import sys
from pathlib import Path

# 导入基础模块
try:
    print(f"[VideoCaptioner] 开始导入依赖...")
    from .base_node import setup_videocaptioner_path, check_dependencies
    print(f"[VideoCaptioner] base_node 导入成功")
    
    setup_videocaptioner_path()
    print(f"[VideoCaptioner] VideoCaptioner 路径设置成功")
    
    check_dependencies()
    print(f"[VideoCaptioner] 依赖检查通过")
    
    # 导入 VideoCaptioner 模块
    print(f"[VideoCaptioner] 导入 VideoCaptioner 核心模块...")
    from app.core.bk_asr.transcribe import transcribe
    print(f"[VideoCaptioner] transcribe 导入成功")
    
    from app.core.bk_asr.asr_data import ASRData, ASRDataSeg
    print(f"[VideoCaptioner] ASRData 导入成功")
    
    from app.core.entities import TranscribeConfig, TranscribeModelEnum, FasterWhisperModelEnum, VadMethodEnum
    print(f"[VideoCaptioner] entities 导入成功")
    
    # 导入 Python 实现
    print(f"[VideoCaptioner] 导入 FasterWhisper Python 实现...")
    from .faster_whisper_python import FasterWhisperPython, FASTER_WHISPER_AVAILABLE
    print(f"[VideoCaptioner] FasterWhisperPython 导入成功 (可用: {FASTER_WHISPER_AVAILABLE})")
    
    # 导入资源管理器
    from .resource_manager import cleanup_resources
    print(f"[VideoCaptioner] cleanup_resources 导入成功")
    
    DEPENDENCIES_OK = True
    print(f"[VideoCaptioner] 所有依赖导入成功！")
except Exception as e:
    import traceback
    print(f"[VideoCaptioner] VideoTranscribe import error: {e}")
    print(f"[VideoCaptioner] 详细错误信息:")
    traceback.print_exc()
    DEPENDENCIES_OK = False
    FASTER_WHISPER_AVAILABLE = False
    # 创建占位符类
    TranscribeConfig = None
    TranscribeModelEnum = None
    FasterWhisperModelEnum = None
    VadMethodEnum = None
    FasterWhisperPython = None
    ASRData = None
    ASRDataSeg = None
    cleanup_resources = None

# 获取 ComfyUI 的 models 目录
def get_comfyui_models_dir():
    """获取 ComfyUI 标准 models 目录"""
    # 从当前文件路径向上查找 ComfyUI 根目录
    comfyui_root = Path(__file__).parent.parent.parent.parent
    models_dir = comfyui_root / "models" / "whisper"
    models_dir.mkdir(parents=True, exist_ok=True)
    return str(models_dir)


class VideoTranscribeNode:
    """
    视频转录节点 - 使用 ASR 引擎将视频音频转换为字幕文本
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO",),  # 音频输入（必选）
                "转录配置": ("TRANSCRIBE_CONFIG",),  # 转录配置（必选）
            },
            "optional": {
                # === 后处理功能（转录节点特有） ===
                "最小置信度": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "过滤低于此置信度的识别结果（0=不过滤）"
                }),
                "时间戳精度": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 6,
                    "tooltip": "时间戳小数位数（毫秒精度：3位，微秒：6位）"
                }),
                "合并阈值": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "自动合并间隔小于此秒数的片段（0=不合并）"
                }),
                "包含元数据": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "在输出中包含置信度、说话人等元数据"
                }),
                "裁剪静音": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "自动裁剪开头和结尾的静音"
                }),
                "过滤关键词": ("STRING", {
                    "default": "",
                    "tooltip": "逗号分隔，包含任一关键词的片段将被删除"
                }),
            }
        }
    
    RETURN_TYPES = ("SUBTITLE_DATA", "STRING",)
    RETURN_NAMES = (
        "字幕数据",                    # 完整字幕对象
        "字幕文本",                    # 纯文本（基础输出）
    )
    FUNCTION = "transcribe_video"
    CATEGORY = "video/subtitle"
    
    def transcribe_video(
        self,
        音频: Any,
        转录配置: Dict[str, Any],
        最小置信度: float = 0.0,
        时间戳精度: int = 3,
        合并阈值: float = 0.0,
        包含元数据: bool = False,
        裁剪静音: bool = False,
        过滤关键词: str = "",
    ) -> Tuple[Any, str]:
        """
        转录视频音频为字幕
        
        Args:
            audio: 音频数据（必选）
            transcribe_config: 转录配置对象（必选）
            min_confidence: 最小置信度阈值（过滤低质量结果）
            timestamp_precision: 时间戳精度（小数位数）
            merge_threshold: 自动合并间隔阈值（秒）
            include_metadata: 是否包含元数据（置信度等）
            trim_silence: 是否裁剪首尾静音
            transcribe_config: 可选的详细配置
            
        Returns:
            (subtitle_data, subtitle_text): 字幕数据和纯文本
        """
        import tempfile
        cleanup_audio = False
        audio_path = None
        
        try:
            # 检查依赖是否正常加载
            if not DEPENDENCIES_OK or TranscribeModelEnum is None:
                raise RuntimeError(
                    "视频转录节点依赖加载失败。\n"
                    "请确保已安装所有依赖：\n"
                    "  pip install sqlalchemy json-repair\n"
                    "并检查 VideoCaptioner 目录是否存在。"
                )
            
            # 处理输入：优先使用 AUDIO，其次使用文件路径
            if 音频 is None:
                raise ValueError("必须提供 audio 音频数据")
            
            # 从音频数据创建临时文件
            print(f"[VideoCaptioner] 从音频数据创建临时文件...")
            audio_path = self._audio_to_file(音频)
            cleanup_audio = True  # 音频临时文件需要清理
            
            # 解析配置对象（必选）
            # ComfyUI 会将自定义类型序列化为字典
            if isinstance(转录配置, dict):
                # 如果字典中有 config_object 键，直接使用它
                if "config_object" in 转录配置:
                    config = 转录配置["config_object"]
                    # 修复序列化后的枚举类型
                    # ComfyUI 可能将枚举对象序列化为字符串，需要转换回枚举
                    self._fix_enum_types(config)
                else:
                    # 否则将字典转换为 TranscribeConfig 对象
                    config = TranscribeConfig(**转录配置)
            else:
                config = 转录配置
            
            # 检查并下载模型（如果需要）
            if config.transcribe_model == TranscribeModelEnum.FASTER_WHISPER:
                from .model_downloader import ensure_model_available
                
                # 获取模型名称
                model_name_map = {
                    FasterWhisperModelEnum.TINY: "tiny",
                    FasterWhisperModelEnum.BASE: "base",
                    FasterWhisperModelEnum.SMALL: "small",
                    FasterWhisperModelEnum.MEDIUM: "medium",
                    FasterWhisperModelEnum.LARGE_V2: "large-v2",
                    FasterWhisperModelEnum.LARGE_V3: "large-v3",
                    FasterWhisperModelEnum.LARGE_V3_TURBO: "large-v3-turbo",
                    FasterWhisperModelEnum.BELLE_LARGE_V3_ZH_PUNCT: "belle-large-v3-zh-punct",
                }
                
                # 调试输出：打印实际的 config.faster_whisper_model 类型和值
                print(f"[VideoCaptioner] DEBUG: config.faster_whisper_model type: {type(config.faster_whisper_model)}")
                print(f"[VideoCaptioner] DEBUG: config.faster_whisper_model value: {config.faster_whisper_model}")
                
                model_name = model_name_map.get(
                    config.faster_whisper_model, 
                    "medium"
                )
                
                # 如果映射失败，再次尝试通过字符串匹配（兜底逻辑）
                if model_name == "medium" and config.faster_whisper_model != FasterWhisperModelEnum.MEDIUM:
                    if str(config.faster_whisper_model) == "belle-large-v3-zh-punct" or \
                       str(config.faster_whisper_model) == "FasterWhisperModelEnum.BELLE_LARGE_V3_ZH_PUNCT":
                        model_name = "belle-large-v3-zh-punct"
                
                models_dir = config.faster_whisper_model_dir or get_comfyui_models_dir()
                
                print(f"[VideoCaptioner] 检查模型: {model_name} (文件夹路径: {models_dir})")
                
                # 确保模型可用（仅检查，不下载）
                if not ensure_model_available(model_name, models_dir):
                    print(f"[VideoCaptioner] 错误: 模型 {model_name} 不存在且禁用了自动下载")
                    print(f"[VideoCaptioner] 请将模型放置在: {models_dir}/faster-whisper-{model_name}")
                    raise RuntimeError(f"模型 {model_name} 未找到，请手动下载并放置在 models/whisper 目录")
                
            # 执行转录
            print(f"[VideoCaptioner] 开始转录: {audio_path}")
            
            def progress_callback(progress: int, message: str):
                print(f"[VideoCaptioner] 进度: {progress}% - {message}")
            
            # 如果是 FasterWhisper，使用 Python 库实现（无需外部程序）
            if config.transcribe_model == TranscribeModelEnum.FASTER_WHISPER and FASTER_WHISPER_AVAILABLE:
                print(f"[VideoCaptioner] 使用 FasterWhisper Python 库（无需外部程序）")
                
                # 获取 initial_prompt（如果有）
                initial_prompt = ""
                if isinstance(转录配置, dict):
                    initial_prompt = 转录配置.get("initial_prompt", "")
                
                # 使用 Python 库实现
                asr_data = self._transcribe_with_python_library(audio_path, config, progress_callback, initial_prompt)
            else:
                # 使用原有的 transcribe 函数（可能需要外部程序）
                asr_data = transcribe(audio_path, config, callback=progress_callback)
            
            # 保持单次转录结果，不做自动重试（稳定模式）
            
            print(f"[VideoCaptioner] 转录完成，共 {len(asr_data.segments)} 个字幕段")
            
            # === 应用后处理功能 ===
            asr_data = self._apply_postprocessing(
                asr_data,
                min_confidence=最小置信度,
                timestamp_precision=时间戳精度,
                merge_threshold=合并阈值,
                trim_silence=裁剪静音,
                remove_keywords=过滤关键词,
            )
            
            # 转换为文本
            subtitle_text = asr_data.to_txt()
            if 包含元数据:
                subtitle_text = self._add_metadata_to_text(asr_data)
            
            print(f"[VideoCaptioner] 后处理完成，最终 {len(asr_data.segments)} 个字幕段")
            
            # 自动清理资源（显存、内存）
            if cleanup_resources:
                try:
                    cleanup_resources(verbose=False)
                except Exception as e:
                    # 记录清理错误但不阻止主流程
                    print(f"[VideoCaptioner] 资源清理警告: {e}")
                    import traceback
                    traceback.print_exc()
            
            return (asr_data, subtitle_text)
            
        except Exception as e:
            print(f"[VideoCaptioner] 转录失败: {str(e)}")
            raise RuntimeError(f"视频转录失败: {str(e)}")
        finally:
            # 清理临时音频文件
            if cleanup_audio and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"[VideoCaptioner] 已清理临时音频文件")
                except Exception as e:
                    print(f"[VideoCaptioner] 临时文件清理失败: {e}")
    
    def _audio_to_file(self, audio: Any) -> str:
        """
        将音频数据保存为临时文件
        
        Args:
            audio: 音频数据（ComfyUI AUDIO 格式）
                   通常是一个字典: {"waveform": tensor, "sample_rate": int}
                   或者是一个元组: (waveform, sample_rate)
            
        Returns:
            临时音频文件路径
        """
        import tempfile
        import torchaudio
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # 处理 ComfyUI 的 AUDIO 格式
            waveform = None
            sample_rate = 16000
            
            if isinstance(audio, dict):
                waveform = audio.get("waveform")
                sample_rate = audio.get("sample_rate", 16000)
            elif isinstance(audio, (tuple, list)) and len(audio) >= 2:
                waveform, sample_rate = audio[0], audio[1]
            elif hasattr(audio, 'waveform') and hasattr(audio, 'sample_rate'):
                # 支持类似 LazyAudioMap 的对象
                waveform = audio.waveform
                sample_rate = audio.sample_rate
            elif hasattr(audio, '__getitem__'):
                # 尝试作为类字典对象访问
                try:
                    waveform = audio['waveform']
                    sample_rate = audio.get('sample_rate', 16000) if hasattr(audio, 'get') else 16000
                except (KeyError, TypeError):
                    pass
            
            if waveform is None:
                raise ValueError(f"不支持的音频格式: {type(audio)}")
            
            # 确保 waveform 是 tensor
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)
            
            # 确保形状正确 (channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # 添加 channel 维度
            elif waveform.dim() == 3:
                # 如果是 (batch, channels, samples)，取第一个 batch
                waveform = waveform[0]
            elif waveform.dim() == 2:
                # 统一为 (channels, samples)
                c, s = waveform.shape
                # 如果第二维更大且第一维很小，视为 (samples, channels)
                if c > s and s <= 8:
                    waveform = waveform.transpose(0, 1)
                    c, s = waveform.shape
                # 多通道转单通道
                if c > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
            
            # 归一化与类型修正
            if not torch.is_floating_point(waveform):
                waveform = waveform.float()
            
            # 峰值归一化 (Peak Normalization) - 确保音量足够 VAD 检测
            max_abs = waveform.abs().max()
            if torch.isfinite(max_abs) and max_abs > 0:
                # 归一化到 0.9，既防止削波也提升小音量
                waveform = waveform / max_abs * 0.9
            
            waveform = waveform.clamp(-1.0, 1.0)

            # 保存为 WAV 文件
            torchaudio.save(temp_path, waveform, sample_rate)
            
            try:
                duration_sec = float(waveform.shape[1]) / float(sample_rate)
            except Exception:
                duration_sec = 0.0
            print(f"[VideoCaptioner] 已创建临时音频: {temp_path} (采样率: {sample_rate} Hz, 时长: {duration_sec:.2f}s)")
            
            return temp_path
            
        except Exception as e:
            print(f"[VideoCaptioner] 音频保存失败: {e}")
            # 如果 torchaudio 失败，尝试使用 scipy
            try:
                import scipy.io.wavfile as wavfile
                import numpy as np
                
                waveform = None
                sample_rate = 16000
                
                if isinstance(audio, dict):
                    waveform = audio.get("waveform")
                    sample_rate = audio.get("sample_rate", 16000)
                elif isinstance(audio, (tuple, list)) and len(audio) >= 2:
                    waveform, sample_rate = audio[0], audio[1]
                elif hasattr(audio, 'waveform') and hasattr(audio, 'sample_rate'):
                    waveform = audio.waveform
                    sample_rate = audio.sample_rate
                elif hasattr(audio, '__getitem__'):
                    try:
                        waveform = audio['waveform']
                        sample_rate = audio.get('sample_rate', 16000) if hasattr(audio, 'get') else 16000
                    except (KeyError, TypeError):
                        pass
                
                if waveform is None:
                    raise ValueError(f"无法从音频对象提取 waveform: {type(audio)}")
                
                # 转换为 numpy
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.cpu().numpy()
                
                # 确保是 1D 或 2D
                if waveform.ndim == 1:
                    pass  # 已经是 1D
                elif waveform.ndim == 2:
                    # 转置为 (samples, channels) 或取第一个通道
                    if waveform.shape[0] < waveform.shape[1]:
                        waveform = waveform.T
                    if waveform.shape[1] > 1:
                        waveform = waveform[:, 0]  # 只取第一个通道
                elif waveform.ndim == 3:
                    waveform = waveform[0, 0]  # 取第一个 batch 的第一个通道
                
                # 归一化到 int16 范围
                if waveform.dtype == np.float32 or waveform.dtype == np.float64:
                    # 峰值归一化
                    max_val = np.max(np.abs(waveform))
                    if max_val > 0:
                        waveform = waveform / max_val * 0.9
                    
                    waveform = (waveform * 32767).astype(np.int16)
                
                wavfile.write(temp_path, sample_rate, waveform)
                print(f"[VideoCaptioner] 已创建临时音频 (scipy): {temp_path}")
                
                return temp_path
            except Exception as e2:
                raise RuntimeError(f"音频保存失败: {e}\n备用方法也失败: {e2}")
    
    def _fix_enum_types(self, config: Any) -> None:
        """
        修复配置对象中被序列化的枚举类型
        ComfyUI 序列化时可能将枚举转换为字符串，需要转换回枚举对象
        
        Args:
            config: TranscribeConfig 对象
        """
        # 修复 transcribe_model
        if hasattr(config, 'transcribe_model') and config.transcribe_model is not None:
            original = config.transcribe_model
            original_type = type(original)
            
            # 情况1: 字符串形式的枚举值
            if isinstance(config.transcribe_model, str):
                model_str_mapping = {
                    "FasterWhisper ✨": TranscribeModelEnum.FASTER_WHISPER,
                    "Whisper": TranscribeModelEnum.WHISPER,
                    "剪映接口": TranscribeModelEnum.JIANYING,
                    "必剪接口": TranscribeModelEnum.BIJIAN,
                    "J 接口": TranscribeModelEnum.JIANYING,  # 兼容旧版
                    "B 接口": TranscribeModelEnum.BIJIAN,  # 兼容旧版
                    # 也支持枚举名称
                    "FASTER_WHISPER": TranscribeModelEnum.FASTER_WHISPER,
                    "WHISPER": TranscribeModelEnum.WHISPER,
                    "JIANYING": TranscribeModelEnum.JIANYING,
                    "BIJIAN": TranscribeModelEnum.BIJIAN,
                }
                config.transcribe_model = model_str_mapping.get(
                    config.transcribe_model,
                    TranscribeModelEnum.WHISPER
                )
            
            # 情况2: 枚举对象但可能是序列化后的
            elif hasattr(original, 'value'):
                # 尝试通过 value 重新映射
                model_value_mapping = {
                    "FasterWhisper ✨": TranscribeModelEnum.FASTER_WHISPER,
                    "Whisper": TranscribeModelEnum.WHISPER,
                    "J 接口": TranscribeModelEnum.JIANYING,
                    "B 接口": TranscribeModelEnum.BIJIAN,
                }
                if original.value in model_value_mapping:
                    config.transcribe_model = model_value_mapping[original.value]
            
            # 情况3: 确保是正确的枚举实例
            else:
                # 检查是否是 TranscribeModelEnum 的实例
                if not isinstance(config.transcribe_model, TranscribeModelEnum):
                    # 尝试通过字符串表示转换
                    str_repr = str(original)
                    if "FASTER_WHISPER" in str_repr:
                        config.transcribe_model = TranscribeModelEnum.FASTER_WHISPER
                    elif "JIANYING" in str_repr or "J 接口" in str_repr or "剪映接口" in str_repr:
                        config.transcribe_model = TranscribeModelEnum.JIANYING
                    elif "BIJIAN" in str_repr or "B 接口" in str_repr or "必剪接口" in str_repr:
                        config.transcribe_model = TranscribeModelEnum.BIJIAN
                    elif "WHISPER" in str_repr:
                        config.transcribe_model = TranscribeModelEnum.WHISPER
                    else:
                        config.transcribe_model = TranscribeModelEnum.WHISPER
        
        # 修复 whisper_model / faster_whisper_model
        if hasattr(config, 'whisper_model') and isinstance(config.whisper_model, str):
            from app.core.entities import WhisperModelEnum
            whisper_model_mapping = {
                "tiny": WhisperModelEnum.TINY,
                "base": WhisperModelEnum.BASE,
                "small": WhisperModelEnum.SMALL,
                "medium": WhisperModelEnum.MEDIUM,
                "large-v1": WhisperModelEnum.LARGE_V1,
                "large-v2": WhisperModelEnum.LARGE_V2,
                "large-v3": WhisperModelEnum.LARGE_V3,
                "turbo": WhisperModelEnum.TURBO,
            }
            config.whisper_model = whisper_model_mapping.get(
                config.whisper_model,
                WhisperModelEnum.MEDIUM
            )
        if hasattr(config, 'faster_whisper_model'):
            if isinstance(config.faster_whisper_model, str):
                whisper_model_mapping = {
                    "tiny": FasterWhisperModelEnum.TINY,
                    "base": FasterWhisperModelEnum.BASE,
                    "small": FasterWhisperModelEnum.SMALL,
                    "medium": FasterWhisperModelEnum.MEDIUM,
                    "large-v2": FasterWhisperModelEnum.LARGE_V2,
                    "large-v3": FasterWhisperModelEnum.LARGE_V3,
                    "large-v3-turbo": FasterWhisperModelEnum.LARGE_V3_TURBO,
                    "belle-large-v3-zh-punct": FasterWhisperModelEnum.BELLE_LARGE_V3_ZH_PUNCT,
                }
                config.faster_whisper_model = whisper_model_mapping.get(
                    config.faster_whisper_model,
                    FasterWhisperModelEnum.MEDIUM
                )
        
        # 修复 faster_whisper_vad_method
        if hasattr(config, 'faster_whisper_vad_method'):
            if isinstance(config.faster_whisper_vad_method, str):
                vad_method_mapping = {
                    "silero_v4_fw": VadMethodEnum.SILERO_V4_FW,
                    "silero_v3": VadMethodEnum.SILERO_V3,
                    "pyannote_v3": VadMethodEnum.PYANNOTE_V3,
                }
                config.faster_whisper_vad_method = vad_method_mapping.get(
                    config.faster_whisper_vad_method,
                    VadMethodEnum.SILERO_V3
                )
    
    def _transcribe_with_python_library(self, audio_path: str, config: Any, callback: Any, initial_prompt: str = "") -> Any:
        """
        使用 Python faster-whisper 库进行转录（无需外部程序）
        
        Args:
            audio_path: 音频文件路径
            config: TranscribeConfig 配置对象
            callback: 进度回调函数
        
        Returns:
            ASRData 对象
        """
        # 模型名称映射
        model_name_map = {
            FasterWhisperModelEnum.TINY: "tiny",
            FasterWhisperModelEnum.BASE: "base",
            FasterWhisperModelEnum.SMALL: "small",
            FasterWhisperModelEnum.MEDIUM: "medium",
            FasterWhisperModelEnum.LARGE_V2: "large-v2",
            FasterWhisperModelEnum.LARGE_V3: "large-v3",
            FasterWhisperModelEnum.LARGE_V3_TURBO: "large-v3-turbo",
            FasterWhisperModelEnum.BELLE_LARGE_V3_ZH_PUNCT: "belle-large-v3-zh-punct",
        }
        
        # 调试输出
        print(f"[VideoCaptioner] DEBUG (Python Lib): config.faster_whisper_model type: {type(config.faster_whisper_model)}")
        print(f"[VideoCaptioner] DEBUG (Python Lib): config.faster_whisper_model value: {config.faster_whisper_model}")

        model_name = model_name_map.get(
            config.faster_whisper_model if hasattr(config, 'faster_whisper_model') else FasterWhisperModelEnum.MEDIUM,
            "medium"
        )
        
        # 如果映射失败，再次尝试通过字符串匹配（兜底逻辑）
        if model_name == "medium":
            val = config.faster_whisper_model if hasattr(config, 'faster_whisper_model') else None
            if val != FasterWhisperModelEnum.MEDIUM:
                if str(val) == "belle-large-v3-zh-punct" or \
                   str(val) == "FasterWhisperModelEnum.BELLE_LARGE_V3_ZH_PUNCT":
                    model_name = "belle-large-v3-zh-punct"
        
        # 设备选择
        device = config.device if hasattr(config, 'device') else "cpu"
        
        # 计算类型
        if device == "cuda":
            compute_type = "float16"  # GPU 使用 float16
        else:
            # 为提升 CPU 识别质量，默认使用 float32
            # 如遇内存或性能问题，后续将自动退回 int8
            compute_type = "float32"
        
        # 语言
        language = config.transcribe_language if hasattr(config, 'transcribe_language') else "zh"
        if language == "auto":
            language = None  # None 表示自动检测
        
        # 模型目录
        models_dir = config.faster_whisper_model_dir if hasattr(config, 'faster_whisper_model_dir') else get_comfyui_models_dir()
        
        # 确保模型可用（仅检查，不下载）
        try:
            from .model_downloader import ensure_model_available
            if not ensure_model_available(model_name, models_dir):
                 print(f"[VideoCaptioner] 错误: 模型 {model_name} 不存在且禁用了自动下载")
                 raise RuntimeError(f"模型 {model_name} 未找到，请手动下载并放置在 models/whisper 目录")
        except Exception as e:
            print(f"[VideoCaptioner] 模型可用性检查失败: {e}")
            raise
        # VAD 参数
        vad_filter = config.faster_whisper_vad_filter if hasattr(config, 'faster_whisper_vad_filter') else True
        vad_threshold = config.faster_whisper_vad_threshold if hasattr(config, 'faster_whisper_vad_threshold') else 0.4
        
        # 词级时间戳
        word_timestamps = config.need_word_time_stamp if hasattr(config, 'need_word_time_stamp') else False
        
        print(f"[FasterWhisper] 配置:")
        print(f"  - 模型: {model_name}")
        print(f"  - 设备: {device}")
        print(f"  - 计算类型: {compute_type}")
        print(f"  - 语言: {language or 'auto'}")
        print(f"  - 模型目录: {models_dir}")
        print(f"  - 词级时间戳: {word_timestamps}")
        
        # 创建 Python 实现
        asr = FasterWhisperPython(
            model_name=model_name,
            model_dir=models_dir,
            device=device,
            compute_type=compute_type,
            language=language,
            vad_filter=vad_filter,
            vad_threshold=vad_threshold,
            initial_prompt=initial_prompt,
            temperature=0.0,
            condition_on_previous_text=True,
            word_timestamps=word_timestamps,
        )
        
        if initial_prompt:
            print(f"[FasterWhisper] 使用识别提示词以提高准确率")
            print(f"[FasterWhisper] 提示词预览: {initial_prompt[:80]}...")
        
        try:
            srt_text = asr.transcribe_to_srt(audio_path, callback=callback)
        except Exception as e:
            err = str(e)
            if "out of memory" in err.lower() and device == "cuda":
                print(f"[FasterWhisper] CUDA OOM，切换到 medium 模型并重试")
                asr = FasterWhisperPython(
                    model_name="medium",
                    model_dir=models_dir,
                    device=device,
                    compute_type="float16",
                    language=language,
                    vad_filter=vad_filter,
                    vad_threshold=vad_threshold,
                    initial_prompt=initial_prompt,
                    temperature=0.0,
                    condition_on_previous_text=True,
                    word_timestamps=word_timestamps,
                )
                srt_text = asr.transcribe_to_srt(audio_path, callback=callback)
            else:
                if device != "cuda":
                    print(f"[FasterWhisper] CPU float32 失败，降级到 int8 重试")
                    asr = FasterWhisperPython(
                        model_name=model_name,
                        model_dir=models_dir,
                        device=device,
                        compute_type="int8",
                        language=language,
                        vad_filter=vad_filter,
                        vad_threshold=vad_threshold,
                        initial_prompt=initial_prompt,
                        temperature=0.0,
                        condition_on_previous_text=True,
                        word_timestamps=word_timestamps,
                    )
                    srt_text = asr.transcribe_to_srt(audio_path, callback=callback)
                else:
                    raise
        
        import re
        degenerate = False
        text_for_check = srt_text
        if text_for_check:
            if re.search(r'(中文){4,}', text_for_check):
                degenerate = True
            else:
                unique_ratio = len(set(text_for_check)) / max(len(text_for_check), 1)
                if len(text_for_check) > 50 and unique_ratio < 0.2:
                    degenerate = True
        if degenerate:
            print("[FasterWhisper] 检测到重复乱文，启用降级策略：自动语言检测与去上下文")
            asr = FasterWhisperPython(
                model_name=model_name,
                model_dir=models_dir,
                device=device,
                compute_type=compute_type,
                language=None,
                vad_filter=vad_filter,
                vad_threshold=vad_threshold,
                initial_prompt=initial_prompt,
                temperature=0.2,
                condition_on_previous_text=False,
                word_timestamps=word_timestamps,
            )
            srt_text = asr.transcribe_to_srt(audio_path, callback=callback)
        
        # 转换为 ASRData 对象
        asr_data = ASRData.from_srt(srt_text)
        
        return asr_data
    
    def _apply_postprocessing(
        self,
        asr_data: ASRData,
        min_confidence: float,
        timestamp_precision: int,
        merge_threshold: float,
        trim_silence: bool,
        remove_keywords: str = "",
    ) -> ASRData:
        """
        应用后处理功能
        
        Args:
            asr_data: 原始转录数据
            min_confidence: 最小置信度阈值
            timestamp_precision: 时间戳精度
            merge_threshold: 合并阈值
            trim_silence: 是否裁剪静音
        
        Returns:
            处理后的 ASRData
        """
        if not asr_data or not asr_data.segments:
            return asr_data
        
        segments = asr_data.segments
        processed_segments = []
        
        # 0. 关键词过滤（删除与语音内容无关的片段）
        if remove_keywords is not None:
            kw = [s.strip() for s in str(remove_keywords).split(",") if s.strip()]
            if kw:
                keyword_set = set([k.lower() for k in kw])
                if segments:
                    original_segments = segments[:]
                    for seg in segments:
                        text_l = seg.text.strip().lower()
                        if any(k in text_l for k in keyword_set):
                            continue
                        processed_segments.append(seg)
                    # 若全部被过滤，则回退以避免空输出
                    segments = processed_segments if processed_segments else original_segments
                    processed_segments = []
            
            # 已移除默认关键词过滤逻辑，防止误删用户正常内容
            # 只有当用户显式提供 keywords 时才过滤

        
        # 1. 置信度过滤
        if min_confidence > 0:
            filtered_count = 0
            for seg in segments:
                # 假设 segment 有 confidence 属性（如果没有则跳过）
                confidence = getattr(seg, 'confidence', 1.0)
                if confidence >= min_confidence:
                    processed_segments.append(seg)
                else:
                    filtered_count += 1
            segments = processed_segments
            processed_segments = []
        
        # 2. 裁剪首尾静音
        if trim_silence and segments:
            # 移除开头和结尾的空文本或纯空白字幕
            while segments and not segments[0].text.strip():
                segments.pop(0)
            while segments and not segments[-1].text.strip():
                segments.pop()
        
        # 3. 合并间隔过小的片段
        if merge_threshold > 0 and segments:
            merged_segments = [segments[0]]
            merged_count = 0
            
            for i in range(1, len(segments)):
                prev = merged_segments[-1]
                curr = segments[i]
                
                # 计算时间间隔
                time_gap = curr.start_time - prev.end_time
                
                if time_gap <= merge_threshold:
                    # 合并到前一个片段
                    prev.end_time = curr.end_time
                    prev.text = prev.text.strip() + " " + curr.text.strip()
                    merged_count += 1
                else:
                    merged_segments.append(curr)
            
            segments = merged_segments
        
        # 4. 调整时间戳精度
        if timestamp_precision != 3:  # 默认是 3 位（毫秒）
            for seg in segments:
                seg.start_time = round(seg.start_time, timestamp_precision)
                seg.end_time = round(seg.end_time, timestamp_precision)
        
        # 创建新的 ASRData 对象
        processed_asr_data = ASRData(segments=segments)
        
        return processed_asr_data
    
    def _add_metadata_to_text(self, asr_data: ASRData) -> str:
        """
        添加元数据到文本输出
        
        Args:
            asr_data: 字幕数据
        
        Returns:
            包含元数据的文本
        """
        if not asr_data or not asr_data.segments:
            return ""
        
        lines = []
        lines.append("=== 字幕详细信息 ===\n")
        
        for i, seg in enumerate(asr_data.segments, 1):
            confidence = getattr(seg, 'confidence', None)
            speaker = getattr(seg, 'speaker', None)
            
            metadata_parts = [f"#{i}"]
            if confidence is not None:
                metadata_parts.append(f"置信度: {confidence:.2%}")
            if speaker:
                metadata_parts.append(f"说话人: {speaker}")
            
            metadata_line = " | ".join(metadata_parts)
            time_line = f"[{seg.start_time:.3f}s - {seg.end_time:.3f}s]"
            
            lines.append(f"{metadata_line}")
            lines.append(f"{time_line}")
            lines.append(f"{seg.text}")
            lines.append("")  # 空行分隔
        
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "VideoTranscribeNode": VideoTranscribeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTranscribeNode": "视频转录（语音识别）"
}
