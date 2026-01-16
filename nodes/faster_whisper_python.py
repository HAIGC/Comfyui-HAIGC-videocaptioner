"""
直接使用 Python faster-whisper 库的实现
不需要外部可执行文件
"""
import os
import gc
import re
from pathlib import Path
from typing import Optional, Callable, Any, List, Tuple
import torch

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("[警告] faster-whisper 库未安装，请运行: pip install faster-whisper")

try:
    from .resource_manager import get_resource_manager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False


class FasterWhisperPython:
    """
    使用 Python faster-whisper 库的 ASR 实现
    无需外部可执行文件
    """
    
    def __init__(
        self,
        model_name: str = "large-v3",
        model_dir: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        vad_filter: bool = True,
        vad_threshold: float = 0.4,
        initial_prompt: str = "",
        temperature: float = 0.0,
        condition_on_previous_text: bool = True,
        word_timestamps: bool = False,
    ):
        """
        初始化 FasterWhisper Python 实现
        
        Args:
            model_name: 模型名称 (tiny, base, small, medium, large-v2, large-v3, large-v3-turbo)
            model_dir: 模型目录（可选，默认自动下载到缓存）
            device: 设备 ("cpu" 或 "cuda")
            compute_type: 计算类型 ("int8", "int8_float16", "float16", "float32")
            language: 语言代码（None 表示自动检测）
            vad_filter: 是否使用 VAD 过滤
            vad_threshold: VAD 阈值
            word_timestamps: 是否启用词级时间戳
        """
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper 库未安装。\n"
                "请运行: pip install faster-whisper\n"
                "或: pip install -r requirements.txt"
            )
        
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language if language and language != "auto" else None
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold
        self.initial_prompt = initial_prompt
        self.temperature = temperature
        self.condition_on_previous_text = condition_on_previous_text
        self.word_timestamps = word_timestamps
        
        # 设置模型下载目录
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            # 使用 ComfyUI 的模型目录
            comfyui_root = Path(__file__).parent.parent.parent.parent
            self.model_dir = comfyui_root / "models" / "whisper"
            self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 调整 compute_type 根据设备
        if device == "cpu":
            if compute_type in ["float16", "int8_float16"]:
                self.compute_type = "int8"
                print(f"[提示] CPU 设备不支持 {compute_type}，自动切换到 int8")
        
        self.model = None
        self.auto_cleanup = True  # 默认自动清理资源
        print(f"[FasterWhisper] 准备加载模型: {model_name}")
        print(f"[FasterWhisper] 设备: {device}, 计算类型: {self.compute_type}")
        print(f"[FasterWhisper] 模型目录: {self.model_dir}")
    
    def load_model(self):
        """加载模型"""
        if self.model is not None:
            return
        
        try:
            print(f"[FasterWhisper] 正在加载模型...")
            
            # 设置下载目录为环境变量
            os.environ["HF_HOME"] = str(self.model_dir)
            
            # 尝试查找本地模型目录
            # 使用 model_downloader 的统一逻辑获取路径
            from .model_downloader import get_model_path
            
            # self.model_name 已经是字符串名称，例如 "belle-large-v3-zh-punct"
            local_model_path = get_model_path(self.model_name, str(self.model_dir))
            
            if local_model_path.exists():
                print(f"[FasterWhisper] 发现本地模型: {local_model_path}")
                model_path_to_use = str(local_model_path)
            else:
                raise FileNotFoundError(
                    f"模型未找到: {local_model_path}\n"
                    f"请手动下载模型并放置在: {local_model_path}"
                )
            
            self.model = WhisperModel(
                model_path_to_use,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.model_dir),
                local_files_only=True,
            )
            print(f"[FasterWhisper] 模型加载成功！")
            
        except Exception as e:
            print(f"[错误] 加载模型失败: {e}")
            raise RuntimeError(f"模型加载失败: {e}")
    
    def transcribe(
        self,
        audio_path: str,
        callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[Tuple[float, float, str]]:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            callback: 进度回调函数
        
        Returns:
            List of (start_time, end_time, text) tuples
        """
        # 加载模型
        self.load_model()
        
        if callback:
            callback(10, "开始识别...")
        
        try:
            print(f"[FasterWhisper] 开始转录: {audio_path}")
            
            # VAD 参数
            vad_parameters = None
            if self.vad_filter:
                vad_parameters = {
                    "threshold": self.vad_threshold,
                    "min_speech_duration_ms": 250,
                    "max_speech_duration_s": 30,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 400,
                }
            
            # 执行转录
            # 强制开启 word_timestamps 以获得更准确的 segment 时间戳（修正静音包含问题）
            # 只有当用户显式需要词级输出时，我们才在后续处理中展开它
            enable_word_timestamps = True
            
            segments, info = self.model.transcribe(
                audio_path,
                language=self.language,
                vad_filter=self.vad_filter,
                vad_parameters=vad_parameters,
                beam_size=5,
                best_of=5,
                temperature=self.temperature,
                condition_on_previous_text=self.condition_on_previous_text,
                initial_prompt=self.initial_prompt if self.initial_prompt else None,
                word_timestamps=enable_word_timestamps,
            )
            
            print(f"[FasterWhisper] 检测到语言: {info.language} (置信度: {info.language_probability:.2f})")
            
            if callback:
                callback(30, f"识别中... 语言: {info.language}")
            
            # 收集结果
            results = []
            total_segments = 0
            
            # 纯标点过滤正则
            punct_pattern = re.compile(r'^[。！？；，、：.!?;,:\s]+$')
            
            for i, segment in enumerate(segments):
                # 过滤纯标点或空文本
                text = segment.text.strip()
                if not text or punct_pattern.match(text):
                    continue

                # 如果用户配置启用了词级时间戳且存在词级信息，则按词输出
                if self.word_timestamps and hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        # 不去除空格，以便后续合并时保持正确的间隔（特别是英文）
                        word_text = word.word
                        # 同样过滤词级结果中的纯标点（可选，视需求而定，通常保留词级标点是OK的，但如果是纯标点词则过滤）
                        if word_text and word_text.strip() and not punct_pattern.match(word_text.strip()):
                            results.append((
                                word.start,
                                word.end,
                                word_text
                            ))
                            total_segments += 1
                else:
                    # 否则按段落输出
                    # 此时 segment.start/end 已经由 word_timestamps=True 进行了修正
                    results.append((
                        segment.start,
                        segment.end,
                        text
                    ))
                    total_segments += 1
                
                # 更新进度
                if callback and i % 10 == 0:
                    progress = min(30 + (i * 60 // max(total_segments, 1)), 90)
                    callback(progress, f"已识别 {total_segments} 段")
            
            if callback:
                callback(100, f"识别完成！共 {len(results)} 段")
            
            print(f"[FasterWhisper] 转录完成，共 {len(results)} 段字幕")
            
            # 自动清理资源
            if self.auto_cleanup:
                self.cleanup()
            
            return results
            
        except Exception as e:
            print(f"[错误] 转录失败: {e}")
            raise RuntimeError(f"转录失败: {e}")
    
    def transcribe_to_srt(
        self,
        audio_path: str,
        callback: Optional[Callable[[int, str], None]] = None,
    ) -> str:
        """
        转录并返回 SRT 格式字符串
        
        Args:
            audio_path: 音频文件路径
            callback: 进度回调函数
        
        Returns:
            SRT 格式字符串
        """
        segments = self.transcribe(audio_path, callback)
        
        # 转换为 SRT 格式
        srt_lines = []
        for i, (start, end, text) in enumerate(segments, 1):
            # 时间格式：00:00:00,000
            start_time = self._format_timestamp(start)
            end_time = self._format_timestamp(end)
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # 空行分隔
        
        return "\n".join(srt_lines)
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """将秒转换为 SRT 时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def cleanup(self, verbose: bool = True):
        """
        显式清理资源
        释放模型和清理 GPU 显存、CPU 内存
        """
        if verbose:
            print("[FasterWhisper] 开始清理资源...")
        
        # 使用资源管理器（如果可用）
        if RESOURCE_MANAGER_AVAILABLE:
            manager = get_resource_manager(verbose=verbose)
            manager.print_memory_info("清理前:")
        
        # 清理模型
        if self.model is not None:
            del self.model
            self.model = None
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理 GPU 显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # 使用资源管理器（如果可用）
        if RESOURCE_MANAGER_AVAILABLE:
            manager = get_resource_manager(verbose=verbose)
            manager.print_memory_info("清理后:")
        elif verbose:
            print("[FasterWhisper] 资源清理完成")
    
    def __del__(self):
        """析构函数：清理资源"""
        try:
            self.cleanup(verbose=False)
        except:
            pass  # 忽略析构时的错误


def test_faster_whisper_python():
    """测试函数"""
    print("=" * 60)
    print("测试 FasterWhisper Python 实现")
    print("=" * 60)
    
    if not FASTER_WHISPER_AVAILABLE:
        print("❌ faster-whisper 库未安装")
        print("请运行: pip install faster-whisper")
        return False
    
    print("✅ faster-whisper 库已安装")
    
    try:
        # 创建实例（使用最小的模型进行测试）
        asr = FasterWhisperPython(
            model_name="tiny",
            device="cpu",
            compute_type="int8",
        )
        print("✅ FasterWhisper 初始化成功")
        return True
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False


if __name__ == "__main__":
    test_faster_whisper_python()

