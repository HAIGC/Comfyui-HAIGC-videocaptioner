import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Callable, Any

from ..utils.logger import setup_logger
from .asr_data import ASRData, ASRDataSeg
from .base import BaseASR
import torch
from typing import Optional as _Optional

logger = setup_logger("whisper_torch")


class WhisperTorchASR(BaseASR):
    def __init__(
        self,
        audio_path: str,
        whisper_model: str,
        language: str = "zh",
        device: str = "cpu",
        prompt: Optional[str] = None,
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
    ):
        super().__init__(audio_path, use_cache)
        assert os.path.exists(audio_path), f"音频文件 {audio_path} 不存在"
        self.model_name = whisper_model
        self.language = language
        self.device = device
        self.prompt = prompt
        self.need_word_time_stamp = need_word_time_stamp
        # 运行期增强选项默认值（由外部在 run 前设置）
        self.vocal_separation: bool = False
        self.noise_reduction_level: int = 20

    def _make_segments(self, resp_data: str) -> List[ASRDataSeg]:
        asr_data = ASRData.from_srt(resp_data)
        filtered_segments = []
        for seg in asr_data.segments:
            text = seg.text.strip()
            if text:
                filtered_segments.append(seg)
        return filtered_segments

    def _run(
        self, callback: Optional[Callable[[int, str], None]] = None, **kwargs: Any
    ) -> str:
        def _default_callback(x, y):
            pass

        if callback is None:
            callback = _default_callback

        try:
            import whisper  # pip package: openai-whisper
        except ImportError:
            repo_path = os.environ.get("WHISPER_REPO_PATH")
            if repo_path and os.path.isdir(repo_path):
                import sys
                sys.path.insert(0, repo_path)
                try:
                    import whisper  # from local source
                except Exception as e2:
                    raise RuntimeError(
                        f"无法从本地仓库加载 Whisper: {repo_path}. 错误: {e2}"
                    ) from e2
            else:
                raise RuntimeError(
                    "未安装 openai-whisper 库，请运行: pip install openai-whisper；"
                    "或者设置环境变量 WHISPER_REPO_PATH 指向本地仓库路径"
                )

        with tempfile.TemporaryDirectory() as temp_path:
            temp_dir = Path(temp_path)
            wav_path = temp_dir / "audio.wav"
            output_path = wav_path.with_suffix(".srt")
            enhanced_wav = temp_dir / "voice_enhanced.wav"

            # 复制输入音频为临时 wav
            if isinstance(self.audio_path, str):
                shutil.copy2(self.audio_path, wav_path)
            else:
                if self.file_binary:
                    wav_path.write_bytes(self.file_binary)
                else:
                    raise ValueError("No audio data available")

            # 可选人声分离/增强
            try:
                if kwargs.get("vocal_separation", False):
                    callback(10, "人声增强处理中...")
                    # 优先使用 demucs 分离人声
                    used_demucs = self._separate_vocals_demucs(str(wav_path), str(enhanced_wav))
                    if not used_demucs:
                        # 回退到 ffmpeg 轻量降噪
                        level = int(kwargs.get("noise_reduction_level", 20))
                        self._enhance_speech_ffmpeg(str(wav_path), str(enhanced_wav), level=level)
            except Exception as e_vs:
                logger.warning(f"人声增强失败，使用原始音频: {e_vs}")
                enhanced_wav = wav_path
            else:
                if not enhanced_wav.exists():
                    enhanced_wav = wav_path

            callback(15, "加载 Whisper 模型...")
            logger.info(f"加载 Whisper 模型: {self.model_name}, 设备: {self.device}")
            use_transformers = self.model_name in ("large-v3", "turbo")
            if use_transformers:
                try:
                    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
                except Exception as e_tf:
                    raise RuntimeError(f"需要安装 transformers 以使用 {self.model_name}: {e_tf}")
                comfyui_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
                models_dir = comfyui_root / "models" / "whisper"
                models_dir.mkdir(parents=True, exist_ok=True)
                repo_id = "openai/whisper-large-v3" if self.model_name == "large-v3" else "openai/whisper-large-v3-turbo"
                # 先尝试使用已有本地目录（避免重复下载）
                def _find_local_dir(base: Path, name: str) -> _Optional[Path]:
                    candidates = [
                        base / f"transformers-{name}",
                        base / f"whisper-{name}",
                        base / name,  # 用户手工目录，如 whisper-large-v3
                    ]
                    for c in candidates:
                        try:
                            if c.exists():
                                has_model = (c / "model.safetensors").exists() or (c / "pytorch_model.bin").exists()
                                has_config = (c / "config.json").exists()
                                if has_model and has_config:
                                    return c
                        except Exception:
                            pass
                    # 尝试在仓库缓存结构中查找
                    for p in base.glob(f"**/*{name}*/config.json"):
                        d = p.parent
                        if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
                            return d
                    return None
                local_dir = _find_local_dir(models_dir, self.model_name)
                if local_dir is None:
                    # 若本地不存在或不完整，执行下载
                    try:
                        from huggingface_hub import snapshot_download
                        local_dir = models_dir / f"transformers-{self.model_name}"
                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=str(local_dir),
                            local_dir_use_symlinks=False,
                            resume_download=True,
                        )
                    except Exception:
                        # 回退为在线加载并缓存
                        local_dir = None
                model_source = str(local_dir) if local_dir else repo_id
                device_str = "cuda:0" if (self.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
                dtype = torch.float16 if device_str.startswith("cuda") else torch.float32
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_source, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True, cache_dir=str(models_dir)
                )
                processor = AutoProcessor.from_pretrained(model_source, cache_dir=str(models_dir))
                model.to(device_str)
                asr_pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=dtype,
                    device=device_str,
                )
            else:
                try:
                    model = whisper.load_model(self.model_name, device=self.device)
                except Exception as e_load:
                    try:
                        from huggingface_hub import snapshot_download
                    except Exception as e_hf:
                        raise RuntimeError(f"加载失败且未安装 huggingface_hub: {e_hf}") from e_hf
                    comfyui_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
                    models_dir = comfyui_root / "models" / "whisper"
                    models_dir.mkdir(parents=True, exist_ok=True)
                    repo_id = f"openai/whisper-{self.model_name}"
                    local_dir = models_dir / f"torch-whisper-{self.model_name}"
                    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
                    model = whisper.load_model(str(local_dir), device=self.device)

            # fp16 仅在 CUDA 下启用
            fp16 = self.device == "cuda"
            transcribe_kwargs = {
                "language": None if self.language == "auto" else self.language,
                "task": "transcribe",
                "verbose": False,
                "fp16": fp16,
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "condition_on_previous_text": True,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -0.8,
                "no_speech_threshold": 0.3,
                "word_timestamps": self.need_word_time_stamp,
                "patience": 1.0,
                "suppress_tokens": [-1],
            }
            if self.prompt:
                transcribe_kwargs["initial_prompt"] = self.prompt
            # 防幻觉模式：更严格参数，并移除提示词
            if kwargs.get("anti_hallucination", False):
                transcribe_kwargs["condition_on_previous_text"] = False
                transcribe_kwargs["compression_ratio_threshold"] = 2.0
                transcribe_kwargs["logprob_threshold"] = -0.6
                transcribe_kwargs["no_speech_threshold"] = 0.6
                transcribe_kwargs["temperature"] = 0.0
                if "initial_prompt" in transcribe_kwargs:
                    transcribe_kwargs.pop("initial_prompt")

            callback(20, "开始识别...")
            if use_transformers:
                lang_map = {
                    "en": "english",
                    "zh": "chinese",
                    "ja": "japanese",
                    "ko": "korean",
                    "fr": "french",
                    "de": "german",
                    "es": "spanish",
                    "ru": "russian",
                }
                lang_arg = transcribe_kwargs.get("language")
                if lang_arg in lang_map:
                    lang_arg = lang_map[lang_arg]
                max_positions = getattr(model.config, "max_target_positions", 448)
                reserve_tokens = 12  # 为起始/提示/上下文保留，避免越界
                safe_max_new = max(256, max_positions - reserve_tokens)
                gen_kwargs = {
                    "max_new_tokens": safe_max_new,
                    "num_beams": 1,
                    "condition_on_prev_tokens": transcribe_kwargs.get("condition_on_previous_text", True),
                    "compression_ratio_threshold": transcribe_kwargs.get("compression_ratio_threshold", 2.4),
                    "temperature": transcribe_kwargs.get("temperature", 0.0),
                    "logprob_threshold": transcribe_kwargs.get("logprob_threshold", -0.8),
                    "no_speech_threshold": transcribe_kwargs.get("no_speech_threshold", 0.3),
                }
                if lang_arg:
                    gen_kwargs["language"] = lang_arg
                return_ts = "word" if self.need_word_time_stamp else True
                trs = asr_pipe(str(enhanced_wav), return_timestamps=return_ts, generate_kwargs=gen_kwargs)
                chunks = trs.get("chunks", [])
                if not chunks:
                    raise RuntimeError("Whisper Transformers 未返回任何识别片段")
                srt_lines = []
                def fmt_time(secs: float) -> str:
                    hours = int(secs // 3600)
                    minutes = int((secs % 3600) // 60)
                    seconds = int(secs % 60)
                    millis = int((secs % 1) * 1000)
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
                for i, seg in enumerate(chunks, 1):
                    start = float(seg.get("timestamp", [0.0, 0.0])[0] if isinstance(seg.get("timestamp"), (list, tuple)) else seg.get("start", 0.0))
                    end = float(seg.get("timestamp", [start, start])[1] if isinstance(seg.get("timestamp"), (list, tuple)) else seg.get("end", start))
                    text = str(seg.get("text", "")).strip()
                    srt_lines.append(f"{i}")
                    srt_lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
                    srt_lines.append(text)
                    srt_lines.append("")
                    if i % 10 == 0:
                        callback(min(90, 15 + i), f"已识别 {i} 段")
                callback(100, "识别完成")
                return "\n".join(srt_lines)
            else:
                result = model.transcribe(str(enhanced_wav), **transcribe_kwargs)

            segments = result.get("segments", [])
            if not segments:
                # 退化重试：关闭上下文、轻微升温并自动语言
                retry_kwargs = dict(transcribe_kwargs)
                retry_kwargs["condition_on_previous_text"] = False
                retry_kwargs["temperature"] = 0.2
                retry_kwargs["language"] = None
                result = model.transcribe(str(enhanced_wav), **retry_kwargs)
                segments = result.get("segments", [])
                if not segments:
                    raise RuntimeError("Whisper 未返回任何识别片段")

            # 生成 SRT 文本
            def fmt_time(secs: float) -> str:
                hours = int(secs // 3600)
                minutes = int((secs % 3600) // 60)
                seconds = int(secs % 60)
                millis = int((secs % 1) * 1000)
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

            srt_lines = []
            for i, seg in enumerate(segments, 1):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                text = str(seg.get("text", "")).strip()
                srt_lines.append(f"{i}")
                srt_lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
                srt_lines.append(text)
                srt_lines.append("")
                if i % 10 == 0:
                    callback(min(90, 15 + i), f"已识别 {i} 段")

            callback(100, "识别完成")
            return "\n".join(srt_lines)

    def _get_key(self):
        return f"{self.crc32_hex}-{self.model_name}-{self.language}-{self.device}-vs{int(bool(getattr(self, 'vocal_separation', False)))}-nr{int(getattr(self, 'noise_reduction_level', 20))}"

    def _enhance_speech_ffmpeg(self, input_wav: str, output_wav: str, level: int = 20):
        """
        使用 ffmpeg 做轻量级人声增强/降噪，避免额外依赖
        - 高频抑制与低频提升：高通（200Hz）、低通（4000Hz）
        - 频域降噪：afftdn
        """
        import subprocess
        # 将等级映射到噪声强度和门限
        nf = max(10, min(40, level)) * -1  # afftdn 噪声因子（负值）
        gate = 0.02 + (0.01 * (level / 40))  # 动态门限
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_wav,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            f"highpass=f=200,lowpass=f=4000,afftdn=nf={nf},agate=threshold={gate}:ratio=2:attack=10:release=100",
            output_wav,
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        subprocess.run(cmd, check=True, creationflags=creationflags)

    def _separate_vocals_demucs(self, input_wav: str, output_wav: str) -> bool:
        """
        可选使用 demucs 进行人声分离，优先级高于 ffmpeg
        - 需要 demucs 包或命令可用
        - 成功返回 True，并生成 output_wav；失败返回 False
        """
        import shutil as _sh
        import subprocess
        # 尝试 python 包
        try:
            import demucs  # noqa: F401
            use_cli = False
        except Exception:
            use_cli = True
        outdir = Path(tempfile.mkdtemp(prefix="demucs_out_"))
        try:
            if not use_cli:
                # 使用 CLI 方式更稳定统一
                use_cli = True
            if use_cli:
                cmd = [
                    sys.executable,
                    "-m",
                    "demucs",
                    "--two-stems=vocals",
                    "-n",
                    os.environ.get("DEMUCS_MODEL", "mdx_extra_q"),
                    "-o",
                    str(outdir),
                    input_wav,
                ]
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                subprocess.run(cmd, check=True, creationflags=creationflags)
                # 找到生成的 vocals.wav
                vocals = None
                for p in outdir.rglob("vocals.wav"):
                    vocals = p
                    break
                if vocals and vocals.exists():
                    _sh.copy2(str(vocals), output_wav)
                    return True
            return False
        except Exception as e:
            logger.warning(f"demucs 分离失败: {e}")
            return False
        finally:
            try:
                import shutil as _sh2
                _sh2.rmtree(outdir, ignore_errors=True)
            except Exception:
                pass
