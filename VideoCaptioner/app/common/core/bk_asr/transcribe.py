from typing import Any, Dict
from app.core.bk_asr.asr_data import ASRData
from app.core.bk_asr.bcut import BcutASR
from app.core.bk_asr.faster_whisper import FasterWhisperASR
from app.core.bk_asr.jianying import JianYingASR
from app.core.bk_asr.whisper_torch import WhisperTorchASR
from app.core.entities import TranscribeConfig, TranscribeModelEnum


def transcribe(audio_path: str, config: TranscribeConfig, callback=None) -> ASRData:
    """
    使用指定的转录配置对音频文件进行转录

    Args:
        audio_path: 音频文件路径
        config: 转录配置
        callback: 进度回调函数,接收两个参数(progress: int, message: str)

    Returns:
        ASRData: 转录结果数据
    """

    def _default_callback(x, y):
        pass

    if callback is None:
        callback = _default_callback

    # 获取ASR模型类
    ASR_MODELS = {
        TranscribeModelEnum.JIANYING: JianYingASR,
        TranscribeModelEnum.BIJIAN: BcutASR,
        TranscribeModelEnum.FASTER_WHISPER: FasterWhisperASR,
        TranscribeModelEnum.WHISPER: WhisperTorchASR,
    }

    if config.transcribe_model is None:
        raise ValueError("转录模型未设置")
    
    asr_class = ASR_MODELS.get(config.transcribe_model)
    
    # 如果没找到，尝试通过 value 或 name 匹配（解决不同模块加载导致的枚举类不一致问题）
    if not asr_class:
        # 1. 尝试通过 value 匹配
        for enum_member, cls in ASR_MODELS.items():
            if hasattr(config.transcribe_model, 'value') and str(enum_member.value) == str(config.transcribe_model.value):
                asr_class = cls
                break
    
    if not asr_class:
        # 2. 尝试通过 name 匹配
        for enum_member, cls in ASR_MODELS.items():
            if hasattr(config.transcribe_model, 'name') and enum_member.name == config.transcribe_model.name:
                asr_class = cls
                break
                
    if not asr_class:
        raise ValueError(f"无效的转录模型: {config.transcribe_model}")

    # 构建ASR参数
    asr_args: Dict[str, Any] = {
        "use_cache": config.use_asr_cache,
        "need_word_time_stamp": config.need_word_time_stamp,
    }

    # 根据不同模型添加特定参数
    if config.transcribe_model == TranscribeModelEnum.FASTER_WHISPER:
        asr_args["faster_whisper_program"] = config.faster_whisper_program
        asr_args["language"] = config.transcribe_language
        asr_args["whisper_model"] = (
            config.faster_whisper_model.value if config.faster_whisper_model else None
        )
        asr_args["model_dir"] = config.faster_whisper_model_dir
        asr_args["device"] = config.faster_whisper_device
        asr_args["vad_filter"] = config.faster_whisper_vad_filter
        asr_args["vad_threshold"] = config.faster_whisper_vad_threshold
        asr_args["vad_method"] = (
            config.faster_whisper_vad_method.value
            if config.faster_whisper_vad_method
            else None
        )
        asr_args["one_word"] = config.faster_whisper_one_word
        asr_args["prompt"] = config.faster_whisper_prompt
    elif config.transcribe_model == TranscribeModelEnum.WHISPER:
        asr_args["language"] = config.transcribe_language
        asr_args["whisper_model"] = (
            config.whisper_model.value if config.whisper_model else None
        )
        # 复用设备与提示字段
        asr_args["device"] = getattr(config, "faster_whisper_device", "cpu")
        asr_args["prompt"] = getattr(config, "faster_whisper_prompt", None)
        # 运行参数在 run() 传入，而不是构造器
        run_kwargs = {
            "anti_hallucination": getattr(config, "anti_hallucination", True),
        }
    else:
        run_kwargs = {}

    # 创建ASR实例并运行
    asr = asr_class(audio_path, **asr_args)
    # 在运行前设置增强属性以影响缓存键
    try:
        pass
    except Exception:
        pass
    asr_data = asr.run(callback=callback, **run_kwargs)

    # 优化字幕显示时间 #161
    if not config.need_word_time_stamp:
        asr_data.optimize_timing()

    return asr_data


if __name__ == "__main__":
    # 示例用法
    from app.core.entities import WhisperModelEnum

    # 创建配置
    config = TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPER_CPP,
        transcribe_language="zh",
        whisper_model=WhisperModelEnum.MEDIUM,
        use_asr_cache=True,
    )

    # 转录音频
    audio_file = "test.wav"

    def progress_callback(progress: int, message: str):
        print(f"Progress: {progress}%, Message: {message}")

    result = transcribe(audio_file, config, callback=progress_callback)
    print(result)
