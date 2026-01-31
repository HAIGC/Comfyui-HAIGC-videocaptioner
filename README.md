# VideoCaptioner ComfyUI Node

这是一个功能强大的 ComfyUI 视频字幕处理节点套件，基于 LLM（大语言模型）提供智能视频字幕生成、断句、翻译和优化全流程解决方案。

本项目参考自：[VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)

## ✨ 主要功能

- **多引擎语音识别 (ASR)**：支持 FasterWhisper 本地模型、剪映接口、必剪接口，提供高精度语音转文字。
- **智能字幕处理**：
  - **智能断句**：基于 VAD 和语义分析的智能断句。
  - **字幕优化**：
    - **LLM 优化**：使用大模型校正错别字、优化标点符号、统一专业术语。
    - **智能分段**：基于规则（字数、时间间隔）的智能合并与拆分。
  - **批量替换**：支持按时间戳智能合并、按字数匹配替换、强制行替换等多种策略。
- **多语言翻译**：基于 LLM 的高质量字幕翻译。
- **格式支持**：支持导出 SRT, LRC, VTT, ASS, TXT 等多种字幕格式。

## 🧩 节点说明

### 1. VideoTranscribe (视频转录)
核心节点，用于从视频中提取语音并生成字幕。
- **输入**：视频文件、转录配置
- **功能**：支持选择不同的 ASR 模型（Whisper/剪映/必剪）进行转录。

### 2. SubtitleTextProcessor (字幕文本处理器)
用于对生成的字幕文本进行后期处理和校对。
- **功能**：
  - 去除标点、特殊符号、表情等。
  - **批量替换**：支持三种策略：
    - **按时间戳智能合并**：适合带有准确时间戳的文本。
    - **按字数匹配替换**：忽略时间戳，根据每行字数匹配（适合校对）。
    - **按顺序强制替换**：直接按行顺序覆盖。
  - 支持设置字数容差。

### 3. SubtitleOptimize (字幕优化 - LLM)
使用大语言模型对字幕内容进行深度优化。
- **功能**：校正错别字、优化标点、格式调整。
- **输入**：字幕数据、LLM 配置

### 4. SubtitleOptimizer (字幕优化 - 智能分段)
基于规则的字幕分段和合并工具。
- **功能**：
  - 智能合并短句。
  - 拆分长句（符合字幕阅读习惯）。
  - 支持平衡模式、激进模式等多种策略。

### 5. SubtitleTranslate (字幕翻译)
基于 LLM 的字幕翻译节点。
- **功能**：将字幕翻译成目标语言，保持时间轴对齐。

### 6. SubtitleSplit (字幕智能断句)
对长语音段落进行更细粒度的切分。

### 7. 配置节点
- **TranscribeConfig (转录配置)**：设置 ASR 模型、VAD 阈值等。
- **LLMConfig (LLM 配置)**：配置 LLM API（Base URL, API Key 等）。
- **TranslateConfig (翻译配置)**：配置翻译器类型（LLM/DeepLx/微软/谷歌）、目标语言、批处理参数等。

### 8. 工具节点
- **LoadVideo (加载视频)**：简单的视频加载工具，返回视频路径和信息。

## 📦 安装说明

1. 将本插件目录放入 ComfyUI 的 `custom_nodes` 目录下。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   便携环境安装依赖：
    ```bash
   python -m pip install -r requirements.txt
   ```
   模型下载：
   - 推荐模型：[Belle-whisper-large-v3-zh-punct](https://huggingface.co/CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32)
   - 官方模型：[Faster Whisper Models](https://huggingface.co/guillaumekln)
   
   模型放置路径（ComfyUI 根目录下的 `models/whisper`）：
   - `models/whisper/Belle-whisper-large-v3-zh-punct-ct2-float32` (推荐)
   - `models/whisper/faster-whisper-medium`
   - `models/whisper/faster-whisper-large-v3`
   - 等等...
3. 重启 ComfyUI。

## 📞 联系方式

- **作者微信**：HAIGC1994
- **GitHub**：[VideoCaptioner ComfyUI](https://github.com/WEIFENG2333/VideoCaptioner) (参考项目)

---
*注：本项目旨在为 ComfyUI 用户提供便捷的视频字幕处理工作流，核心算法参考了 VideoCaptioner 项目。*
