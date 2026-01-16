"""
VideoCaptioner ComfyUI 节点
基于 LLM 的智能视频字幕生成、断句、翻译全流程处理
"""

import sys
from pathlib import Path
import importlib.util
import traceback

# 添加 VideoCaptioner 到 Python 路径
current_dir = Path(__file__).parent
videocaptioner_path = current_dir / "VideoCaptioner"
if str(videocaptioner_path) not in sys.path:
    sys.path.insert(0, str(videocaptioner_path))

print(f"[VideoCaptioner] Loading from: {current_dir}")
print(f"[VideoCaptioner] VideoCaptioner path exists: {videocaptioner_path.exists()}")

# 初始化节点类映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 注册当前包到 sys.modules 以支持相对导入
import types
if "haigc_videocaptioner" not in sys.modules:
    haigc_package = types.ModuleType("haigc_videocaptioner")
    haigc_package.__path__ = [str(current_dir)]
    haigc_package.__file__ = str(current_dir / "__init__.py")
    sys.modules["haigc_videocaptioner"] = haigc_package

if "haigc_videocaptioner.nodes" not in sys.modules:
    nodes_package = types.ModuleType("haigc_videocaptioner.nodes")
    nodes_package.__path__ = [str(current_dir / "nodes")]
    nodes_package.__file__ = str(current_dir / "nodes" / "__init__.py")
    nodes_package.__package__ = "haigc_videocaptioner.nodes"
    sys.modules["haigc_videocaptioner.nodes"] = nodes_package

# 预加载节点辅助模块（base_node 等）到 sys.modules
def preload_helper_modules():
    """预加载辅助模块以支持相对导入"""
    nodes_dir = current_dir / "nodes"
    helper_modules = [
        "base_node",
        "faster_whisper_python",
        "resource_manager",
        "model_downloader"
    ]
    
    for helper_name in helper_modules:
        full_name = f"haigc_videocaptioner.nodes.{helper_name}"
        if full_name not in sys.modules:
            helper_path = nodes_dir / f"{helper_name}.py"
            if helper_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location(full_name, helper_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        module.__package__ = "haigc_videocaptioner.nodes"
                        sys.modules[full_name] = module
                        spec.loader.exec_module(module)
                        print(f"[VideoCaptioner] Preloaded helper: {helper_name}")
                except Exception as e:
                    print(f"[VideoCaptioner] Failed to preload {helper_name}: {e}")

# 预加载辅助模块
preload_helper_modules()

# 动态导入节点模块的辅助函数
def load_node_module(module_name, file_name):
    """
    加载节点模块并正确设置包信息以支持相对导入
    
    Args:
        module_name: 模块名称（例如 "video_transcribe_node"）
        file_name: 文件名（例如 "video_transcribe_node.py"）
    """
    nodes_dir = current_dir / "nodes"
    file_path = nodes_dir / file_name
    
    # 创建完整的模块名（包含包路径）
    full_module_name = f"haigc_videocaptioner.nodes.{module_name}"
    
    # 如果模块已经加载过，直接返回
    if full_module_name in sys.modules:
        return sys.modules[full_module_name]
    
    # 创建 spec
    spec = importlib.util.spec_from_file_location(full_module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {full_module_name}")
    
    # 创建模块
    module = importlib.util.module_from_spec(spec)
    
    # 设置模块属性以支持相对导入
    module.__package__ = "haigc_videocaptioner.nodes"
    
    # 注册到 sys.modules（必须在 exec_module 之前）
    sys.modules[full_module_name] = module
    
    # 执行模块
    spec.loader.exec_module(module)
    
    return module

# 尝试导入各个节点
try:
    video_transcribe_module = load_node_module("video_transcribe_node", "video_transcribe_node.py")
    NODE_CLASS_MAPPINGS["VideoTranscribe"] = video_transcribe_module.VideoTranscribeNode
    NODE_DISPLAY_NAME_MAPPINGS["VideoTranscribe"] = "视频转录（语音识别）"
    print("[VideoCaptioner] [OK] VideoTranscribe loaded")
except Exception as e:
    print(f"[VideoCaptioner] [ERROR] VideoTranscribe failed: {e}")
    traceback.print_exc()

try:
    subtitle_split_module = load_node_module("subtitle_split_node", "subtitle_split_node.py")
    NODE_CLASS_MAPPINGS["SubtitleSplit"] = subtitle_split_module.SubtitleSplitNode
    NODE_DISPLAY_NAME_MAPPINGS["SubtitleSplit"] = "字幕智能断句"
    print("[VideoCaptioner] [OK] SubtitleSplit loaded")
except Exception as e:
    print(f"[VideoCaptioner] [ERROR] SubtitleSplit failed: {e}")

try:
    subtitle_translate_module = load_node_module("subtitle_translate_node", "subtitle_translate_node.py")
    NODE_CLASS_MAPPINGS["SubtitleTranslate"] = subtitle_translate_module.SubtitleTranslateNode
    NODE_DISPLAY_NAME_MAPPINGS["SubtitleTranslate"] = "字幕翻译"
    print("[VideoCaptioner] [OK] SubtitleTranslate loaded")
except Exception as e:
    print(f"[VideoCaptioner] [ERROR] SubtitleTranslate failed: {e}")

try:
    subtitle_optimize_module = load_node_module("subtitle_optimize_node", "subtitle_optimize_node.py")
    NODE_CLASS_MAPPINGS["SubtitleOptimize"] = subtitle_optimize_module.SubtitleOptimizeNode
    NODE_DISPLAY_NAME_MAPPINGS["SubtitleOptimize"] = "字幕优化"
    print("[VideoCaptioner] [OK] SubtitleOptimize loaded")
except Exception as e:
    print(f"[VideoCaptioner] [ERROR] SubtitleOptimize failed: {e}")

try:
    subtitle_optimizer_module = load_node_module("subtitle_optimizer_node", "subtitle_optimizer_node.py")
    NODE_CLASS_MAPPINGS["SubtitleOptimizer"] = subtitle_optimizer_module.SubtitleOptimizerNode
    NODE_DISPLAY_NAME_MAPPINGS["SubtitleOptimizer"] = "字幕优化（智能分段）"
    print("[VideoCaptioner] [OK] SubtitleOptimizer loaded")
except Exception as e:
    print(f"[VideoCaptioner] [ERROR] SubtitleOptimizer failed: {e}")

try:
    config_nodes_module = load_node_module("config_nodes", "config_nodes.py")
    NODE_CLASS_MAPPINGS["TranscribeConfig"] = config_nodes_module.TranscribeConfigNode
    NODE_CLASS_MAPPINGS["LLMConfig"] = config_nodes_module.LLMConfigNode
    NODE_CLASS_MAPPINGS["TranslateConfig"] = config_nodes_module.TranslateConfigNode
    NODE_DISPLAY_NAME_MAPPINGS["TranscribeConfig"] = "转录配置"
    NODE_DISPLAY_NAME_MAPPINGS["LLMConfig"] = "LLM 配置"
    NODE_DISPLAY_NAME_MAPPINGS["TranslateConfig"] = "翻译配置"
    print("[VideoCaptioner] [OK] Config nodes loaded (3 nodes)")
except Exception as e:
    print(f"[VideoCaptioner] [ERROR] Config nodes failed: {e}")

try:
    utility_nodes_module = load_node_module("utility_nodes", "utility_nodes.py")
    NODE_CLASS_MAPPINGS["LoadVideo"] = utility_nodes_module.LoadVideoNode
    NODE_DISPLAY_NAME_MAPPINGS["LoadVideo"] = "加载视频"
    print("[VideoCaptioner] [OK] Utility nodes loaded (1 node)")
except Exception as e:
    print(f"[VideoCaptioner] [ERROR] Utility nodes failed: {e}")

# 音频增强与保存节点已移除

try:
    subtitle_text_processor_module = load_node_module("subtitle_text_processor_node", "subtitle_text_processor_node.py")
    NODE_CLASS_MAPPINGS["SubtitleTextProcessor"] = subtitle_text_processor_module.SubtitleTextProcessorNode
    NODE_DISPLAY_NAME_MAPPINGS["SubtitleTextProcessor"] = "字幕文本处理器"
    print("[VideoCaptioner] [OK] SubtitleTextProcessor loaded")
except Exception as e:
    print(f"[VideoCaptioner] [ERROR] SubtitleTextProcessor failed: {e}")

# 输出加载统计
loaded_count = len(NODE_CLASS_MAPPINGS)
print(f"[VideoCaptioner] Successfully loaded {loaded_count} nodes")

if loaded_count == 0:
    print("[VideoCaptioner] ERROR: No nodes were loaded! Please check:")
    print(f"  1. VideoCaptioner directory exists: {videocaptioner_path}")
    print(f"  2. Install dependencies: pip install -r requirements-minimal.txt")
    print(f"  3. Check the error messages above")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
