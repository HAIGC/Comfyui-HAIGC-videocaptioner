"""
基础节点 - 提供依赖检查和错误处理
"""

import sys
from pathlib import Path

def setup_videocaptioner_path():
    """设置 VideoCaptioner 路径"""
    current_dir = Path(__file__).parent.parent
    videocaptioner_path = current_dir / "VideoCaptioner"
    
    print(f"[VideoCaptioner] base_node setup:")
    print(f"  Current dir: {current_dir}")
    print(f"  VideoCaptioner path: {videocaptioner_path}")
    print(f"  Path exists: {videocaptioner_path.exists()}")
    
    if not videocaptioner_path.exists():
        raise ImportError(f"VideoCaptioner directory not found at: {videocaptioner_path}")
    
    # 清除可能冲突的 app 模块（ComfyUI 也有一个 app 模块）
    print(f"  Clearing conflicting modules...")
    modules_to_remove = [k for k in list(sys.modules.keys()) if k == 'app' or k.startswith('app.')]
    for mod in modules_to_remove:
        print(f"    Removing: {mod}")
        del sys.modules[mod]
    
    # 确保路径在 sys.path 最前面
    videocaptioner_str = str(videocaptioner_path)
    if videocaptioner_str in sys.path:
        sys.path.remove(videocaptioner_str)
    sys.path.insert(0, videocaptioner_str)
    
    print(f"  Added to sys.path[0]: {sys.path[0]}")
    
    # 验证能否导入
    try:
        import app
        print(f"  app module location: {app.__file__ if hasattr(app, '__file__') else 'no __file__'}")
        import app.core
        print(f"  app.core imported successfully")
    except Exception as e:
        import traceback
        print(f"  [ERROR] Import test failed: {e}")
        print(f"  详细错误:")
        traceback.print_exc()
        raise ImportError(f"无法导入 VideoCaptioner 模块: {e}")
    
    return videocaptioner_path

def check_dependencies():
    """检查必要的依赖"""
    missing_deps = []
    optional_missing = []
    
    # 检查核心依赖
    required_deps = {
        'sqlalchemy': 'sqlalchemy',
        'json_repair': 'json-repair',
        'requests': 'requests',
    }
    
    for module_name, package_name in required_deps.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_deps.append(package_name)
    
    # 可选依赖
    optional_deps = {
        'openai': 'openai',
        'faster_whisper': 'faster-whisper',
        'whisper': 'openai-whisper',
        'huggingface_hub': 'huggingface-hub',
        'demucs': 'demucs',
        'retry': 'retry',
        'scipy': 'scipy',
    }
    for module_name, package_name in optional_deps.items():
        try:
            __import__(module_name)
        except ImportError:
            optional_missing.append(package_name)
    
    if missing_deps:
        deps_str = ' '.join(missing_deps)
        raise ImportError(
            f"缺少依赖包: {', '.join(missing_deps)}\n"
            f"请在 ComfyUI 的 Python 环境中安装：\n"
            f"  python -m pip install {deps_str}\n"
            f"或运行安装脚本: install_requirements.bat"
        )
    
    if optional_missing:
        print(f"[VideoCaptioner] 可选依赖未安装: {', '.join(optional_missing)}")
    
    print(f"[VideoCaptioner] All core dependencies check passed")
    return True

