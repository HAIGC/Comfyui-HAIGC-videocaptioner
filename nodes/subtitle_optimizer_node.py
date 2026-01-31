"""
字幕优化节点 - 提供强大的字幕分段和优化功能
支持多种优化模式和自定义参数
"""

import sys
from pathlib import Path
from typing import Tuple, Any

# 添加 VideoCaptioner 路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir / "VideoCaptioner"))

try:
    from app.core.bk_asr.asr_data import ASRData
    from app.core.utils.optimize_subtitles import optimize_subtitles, count_words
except ImportError as e:
    print(f"[SubtitleOptimizer] 导入失败: {e}")
    ASRData = None


class SubtitleOptimizerNode:
    """
    字幕优化节点 - 智能分段和优化
    
    支持多种优化模式：
    - 智能模式：自动优化短句
    - 自定义模式：手动设置参数
    - 平衡模式：在质量和速度间平衡
    - 激进模式：最大化合并
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "字幕数据": ("SUBTITLE_DATA",),  # 输入字幕数据
                "优化模式": ([
                    "智能模式 ✨",
                    "按字数限制 📏",
                    "自定义模式 🎛️", 
                    "平衡模式 ⚖️",
                    "激进模式 🚀",
                    "关闭优化 ❌"
                ], {
                    "default": "智能模式 ✨"
                }),
                # 自定义参数（仅在自定义模式下生效）
                "最小词数阈值": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "number",
                    "tooltip": "短句阈值：词数≤此值会尝试合并"
                }),
                "最大词数限制": ("INT", {
                    "default": 10,
                    "min": 5,
                    "max": 50,
                    "step": 1,
                    "display": "number",
                    "tooltip": "长句限制：合并后不超过此词数"
                }),
                "时间间隔": ("INT", {
                    "default": 300,
                    "min": 0,
                    "max": 2000,
                    "step": 10,
                    "display": "number",
                    "tooltip": "时间间隔阈值（毫秒）：智能模式=合并阈值，按停顿分段=停顿阈值"
                }),
                "合并策略": ([
                    "保守合并",
                    "标准合并", 
                    "积极合并"
                ], {
                    "default": "标准合并"
                }),
                # 字数限制参数（用于"按字数限制"模式）
                "每段最大字符数": ("INT", {
                    "default": 40,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "tooltip": "每段最大字符数（硬限制），符合字幕标准：单行15-20字，双行30-40字"
                }),
                "分段阈值": ("INT", {
                    "default": 35,
                    "min": 10,
                    "max": 90,
                    "step": 1,
                    "display": "number",
                    "tooltip": "超过此字符数时开始寻找分段点（软限制），应小于最大字符数"
                }),
            }
        }
    
    RETURN_TYPES = ("SUBTITLE_DATA", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = (
        "优化后字幕",              # 优化后的字幕对象
        "优化报告",             # 优化报告
        "完整时间戳文本",    # 完整时间戳 [HH:MM:SS.mmm]
        "简洁时间戳文本",  # 简洁秒数 (0.0, 1.5)
        "SRT格式文本",        # SRT 格式
        "JSON格式文本",       # JSON 格式
        "CSV格式文本",        # CSV 格式
    )
    FUNCTION = "optimize"
    CATEGORY = "video/subtitle"
    
    def optimize(
        self,
        字幕数据: Any,
        优化模式: str,
        最小词数阈值: int,
        最大词数限制: int,
        时间间隔: int,
        合并策略: str,
        每段最大字符数: int,
        分段阈值: int,
    ) -> Tuple[Any, str, str, str, str, str, str]:
        """
        优化字幕分段
        
        Args:
            subtitle_data: 输入的字幕数据对象
            optimize_mode: 优化模式
            min_word_threshold: 最小词数阈值
            max_word_limit: 最大词数限制
            time_gap_ms: 时间间隔阈值（毫秒）
            merge_strategy: 合并策略
            
        Returns:
            (优化后的字幕数据, 优化报告, 5种格式化文本)
        """
        
        if not 字幕数据 or not hasattr(字幕数据, 'segments'):
            empty_str = ""
            return (字幕数据, "错误：无效的字幕数据", empty_str, empty_str, empty_str, empty_str, empty_str)
        
        # 保存原始段数
        original_count = len(字幕数据.segments)
        
        # 根据模式决定是否优化
        if "关闭" in 优化模式:
            report = f"✋ 优化已关闭\n原始段数: {original_count}"
            print(f"[SubtitleOptimizer] {report}")
            # 生成格式化文本
            text_with_timestamp = self._format_text_with_timestamp(字幕数据)
            text_simple_timestamp = self._format_text_with_simple_timestamp(字幕数据)
            text_srt = self._format_text_srt(字幕数据)
            text_json = self._format_text_json(字幕数据)
            text_csv = self._format_text_csv(字幕数据)
            return (字幕数据, report, text_with_timestamp, text_simple_timestamp, text_srt, text_json, text_csv)
        
        # 复制数据以避免修改原始数据
        from copy import deepcopy
        optimized_data = deepcopy(字幕数据)
        
        # 根据模式设置参数
        if "智能" in 优化模式:
            # 智能模式：使用默认优化算法
            params = {
                "min_words": 4,
                "max_words": 10,
                "time_gap": 100
            }
            print(f"[SubtitleOptimizer] 智能模式: min={params['min_words']}, max={params['max_words']}, gap={params['time_gap']}ms")
            # 先按标点分段，确保每句成为独立行
            self._split_by_punctuation(optimized_data)
            self._optimize_with_params(optimized_data, params)
            
        elif "字数" in 优化模式:
            # 按字数限制分段模式：先按标点分段，再按字数限制进一步切分
            print(f"[SubtitleOptimizer] 按字数限制分段模式: max={每段最大字符数}字, threshold={分段阈值}字")
            self._split_by_char_limit(optimized_data, 每段最大字符数, 分段阈值)
            
        elif "平衡" in 优化模式:
            # 平衡模式：适度合并
            params = {
                "min_words": 3,
                "max_words": 12,
                "time_gap": 150
            }
            print(f"[SubtitleOptimizer] 平衡模式: min={params['min_words']}, max={params['max_words']}, gap={params['time_gap']}ms")
            self._optimize_with_params(optimized_data, params)
            
        elif "激进" in 优化模式:
            # 激进模式：最大化合并
            params = {
                "min_words": 6,
                "max_words": 15,
                "time_gap": 300
            }
            print(f"[SubtitleOptimizer] 激进模式: min={params['min_words']}, max={params['max_words']}, gap={params['time_gap']}ms")
            self._optimize_with_params(optimized_data, params)
            
        elif "自定义" in 优化模式:
            # 自定义模式：使用用户参数
            params = {
                "min_words": 最小词数阈值,
                "max_words": 最大词数限制,
                "time_gap": 时间间隔
            }
            print(f"[SubtitleOptimizer] 自定义模式: min={params['min_words']}, max={params['max_words']}, gap={params['time_gap']}ms")
            print(f"[SubtitleOptimizer] 合并策略: {合并策略}")
            
            # 根据合并策略调整参数
            if 合并策略 == "保守合并":
                params["min_words"] = max(1, params["min_words"] - 1)
                params["time_gap"] = int(params["time_gap"] * 0.7)
            elif 合并策略 == "积极合并":
                params["min_words"] = params["min_words"] + 1
                params["max_words"] = params["max_words"] + 3
                params["time_gap"] = int(params["time_gap"] * 1.5)
            
            self._optimize_with_params(optimized_data, params)
        
        # 生成优化后的段数
        optimized_count = len(optimized_data.segments)
        reduction = original_count - optimized_count
        reduction_pct = (reduction / original_count * 100) if original_count > 0 else 0
        
        # 生成报告
        mode_name = 优化模式.split()[0]
        report = self._generate_report(
            mode_name,
            original_count,
            optimized_count,
            reduction,
            reduction_pct,
            optimized_data
        )
        
        print(f"[SubtitleOptimizer] 优化完成: {original_count} → {optimized_count} 段 (↓{reduction_pct:.1f}%)")
        
        # 生成各种格式的时间戳文本
        text_with_timestamp = self._format_text_with_timestamp(optimized_data)
        text_simple_timestamp = self._format_text_with_simple_timestamp(optimized_data)
        text_srt = self._format_text_srt(optimized_data)
        text_json = self._format_text_json(optimized_data)
        text_csv = self._format_text_csv(optimized_data)
        
        print(f"[SubtitleOptimizer] 已生成 5 种格式化输出")
        
        return (optimized_data, report, text_with_timestamp, text_simple_timestamp, text_srt, text_json, text_csv)
    
    def _optimize_with_params(self, asr_data: Any, params: dict):
        """
        使用指定参数优化字幕
        
        Args:
            asr_data: ASRData 对象
            params: 参数字典 {min_words, max_words, time_gap}
        """
        segments = asr_data.segments
        i = len(segments) - 1
        
        while i > 0:
            prev_seg = segments[i - 1]
            curr_seg = segments[i]
            
            # 计算词数
            prev_words = count_words(prev_seg.text)
            curr_words = count_words(curr_seg.text)
            
            # 计算时间间隔
            time_gap = abs(curr_seg.start_time - prev_seg.end_time)
            
            # 判断是否合并
            punctuation_marks = '。！？；，、：.!?;,:'
            ends_with_punct = False
            try:
                t = prev_seg.text.strip()
                ends_with_punct = len(t) > 0 and t[-1] in punctuation_marks
            except Exception:
                ends_with_punct = False
            should_merge = (
                prev_words <= params["min_words"] and
                time_gap < params["time_gap"] and
                (prev_words + curr_words) <= params["max_words"] and
                not ends_with_punct
            )
            
            if should_merge:
                # 执行合并
                try:
                    asr_data.merge_with_next_segment(i - 1)
                except Exception as e:
                    print(f"[SubtitleOptimizer] 合并失败: {e}")
            
            i -= 1
    
    def _split_by_punctuation(self, asr_data: Any):
        """
        按标点符号分段
        
        在所有标点符号处拆分字幕段（包括逗号、句号、问号、感叹号等）
        每个标点符号结束一个段落，且标点符号保留在段落末尾
        
        Args:
            asr_data: ASRData 对象（会被原地修改）
        """
        from app.core.bk_asr.asr_data import ASRDataSeg
        
        # 定义所有需要断句的标点符号（中英文）
        # 包含：句号、问号、感叹号、分号、逗号、顿号、冒号
        punctuation_marks = '。！？；，、：.!?;,:'
        
        new_segments = []
        
        for seg in asr_data.segments:
            text = seg.text.strip()
            
            # 如果文本为空，跳过
            if not text:
                continue
            
            # 查找所有标点符号的位置
            sentences = []
            current_sentence = ""
            
            for i, char in enumerate(text):
                current_sentence += char
                # 如果当前字符是标点符号
                if char in punctuation_marks:
                    # 检查下一个字符是否也是标点符号（处理连续标点，如 "..." 或 "？！"）
                    if i + 1 < len(text) and text[i+1] in punctuation_marks:
                        continue
                    
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip())
                    current_sentence = ""
            
            # 添加剩余的文本（如果有）
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            
            # 如果没有找到标点符号（或者整个文本就是一个句子），保持原样
            if len(sentences) == 0:
                new_segments.append(seg)
                continue
            elif len(sentences) == 1:
                # 只有一句话，保持原样
                new_segments.append(seg)
                continue
            
            # 按句子数量分配时间
            segment_duration = seg.end_time - seg.start_time
            total_chars = sum(len(s) for s in sentences)
            
            if total_chars == 0:
                new_segments.append(seg)
                continue
            
            current_time = seg.start_time
            
            for i, sentence in enumerate(sentences):
                # 按字符数比例分配时间
                sentence_chars = len(sentence)
                sentence_duration = int(segment_duration * sentence_chars / total_chars)
                
                # 计算结束时间
                if i == len(sentences) - 1:
                    # 最后一句：使用原始结束时间
                    end_time = seg.end_time
                else:
                    end_time = current_time + sentence_duration
                
                # 避免时间重叠
                if end_time <= current_time:
                    end_time = current_time + 100  # 至少100ms
                # 自动补全结尾标点
                end_puncts = '。！？；，、：.!?;,:'
                need_punct = True
                if len(sentence) > 0 and sentence[-1] in end_puncts:
                    need_punct = False
                if need_punct:
                    is_last = (i == len(sentences) - 1)
                    q_words = ["吗","么","？","?","为何","为什么","怎样","怎么","是否","是不是"]
                    e_words = ["!","！","太","真","非常","极其","特别","好棒","震撼","惊人"]
                    has_q = any(w in sentence for w in q_words)
                    has_e = any(w in sentence for w in e_words)
                    ascii_ratio = sum(1 for c in sentence if c.isascii())/max(len(sentence),1)
                    if ascii_ratio > 0.5:
                        add_p = "?" if has_q else ("!" if has_e else ("." if is_last else ","))
                    else:
                        add_p = "？" if has_q else ("！" if has_e else ("。" if is_last else "，"))
                    sentence = sentence + add_p
                # 创建新的字幕段
                new_seg = ASRDataSeg(
                    start_time=current_time,
                    end_time=end_time,
                    text=sentence
                )
                
                new_segments.append(new_seg)
                current_time = end_time
        
        # 替换原始segments
        asr_data.segments = new_segments
    
    def _split_by_speech_pause(self, asr_data: Any, pause_threshold: int):
        """
        按说话停顿分段
        
        基于语音的自然停顿（时间间隔）来分段
        - 间隔大于阈值：保留断句（说话有停顿）
        - 间隔小于阈值：合并（连续说话），除非遇到标点符号
        
        Args:
            asr_data: ASRData 对象（会被原地修改）
            pause_threshold: 停顿阈值（毫秒），大于此值视为自然停顿
        """
        segments = asr_data.segments
        
        # 统计信息
        merge_count = 0
        keep_count = 0
        punct_break_count = 0
        
        # 定义强制分段的标点符号
        break_punct = '。！？；，、.!?,:;：'
        
        # 从后向前遍历，方便删除和合并
        i = len(segments) - 1
        
        while i > 0:
            prev_seg = segments[i - 1]
            curr_seg = segments[i]
            
            # 计算两段之间的时间间隔
            time_gap = curr_seg.start_time - prev_seg.end_time
            
            # 检查前一段是否以标点符号结尾
            prev_text = prev_seg.text.strip()
            has_break_punct = prev_text and prev_text[-1] in break_punct
            
            # 如果间隔小于阈值，且没有标点分隔，说明是连续说话，应该合并
            if time_gap < pause_threshold:
                if has_break_punct:
                    # 有标点符号，强制分段
                    punct_break_count += 1
                else:
                    try:
                        # 合并前一段和当前段
                        asr_data.merge_with_next_segment(i - 1)
                        merge_count += 1
                    except Exception as e:
                        print(f"[SubtitleOptimizer] 合并失败: {e}")
            else:
                # 间隔大于阈值，保留断句
                keep_count += 1
            
            i -= 1
        
        print(f"[SubtitleOptimizer] 停顿分析: 合并了 {merge_count} 处连续说话, 保留了 {keep_count} 处自然停顿, {punct_break_count} 处标点分段")
    
    def _split_by_char_limit(self, asr_data: Any, max_chars: int, threshold: int):
        """
        按字数限制分段
        
        处理逻辑：
        1. 先按标点符号分段
        2. 检查每段字符数：
           - 如果 ≤ threshold：保持不变
           - 如果 > threshold 且 ≤ max_chars：尝试在标点处优化切分
           - 如果 > max_chars：强制切分（优先在标点/空格处，否则硬切）
        
        Args:
            asr_data: ASRData 对象（会被原地修改）
            max_chars: 每段最大字符数（硬限制）
            threshold: 超过此值时开始寻找分段点（软限制）
        """
        # 第一步：先按标点符号分段
        print(f"[SubtitleOptimizer] 步骤1: 按标点符号分段")
        self._split_by_punctuation(asr_data)
        
        # 第二步：检查并处理超长段落
        print(f"[SubtitleOptimizer] 步骤2: 检查字符数限制")
        self._enforce_char_limit(asr_data, max_chars, threshold)

    def _enforce_char_limit(self, asr_data: Any, max_chars: int, threshold: int):
        """
        强制执行字符数限制（不预先按标点分段）
        """
        from app.core.bk_asr.asr_data import ASRDataSeg
        
        new_segments = []
        split_count = 0
        
        for seg in asr_data.segments:
            text = seg.text.strip()
            char_count = len(text)
            
            # 如果字符数在阈值内，直接保留
            if char_count <= threshold:
                new_segments.append(seg)
                continue
            
            # 如果超过阈值，需要分段
            if char_count > max_chars:
                # 超过硬限制，强制分段
                print(f"[SubtitleOptimizer]   超长段落({char_count}字): '{text[:20]}...' - 强制切分")
                sub_segs = self._force_split_segment(seg, max_chars)
                new_segments.extend(sub_segs)
                split_count += len(sub_segs) - 1
            else:
                # 在阈值和最大值之间，尝试智能分段
                print(f"[SubtitleOptimizer]   偏长段落({char_count}字): '{text[:20]}...' - 尝试优化")
                sub_segs = self._smart_split_segment(seg, threshold, max_chars)
                new_segments.extend(sub_segs)
                if len(sub_segs) > 1:
                    split_count += len(sub_segs) - 1
        
        # 替换原始segments
        asr_data.segments = new_segments
        print(f"[SubtitleOptimizer] 字数限制分析: 切分了 {split_count} 个超长段落")
    
    def _smart_split_segment(self, seg: Any, threshold: int, max_chars: int) -> list:
        """
        智能切分段落（在阈值和最大值之间）
        
        尝试在合适的标点符号处切分，使每段长度更合理
        
        Args:
            seg: 要切分的字幕段
            threshold: 软限制
            max_chars: 硬限制
            
        Returns:
            切分后的字幕段列表
        """
        from app.core.bk_asr.asr_data import ASRDataSeg
        
        text = seg.text.strip()
        
        # 定义主要标点符号（优先在这些位置切分）
        major_punctuation = '。！？；.!?;'
        # 次要标点符号（如果没有主要标点，可以在这里切分）
        minor_punctuation = '，、：,:：'
        
        # 寻找最佳切分点
        best_split_pos = -1
        
        # 优先在 threshold 附近寻找主要标点
        for i in range(threshold - 5, min(threshold + 5, len(text))):
            if i > 0 and i < len(text) and text[i] in major_punctuation:
                best_split_pos = i + 1  # 标点后切分
                break
        
        # 如果没找到主要标点，寻找次要标点
        if best_split_pos == -1:
            for i in range(threshold - 5, min(threshold + 5, len(text))):
                if i > 0 and i < len(text) and text[i] in minor_punctuation:
                    best_split_pos = i + 1
                    break
        
        # 如果都没找到，就在 threshold 位置强制切分
        if best_split_pos == -1:
            # 尝试在空格处切分
            for i in range(threshold - 3, min(threshold + 3, len(text))):
                if i > 0 and i < len(text) and text[i] == ' ':
                    best_split_pos = i + 1
                    break
        
        # 如果还是没找到，在 threshold 位置硬切
        if best_split_pos == -1:
            best_split_pos = threshold
        
        # 如果切分点太靠后，直接返回原段落
        if best_split_pos >= len(text) - 3:
            return [seg]
        
        # 执行切分
        part1_text = text[:best_split_pos].strip()
        part2_text = text[best_split_pos:].strip()
        
        if not part1_text or not part2_text:
            return [seg]
        
        # 计算时间分配（按字符数比例）
        total_duration = seg.end_time - seg.start_time
        part1_ratio = len(part1_text) / len(text)
        part1_duration = int(total_duration * part1_ratio)
        
        mid_time = seg.start_time + part1_duration
        
        # 创建两个新段（ASRDataSeg 参数顺序：text, start_time, end_time）
        seg1 = ASRDataSeg(
            text=part1_text,
            start_time=seg.start_time,
            end_time=mid_time
        )
        
        seg2 = ASRDataSeg(
            text=part2_text,
            start_time=mid_time,
            end_time=seg.end_time
        )
        
        return [seg1, seg2]
    
    def _force_split_segment(self, seg: Any, max_chars: int) -> list:
        """
        强制切分超长段落
        
        按照 max_chars 切分，优先在标点符号或空格处切分
        
        Args:
            seg: 要切分的字幕段
            max_chars: 每段最大字符数
            
        Returns:
            切分后的字幕段列表
        """
        from app.core.bk_asr.asr_data import ASRDataSeg
        
        text = seg.text.strip()
        total_duration = seg.end_time - seg.start_time
        
        # 所有标点符号
        all_punctuation = '。！？；，、：.!?;,:'
        
        segments = []
        current_pos = 0
        current_time = seg.start_time
        
        while current_pos < len(text):
            # 确定本段的结束位置
            end_pos = min(current_pos + max_chars, len(text))
            
            # 如果不是最后一段，尝试在标点或空格处切分
            if end_pos < len(text):
                # 向前搜索最近的标点符号
                best_cut = end_pos
                for i in range(end_pos - 1, max(current_pos + max_chars // 2, end_pos - 10), -1):
                    if text[i] in all_punctuation:
                        best_cut = i + 1  # 标点后切分
                        break
                    elif text[i] == ' ':
                        best_cut = i + 1  # 空格后切分（如果没找到标点）
                
                end_pos = best_cut
            
            # 提取文本
            segment_text = text[current_pos:end_pos].strip()
            
            if segment_text:
                # 计算时间（按字符数比例）
                char_ratio = len(segment_text) / len(text)
                segment_duration = int(total_duration * char_ratio)
                segment_end_time = current_time + segment_duration
                
                # 最后一段使用原始结束时间
                if end_pos >= len(text):
                    segment_end_time = seg.end_time
                
                # 创建新段（ASRDataSeg 参数顺序：text, start_time, end_time）
                new_seg = ASRDataSeg(
                    text=segment_text,
                    start_time=current_time,
                    end_time=segment_end_time
                )
                
                segments.append(new_seg)
                current_time = segment_end_time
            
            current_pos = end_pos
        
        return segments if segments else [seg]
    
    def _append_punctuation(self, asr_data: Any):
        end_puncts = '。！？；，、：.!?;,:'
        q_words = ["吗","么","为何","为什么","怎样","怎么","是否","是不是","能否","可否","？","?"]
        e_words = ["！","!","太","真","非常","极其","特别","好棒","震撼","惊人","厉害","精彩","快看","注意"]
        for seg in asr_data.segments:
            t = seg.text.strip()
            if not t:
                continue
            if t[-1] in end_puncts:
                continue
            has_q = any(w in t for w in q_words)
            has_e = any(w in t for w in e_words)
            ascii_ratio = sum(1 for c in t if c.isascii())/max(len(t),1)
            if ascii_ratio > 0.5:
                add_p = "?" if has_q else ("!" if has_e else ".")
            else:
                add_p = "？" if has_q else ("！" if has_e else "。")
            seg.text = t + add_p
    
    def _append_punctuation_by_pause(self, asr_data: Any, pause_threshold_ms: int):
        end_puncts = '。！？；，、：.!?;,:'
        q_words = ["吗","么","为何","为什么","怎样","怎么","是否","是不是","能否","可否","？","?"]
        e_words = ["！","!","太","真","非常","极其","特别","好棒","震撼","惊人","厉害","精彩","快看","注意"]
        segs = asr_data.segments
        n = len(segs)
        for i, seg in enumerate(segs):
            text = seg.text.strip()
            if not text:
                continue
            if text[-1] in end_puncts:
                continue
            next_gap = None
            if i < n - 1:
                next_gap = segs[i+1].start_time - seg.end_time
            # 问句/感叹优先
            has_q = any(w in text for w in q_words)
            has_e = any(w in text for w in e_words)
            ascii_ratio = sum(1 for c in text if c.isascii())/max(len(text),1)
            if ascii_ratio > 0.5:
                comma = ","
                dot = "."
                q = "?"
                e = "!"
            else:
                comma = "，"
                dot = "。"
                q = "？"
                e = "！"
            if has_q:
                seg.text = text + q
            elif has_e:
                seg.text = text + e
            else:
                # 根据自然停顿判断逗号/句号
                if next_gap is not None and next_gap < pause_threshold_ms:
                    seg.text = text + comma
                else:
                    seg.text = text + dot
    
    def _generate_report(
        self, 
        mode: str, 
        original: int, 
        optimized: int, 
        reduction: int, 
        pct: float,
        asr_data: Any
    ) -> str:
        """生成优化报告"""
        
        # 计算平均长度
        if optimized > 0:
            total_duration = sum(
                seg.end_time - seg.start_time 
                for seg in asr_data.segments
            )
            avg_duration = total_duration / optimized / 1000  # 转换为秒
        else:
            avg_duration = 0
        
        # 统计词数分布
        word_counts = [count_words(seg.text) for seg in asr_data.segments]
        short_count = sum(1 for w in word_counts if w <= 3)
        medium_count = sum(1 for w in word_counts if 4 <= w <= 10)
        long_count = sum(1 for w in word_counts if w > 10)
        
        report = f"""
═══════════════════════════════════════
         字幕优化报告
═══════════════════════════════════════

🎯 优化模式: {mode}

📊 段数统计:
  • 优化前: {original} 段
  • 优化后: {optimized} 段
  • 减少: {reduction} 段 (↓{pct:.1f}%)

⏱️  平均时长:
  • 每段: {avg_duration:.2f} 秒

📝 词数分布:
  • 短句 (≤3词): {short_count} 段 ({short_count/optimized*100:.1f}%)
  • 中句 (4-10词): {medium_count} 段 ({medium_count/optimized*100:.1f}%)
  • 长句 (>10词): {long_count} 段 ({long_count/optimized*100:.1f}%)

✅ 优化状态: 完成

═══════════════════════════════════════
"""
        return report.strip()
    
    @staticmethod
    def _format_text_with_timestamp(asr_data: Any) -> str:
        """
        将 ASRData 格式化为带时间戳的文本（完整格式）
        
        格式示例：
        [00:00:01.000 --> 00:00:03.000] 你好世界
        [00:00:03.500 --> 00:00:05.200] 这是第二句话
        
        Args:
            asr_data: ASRData 对象
            
        Returns:
            带时间戳的文本字符串
        """
        lines = []
        
        for segment in asr_data.segments:
            # 格式化时间戳 (毫秒 -> HH:MM:SS.mmm)
            start_ms = segment.start_time
            end_ms = segment.end_time
            
            # 转换为 HH:MM:SS.mmm 格式
            def ms_to_timestamp(ms: int) -> str:
                total_seconds = ms // 1000
                milliseconds = ms % 1000
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
            
            start_time_str = ms_to_timestamp(start_ms)
            end_time_str = ms_to_timestamp(end_ms)
            
            # 格式: [开始时间 --> 结束时间] 文本
            line = f"[{start_time_str} --> {end_time_str}] {segment.text}"
            lines.append(line)
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_text_with_simple_timestamp(asr_data: Any) -> str:
        """
        将 ASRData 格式化为简洁时间戳文本（秒数格式）
        
        格式示例：
        (0.0, 0.26) 大难，
        (0.3, 1.4) 我来参加投稿了，
        (1.5, 2.26) 快告诉我，
        
        Args:
            asr_data: ASRData 对象
            
        Returns:
            简洁时间戳文本字符串
        """
        lines = []
        
        for segment in asr_data.segments:
            # 转换为秒（保留2位小数）
            start_seconds = segment.start_time / 1000.0
            end_seconds = segment.end_time / 1000.0
            
            # 格式: (开始秒, 结束秒) 文本
            # 去除不必要的小数位（如果是整数就显示整数）
            start_str = f"{start_seconds:.2f}".rstrip('0').rstrip('.')
            end_str = f"{end_seconds:.2f}".rstrip('0').rstrip('.')
            
            line = f"({start_str}, {end_str}) {segment.text}"
            lines.append(line)
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_text_srt(asr_data: Any) -> str:
        """
        将 ASRData 格式化为 SRT 字幕格式
        
        格式示例：
        1
        00:00:00,000 --> 00:00:00,260
        大难，
        
        2
        00:00:00,300 --> 00:00:01,400
        我来参加投稿了，
        
        Args:
            asr_data: ASRData 对象
            
        Returns:
            SRT 格式字符串
        """
        lines = []
        
        for i, segment in enumerate(asr_data.segments, 1):
            # 转换为 SRT 时间格式 (HH:MM:SS,mmm)
            def ms_to_srt_time(ms: int) -> str:
                total_seconds = ms // 1000
                milliseconds = ms % 1000
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
            
            start_time = ms_to_srt_time(segment.start_time)
            end_time = ms_to_srt_time(segment.end_time)
            
            # SRT 格式：序号、时间戳、文本、空行
            lines.append(str(i))
            lines.append(f"{start_time} --> {end_time}")
            lines.append(segment.text)
            lines.append("")  # 空行分隔
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_text_json(asr_data: Any) -> str:
        """
        将 ASRData 格式化为 JSON 格式
        
        格式示例：
        [
          {
            "index": 1,
            "start": 0.0,
            "end": 0.26,
            "duration": 0.26,
            "text": "大难，"
          },
          ...
        ]
        
        Args:
            asr_data: ASRData 对象
            
        Returns:
            JSON 格式字符串
        """
        import json
        
        segments = []
        for i, segment in enumerate(asr_data.segments, 1):
            start_seconds = segment.start_time / 1000.0
            end_seconds = segment.end_time / 1000.0
            
            segments.append({
                "index": i,
                "start": round(start_seconds, 3),
                "end": round(end_seconds, 3),
                "duration": round(end_seconds - start_seconds, 3),
                "text": segment.text
            })
        
        return json.dumps(segments, ensure_ascii=False, indent=2)
    
    @staticmethod
    def _format_text_csv(asr_data: Any) -> str:
        """
        将 ASRData 格式化为 CSV 格式
        
        格式示例：
        index,start,end,duration,text
        1,0.0,0.26,0.26,"大难，"
        2,0.3,1.4,1.1,"我来参加投稿了，"
        3,1.5,2.26,0.76,"快告诉我，"
        
        Args:
            asr_data: ASRData 对象
            
        Returns:
            CSV 格式字符串
        """
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # 写入表头
        writer.writerow(["index", "start", "end", "duration", "text"])
        
        # 写入数据
        for i, segment in enumerate(asr_data.segments, 1):
            start_seconds = round(segment.start_time / 1000.0, 3)
            end_seconds = round(segment.end_time / 1000.0, 3)
            duration = round(end_seconds - start_seconds, 3)
            
            writer.writerow([i, start_seconds, end_seconds, duration, segment.text])
        
        return output.getvalue()


NODE_CLASS_MAPPINGS = {
    "SubtitleOptimizerNode": SubtitleOptimizerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleOptimizerNode": "字幕优化 (智能分段)"
}
