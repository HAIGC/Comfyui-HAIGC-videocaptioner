"""
字幕文本处理节点 - 提供符号去除、文本排序、内容替换等功能
"""

import re
from typing import Tuple, Any, List
import unicodedata

try:
    from .base_node import setup_videocaptioner_path
    setup_videocaptioner_path()
    from app.core.bk_asr.asr_data import ASRData, ASRDataSeg
    DEPENDENCIES_OK = True
except Exception as e:
    print(f"[SubtitleTextProcessor] 导入失败: {e}")
    DEPENDENCIES_OK = False
    ASRData = None
    ASRDataSeg = None


class SubtitleTextProcessorNode:
    """
    字幕文本处理节点
    提供多种文本清理和处理功能
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # === 数据输入（二选一） ===
                "字幕数据": ("SUBTITLE_DATA",),
                "文本输入": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "直接输入要处理的文本（与字幕数据二选一）"
                }),
                
                # === 符号处理 ===
                "去除标点符号": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除所有中英文标点符号"
                }),
                "去除特殊符号": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除 @#$%^&* 等特殊字符"
                }),
                "去除表情符号": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除 Emoji 表情"
                }),
                "去除数字": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除所有数字"
                }),
                "去除英文": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除所有英文字母"
                }),
                
                # === 括号内容处理 ===
                "去除圆括号内容": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除 (内容) 和 （内容）及括号本身"
                }),
                "去除方括号内容": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除 [内容] 和 【内容】及括号本身"
                }),
                "去除花括号内容": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除 {内容} 及括号本身"
                }),
                "去除书名号内容": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除 《内容》及书名号本身"
                }),
                "去除双引号内容": ("BOOLEAN", {
                    "default": False,
                    "tooltip": '去除 "内容" 和 "内容" 及引号本身'
                }),
                "去除单引号内容": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除 '内容' 和 '内容' 及引号本身"
                }),
                "去除所有括号内容": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "一键去除所有类型的括号及其内容"
                }),
                
                # === 空白处理 ===
                "去除多余空格": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "将多个连续空格合并为一个"
                }),
                "去除首尾空格": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "去除每行开头和结尾的空格"
                }),
                "去除所有空格": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "去除文本中的所有空格"
                }),
                
                # === 行处理 ===
                "去除空行": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "去除空白的字幕行"
                }),
                "合并重复行": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "合并相邻的重复字幕"
                }),
                "按时间排序": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "按时间戳重新排序字幕"
                }),
                "去除短字幕": (["不限制", "少于2字", "少于3字", "少于5字"], {
                    "default": "不限制",
                    "tooltip": "去除过短的字幕行"
                }),
                
                # === 内容替换 ===
                "替换规则": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "每行一个替换规则，格式：旧文本>>新文本\n例如：嗯>>，啊>>，那个>>"
                }),
                "批量替换文本": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "按行批量替换字幕内容（需多行文本）\n支持自动识别时间戳格式 (0.00, 1.00) 文本\n或者按行顺序替换"
                }),
                "匹配策略": (["按时间戳智能合并", "按字数匹配替换", "按顺序强制替换", "按整段智能分行"], {
                    "default": "按时间戳智能合并",
                    "tooltip": "选择批量替换的匹配策略：\n- 按时间戳智能合并：适合带有准确时间戳的文本，会自动合并/覆盖原有字幕\n- 按字数匹配替换：忽略时间戳，根据每行字数在原字幕中寻找匹配项（推荐用于校对）\n- 按顺序强制替换：忽略内容差异，直接按行顺序覆盖\n- 按整段智能分行：将输入文本合并为一段，并根据原字幕各行时长比例自动分配文字"
                }),
                "字数容差": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "按字数匹配时的允许误差范围（默认 ±3 个字）"
                }),
                "大小写转换": (["不转换", "全部大写", "全部小写", "首字母大写"], {
                    "default": "不转换"
                }),
                
                # === 过滤词 ===
                "删除指定词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "每行一个要删除的词语，支持正则表达式\n例如：嗯\n啊\n那个"
                }),
                
                # === 高级选项 ===
                "保留换行符": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "处理后是否保留原有的换行结构"
                }),
            }
        }
    
    RETURN_TYPES = ("SUBTITLE_DATA", "STRING", "STRING")
    RETURN_NAMES = ("处理后字幕", "处理后文本", "处理统计")
    FUNCTION = "process_text"
    CATEGORY = "video/subtitle"
    
    def process_text(
        self,
        字幕数据: Any = None,
        文本输入: str = "",
        去除标点符号: bool = False,
        去除特殊符号: bool = False,
        去除表情符号: bool = False,
        去除数字: bool = False,
        去除英文: bool = False,
        去除圆括号内容: bool = False,
        去除方括号内容: bool = False,
        去除花括号内容: bool = False,
        去除书名号内容: bool = False,
        去除双引号内容: bool = False,
        去除单引号内容: bool = False,
        去除所有括号内容: bool = False,
        去除多余空格: bool = True,
        去除首尾空格: bool = True,
        去除所有空格: bool = False,
        去除空行: bool = True,
        合并重复行: bool = False,
        按时间排序: bool = False,
        去除短字幕: str = "不限制",
        替换规则: str = "",
        批量替换文本: str = "",
        匹配策略: str = "按时间戳智能合并",
        字数容差: int = 3,
        大小写转换: str = "不转换",
        删除指定词: str = "",
        保留换行符: bool = True,
    ) -> Tuple[Any, str, str]:
        """
        处理字幕文本
        """
        if not DEPENDENCIES_OK or ASRData is None:
            raise RuntimeError("字幕文本处理节点依赖加载失败")
        
        # 判断输入类型
        is_text_mode = False
        if 文本输入 and 文本输入.strip():
            # 文本模式
            is_text_mode = True
            print(f"[SubtitleTextProcessor] 文本模式：处理纯文本输入")
            
            # 将文本按行分割，创建简单的字幕段
            lines = 文本输入.strip().split('\n')
            processed_segments = []
            for i, line in enumerate(lines):
                seg = ASRDataSeg(
                    text=line,
                    start_time=i * 1000,  # 虚拟时间戳（毫秒）
                    end_time=(i + 1) * 1000,
                    translated_text=""
                )
                processed_segments.append(seg)
            original_count = len(processed_segments)
            
        elif 字幕数据 is not None:
            # 字幕数据模式
            if not isinstance(字幕数据, ASRData):
                raise TypeError(f"字幕数据必须是 ASRData 对象，当前类型: {type(字幕数据)}")
            print(f"[SubtitleTextProcessor] 字幕模式：处理字幕数据")
            
            original_count = len(字幕数据.segments)
            
            # 复制字幕段以避免修改原始数据
            processed_segments = []
            for seg in 字幕数据.segments:
                new_seg = ASRDataSeg(
                    text=seg.text,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    translated_text=seg.translated_text if hasattr(seg, 'translated_text') else ""
                )
                processed_segments.append(new_seg)
        else:
            raise ValueError("请提供字幕数据或文本输入（至少选择一个）")
        
        removed_count = 0
        modified_count = 0
        
        # 准备批量替换数据
        batch_repl_lines = []
        new_segments_from_batch = [] # List of ASRDataSeg
        has_timestamps = False
        
        if 批量替换文本.strip():
            raw_lines = [l.strip() for l in 批量替换文本.strip().split('\n') if l.strip()]
            
            # 尝试检测时间戳格式 (0.00, 1.00) 文本
            timestamp_pattern = re.compile(r'^[\[\(](\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)[\]\)]\s*(.*)$')
            
            for line in raw_lines:
                match = timestamp_pattern.match(line)
                if match:
                    try:
                        start_t = float(match.group(1)) * 1000  # 转毫秒
                        end_t = float(match.group(2)) * 1000    # 转毫秒
                        content = match.group(3).strip()
                        
                        if content:
                            new_seg = ASRDataSeg(
                                text=content,
                                start_time=start_t,
                                end_time=end_t,
                                translated_text=""
                            )
                            new_segments_from_batch.append(new_seg)
                            has_timestamps = True
                            
                        # 无论是纯文本还是带时间戳的，都添加到批量替换列表中，以支持非智能合并模式
                        # 即使内容为空（表示删除），也添加进去，以便在"强制替换"或"字数匹配"模式下能正确对应行号
                        batch_repl_lines.append(content)
                    except:
                        batch_repl_lines.append(line)
                else:
                    batch_repl_lines.append(line)
                
            if has_timestamps:
                print(f"[SubtitleTextProcessor] 检测到 {len(new_segments_from_batch)} 条带时间戳的有效替换文本")
        
        # 根据策略执行预处理
        # 策略 1: 按时间戳智能合并
        if 匹配策略 == "按时间戳智能合并" and has_timestamps:
             print("[SubtitleTextProcessor] 执行智能合并策略...")
             # 1. 找出所有会被新段落"覆盖"的原始段落，并标记删除
             segments_to_keep = []
             removed_count = 0
             
             for original_seg in processed_segments:
                 should_remove = False
                 for new_seg in new_segments_from_batch:
                     # 计算重叠
                     start_max = max(original_seg.start_time, new_seg.start_time)
                     end_min = min(original_seg.end_time, new_seg.end_time)
                     overlap = max(0, end_min - start_max)
                     
                     original_duration = original_seg.end_time - original_seg.start_time
                     if original_duration <= 0: original_duration = 1 # 避免除零
                     
                     # 如果重叠超过 50% 或者 超过 500ms (且占有一定比例)
                     if (overlap > original_duration * 0.5) or (overlap > 500 and overlap > original_duration * 0.3):
                         should_remove = True
                         removed_count += 1
                         break
                 
                 if not should_remove:
                     segments_to_keep.append(original_seg)
             
             # 2. 合并保留的原始段落和新段落
             processed_segments = segments_to_keep + new_segments_from_batch
             # 3. 重新排序
             processed_segments.sort(key=lambda x: x.start_time)
             print(f"[SubtitleTextProcessor] 智能合并完成: 移除了 {removed_count} 个旧段落, 添加了 {len(new_segments_from_batch)} 个新段落")
             
             # 在这种模式下，我们不需要后续的"批量替换"循环来修改文本，
             # 因为文本已经包含在 new_segments_from_batch 中了。
             # 我们清空 batch_repl_lines 以跳过后续逻辑
             batch_repl_lines = [] 

        
        batch_repl_cursor = 0
        batch_replaced_count = 0 
        
        # 策略 2: 按字数匹配替换
        if 匹配策略 == "按字数匹配替换" and batch_repl_lines:
            print(f"[SubtitleTextProcessor] 执行按字数匹配策略 (容差 ±{字数容差})...")
            seg_idx = 0
            match_count = 0
            
            for repl_text in batch_repl_lines:
                # 在当前及之后的段落中寻找匹配
                # 限制查找范围，比如往后找 10 个，避免过度跳跃
                found_match = False
                search_limit = 10 
                best_match_idx = -1
                
                # 第一次扫描：寻找字数符合容差的最近匹配
                for i in range(seg_idx, min(len(processed_segments), seg_idx + search_limit)):
                    seg = processed_segments[i]
                    current_len = len(seg.text.strip())
                    repl_len = len(repl_text.strip())
                    diff = abs(current_len - repl_len)
                    
                    if diff <= 字数容差:
                        # 找到了符合条件的
                        best_match_idx = i
                        found_match = True
                        break
                
                if found_match:
                    processed_segments[best_match_idx].text = repl_text
                    seg_idx = best_match_idx + 1 # 移动游标到匹配项之后
                    match_count += 1
                else:
                    # 如果没找到匹配，为了"一字不漏"，我们强制替换下一个可用段落
                    if seg_idx < len(processed_segments):
                        print(f"Warning: 未找到匹配 '{repl_text}' 的段落，强制替换索引 {seg_idx}")
                        processed_segments[seg_idx].text = repl_text
                        seg_idx += 1
                        match_count += 1
                    else:
                        print(f"Error: 剩余段落不足，无法替换: '{repl_text}'")
            
            print(f"[SubtitleTextProcessor] 字数匹配完成，共替换 {match_count} 行")
            # 清空 batch_repl_lines 以跳过后续逻辑
            batch_repl_lines = []
            
        
        # 策略 3: 按整段智能分行
        if 匹配策略 == "按整段智能分行" and batch_repl_lines:
            print("[SubtitleTextProcessor] 执行整段智能分行策略...")
            
            # 1. 提取纯文本（移除可能的时间戳）并合并
            clean_lines = []
            for line in batch_repl_lines:
                # 尝试去除时间戳前缀
                ts_match = re.match(r'^[\[\(](\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)[\]\)]\s*(.*)$', line)
                if ts_match:
                    content = ts_match.group(3).strip()
                else:
                    content = line.strip()
                if content:
                    clean_lines.append(content)
            
            # 合并文本：直接连接
            full_text = "".join(clean_lines)
            
            # 2. 计算所有段落的总时长权重
            total_duration = 0
            durations = []
            
            for seg in processed_segments:
                dur = seg.end_time - seg.start_time
                if dur <= 0: dur = 100 # 避免无效时长
                durations.append(dur)
                total_duration += dur
            
            # 3. 按比例分配文本
            if total_duration > 0 and full_text:
                current_char_idx = 0
                total_chars = len(full_text)
                accumulated_duration = 0
                
                print(f"[SubtitleTextProcessor] 总时长: {total_duration}ms, 总字数: {total_chars}")
                
                for i, seg in enumerate(processed_segments):
                    # 如果是最后一个段落，直接分配剩余所有文本
                    if i == len(processed_segments) - 1:
                        seg.text = full_text[current_char_idx:]
                    else:
                        # 使用累积时长计算位置，减少累积误差
                        seg_duration = durations[i]
                        accumulated_duration += seg_duration
                        
                        target_end_ratio = accumulated_duration / total_duration
                        target_end_idx = int(round(total_chars * target_end_ratio))
                        
                        # 确保进度向前
                        if target_end_idx <= current_char_idx:
                            # 如果计算结果没有前进（因为段落太短），尝试至少分配一个字
                            # 除非已经分完了
                            if current_char_idx < total_chars:
                                target_end_idx = current_char_idx + 1
                            else:
                                target_end_idx = current_char_idx
                        
                        # 截取文本
                        seg.text = full_text[current_char_idx:target_end_idx]
                        current_char_idx = target_end_idx
                
                print(f"[SubtitleTextProcessor] 智能分行完成，已分配到 {len(processed_segments)} 个段落")
                
                # 清空 batch_repl_lines 以跳过后续逻辑
                batch_repl_lines = []


        # 处理每个字幕段 (常规处理 & 强制逐行/剩余模式)
        for seg in processed_segments:
            original_text = seg.text
            text = seg.text
            
            # 1. 去除括号及其内容（需要在其他处理之前）
            if 去除所有括号内容:
                # 一键去除所有括号
                text = self._remove_brackets_content(text, all_types=True)
            else:
                # 按类型去除
                if 去除圆括号内容:
                    text = re.sub(r'\([^)]*\)', '', text)  # 英文圆括号
                    text = re.sub(r'（[^）]*）', '', text)  # 中文圆括号
                
                if 去除方括号内容:
                    text = re.sub(r'\[[^\]]*\]', '', text)  # 英文方括号
                    text = re.sub(r'【[^】]*】', '', text)  # 中文方括号
                
                if 去除花括号内容:
                    text = re.sub(r'\{[^}]*\}', '', text)  # 花括号
                
                if 去除书名号内容:
                    text = re.sub(r'《[^》]*》', '', text)  # 书名号
                
                if 去除双引号内容:
                    text = re.sub(r'"[^"]*"', '', text)  # 英文双引号
                    text = re.sub(r'"[^"]*"', '', text)  # 中文双引号
                
                if 去除单引号内容:
                    text = re.sub(r"'[^']*'", '', text)  # 英文单引号
                    text = re.sub(r"'[^']*'", '', text)  # 中文单引号
            
            # 2. 去除符号和字符
            if 去除标点符号:
                # 中文标点
                text = re.sub(r'[，。！？；：、""''（）《》【】…—·]', '', text)
                # 英文标点
                text = re.sub(r'[,\.!?;:"\'\(\)\[\]\-]', '', text)
            
            if 去除特殊符号:
                text = re.sub(r'[~`@#$%^&*+={}|\\/<>]', '', text)
            
            if 去除表情符号:
                # 去除 Emoji
                text = self._remove_emoji(text)
            
            if 去除数字:
                text = re.sub(r'\d+', '', text)
            
            if 去除英文:
                text = re.sub(r'[a-zA-Z]+', '', text)
            
            # 2. 删除指定词
            if 删除指定词.strip():
                for word in 删除指定词.strip().split('\n'):
                    word = word.strip()
                    if word:
                        try:
                            # 尝试作为正则表达式
                            text = re.sub(word, '', text)
                        except:
                            # 如果不是有效的正则，作为普通文本替换
                            text = text.replace(word, '')
            
            # 3. 内容替换
            if 替换规则.strip():
                for rule in 替换规则.strip().split('\n'):
                    rule = rule.strip()
                    if '>>' in rule:
                        old, new = rule.split('>>', 1)
                        text = text.replace(old.strip(), new.strip())
            
            # 3.5 批量替换 (Batch Replacement)
            # 如果之前的策略已经处理完了 batch_repl_lines，这里就不会执行
            # 只有当 匹配策略 == "按顺序强制替换" 或者 "按字数匹配替换"但被上面的逻辑跳过（理论上不会）
            # 或者 默认/自动 模式下
            
            # 如果是"按顺序强制替换"，或者其他模式下仍有剩余行
            if len(batch_repl_lines) > 0 and batch_repl_cursor < len(batch_repl_lines):
                # 顺序匹配模式
                repl_line = batch_repl_lines[batch_repl_cursor]
                
                # 去除可能包含的时间戳前缀
                clean_repl_line = repl_line
                ts_match = re.match(r'^[\[\(](\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)[\]\)]\s*(.*)$', repl_line)
                if ts_match:
                    clean_repl_line = ts_match.group(3).strip()
                
                # 决定是否替换
                should_replace = False
                
                if 匹配策略 == "按顺序强制替换":
                    should_replace = True
                elif 匹配策略 == "按字数匹配替换":
                     # 上面的专用逻辑块应该已经处理了，如果走到这里说明有问题，或者我们仅作为 fallback
                     # 如果上面的块清空了 batch_repl_lines，这里就不会执行
                     # 如果没清空（比如为了兼容混合模式？），则执行这里的逻辑
                     # 但我们上面已经清空了。所以这里实际上只服务于 "按顺序强制替换" 或者未知的默认模式
                     pass
                else:
                    # 默认/旧逻辑 fallback (如果策略选了其他但也没命中?)
                    # 比如 "自动" 但没时间戳 -> 默认走这里
                    # 计算字数差异（忽略首尾空格）
                    diff = abs(len(text.strip()) - len(clean_repl_line))
                    if diff <= 字数容差 or len(text.strip()) == 0:
                        should_replace = True
                
                if should_replace:
                    text = clean_repl_line
                    batch_repl_cursor += 1
                    batch_replaced_count += 1
                else:
                    # 如果不匹配，我们是否应该跳过这个替换行？
                    # 用户反馈"不完整"，意味着可能我们太严格了。
                    # 如果我们不替换，cursor 不动，下一次还会用这一行。
                    # 如果列表是 1:1 的，这会导致阻塞。
                    # 为了防止阻塞，我们还是应该尽量让 cursor 前进，
                    # 但如果我们跳过了，后面可能全错。
                    # 妥协方案：如果当前字幕非空且差异巨大，保留原字幕，cursor 不动（假设是额外插入的字幕）
                    # 但如果用户坚持"不完整"，很可能是因为 cursor 被卡住了。
                    # 鉴于用户可以选择"强制逐行"，在非强制模式下，我们保持原逻辑（卡住以防错位），
                    # 但增加了 "Original Empty" 的豁免，这应该能解决大部分"前面没识别出来"的问题。
                    pass
            
            # 4. 空白处理
            if 去除所有空格:
                text = text.replace(' ', '').replace('\t', '')
            elif 去除多余空格:
                text = re.sub(r'\s+', ' ', text)
            
            if 去除首尾空格:
                text = text.strip()
            
            # 5. 大小写转换
            if 大小写转换 == "全部大写":
                text = text.upper()
            elif 大小写转换 == "全部小写":
                text = text.lower()
            elif 大小写转换 == "首字母大写":
                text = text.capitalize()
            
            seg.text = text
            
            if text != original_text:
                modified_count += 1
        
        # 6. 去除短字幕
        if 去除短字幕 != "不限制":
            min_length_map = {
                "少于2字": 2,
                "少于3字": 3,
                "少于5字": 5,
            }
            min_length = min_length_map.get(去除短字幕, 0)
            before_count = len(processed_segments)
            processed_segments = [seg for seg in processed_segments if len(seg.text.strip()) >= min_length]
            removed_count += before_count - len(processed_segments)
        
        # 7. 去除空行
        if 去除空行:
            before_count = len(processed_segments)
            processed_segments = [seg for seg in processed_segments if seg.text.strip()]
            removed_count += before_count - len(processed_segments)
        
        # 8. 合并重复行
        if 合并重复行:
            merged_segments = []
            prev_seg = None
            for seg in processed_segments:
                if prev_seg and prev_seg.text == seg.text:
                    # 合并到前一个段
                    prev_seg.end_time = seg.end_time
                    removed_count += 1
                else:
                    merged_segments.append(seg)
                    prev_seg = seg
            processed_segments = merged_segments
        
        # 9. 按时间排序
        if 按时间排序:
            processed_segments.sort(key=lambda x: x.start_time)
        
        # 创建新的 ASRData 对象
        processed_data = ASRData(segments=processed_segments)
        
        # 生成纯文本
        if 保留换行符:
            text_output = "\n".join([seg.text for seg in processed_segments])
        else:
            text_output = " ".join([seg.text for seg in processed_segments])
        
        # 生成统计信息
        # 构建应用的处理列表
        applied_processes = []
        
        # 括号内容处理
        if 去除所有括号内容:
            applied_processes.append('✓ 去除所有括号内容')
        else:
            if 去除圆括号内容:
                applied_processes.append('✓ 去除圆括号内容')
            if 去除方括号内容:
                applied_processes.append('✓ 去除方括号内容')
            if 去除花括号内容:
                applied_processes.append('✓ 去除花括号内容')
            if 去除书名号内容:
                applied_processes.append('✓ 去除书名号内容')
            if 去除双引号内容:
                applied_processes.append('✓ 去除双引号内容')
            if 去除单引号内容:
                applied_processes.append('✓ 去除单引号内容')
        
        # 符号处理
        if 去除标点符号:
            applied_processes.append('✓ 去除标点符号')
        if 去除特殊符号:
            applied_processes.append('✓ 去除特殊符号')
        if 去除表情符号:
            applied_processes.append('✓ 去除表情符号')
        if 去除数字:
            applied_processes.append('✓ 去除数字')
        if 去除英文:
            applied_processes.append('✓ 去除英文')
        
        # 行处理
        if 去除空行:
            applied_processes.append('✓ 去除空行')
        if 合并重复行:
            applied_processes.append('✓ 合并重复行')
        if 按时间排序:
            applied_processes.append('✓ 按时间排序')
        if 去除短字幕 != '不限制':
            applied_processes.append(f'✓ 去除短字幕: {去除短字幕}')
        
        # 内容处理
        if 替换规则.strip():
            rule_count = len(替换规则.strip().split('\n'))
            applied_processes.append(f'✓ 内容替换: {rule_count} 条规则')
        if batch_replaced_count > 0:
            applied_processes.append(f'✓ 批量替换: {batch_replaced_count} 行')
        if 删除指定词.strip():
            word_count = len(删除指定词.strip().split('\n'))
            applied_processes.append(f'✓ 删除指定词: {word_count} 个词')
        if 大小写转换 != '不转换':
            applied_processes.append(f'✓ 大小写转换: {大小写转换}')
        
        processes_text = '\n'.join(applied_processes) if applied_processes else '无'
        
        input_mode = "文本输入模式" if is_text_mode else "字幕数据模式"
        
        stats = f"""=== 字幕文本处理统计 ===
输入模式: {input_mode}
原始字幕段数: {original_count}
处理后字幕段数: {len(processed_segments)}
修改的字幕: {modified_count}
删除的字幕: {removed_count}

应用的处理:
{processes_text}
"""
        
        print(f"[SubtitleTextProcessor] 处理完成: {original_count} → {len(processed_segments)} 字幕段")
        
        return (processed_data, text_output, stats.strip())
    
    def _remove_emoji(self, text: str) -> str:
        """去除 Emoji 表情符号"""
        # 使用 Unicode 范围去除 Emoji
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)
    
    def _remove_brackets_content(self, text: str, all_types: bool = False) -> str:
        """
        去除所有类型的括号及其内容
        
        Args:
            text: 输入文本
            all_types: 是否去除所有类型的括号
        
        Returns:
            处理后的文本
        """
        if all_types:
            # 圆括号（中英文）
            text = re.sub(r'\([^)]*\)', '', text)
            text = re.sub(r'（[^）]*）', '', text)
            
            # 方括号（中英文）
            text = re.sub(r'\[[^\]]*\]', '', text)
            text = re.sub(r'【[^】]*】', '', text)
            
            # 花括号
            text = re.sub(r'\{[^}]*\}', '', text)
            
            # 书名号
            text = re.sub(r'《[^》]*》', '', text)
            
            # 双引号（中英文）
            text = re.sub(r'"[^"]*"', '', text)
            text = re.sub(r'"[^"]*"', '', text)
            
            # 单引号（中英文）
            text = re.sub(r"'[^']*'", '', text)
            text = re.sub(r"'[^']*'", '', text)
        
        return text


NODE_CLASS_MAPPINGS = {
    "SubtitleTextProcessor": SubtitleTextProcessorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleTextProcessor": "字幕文本处理器"
}

