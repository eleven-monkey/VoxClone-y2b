#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 串行字幕翻译（兼容 OpenAI / SiliconFlow 接口）。

输入格式: (HH:MM:SS.mmm) [Speaker 00] 原文
输出格式: (HH:MM:SS.mmm) [Speaker 00] 中文译文
翻译时严格保留时间戳和说话人标签。

API 失败（如 500 Server Error）时，用本地 llama-cpp-python 加载
tencent/Hy-MT2-1.8B-GGUF 兜底翻译，避免回退原文影响配音效果。
"""

import os
import re
import sys
import time
import json
import argparse
from random import random

import requests

SYSTEM_PROMPT = """# Role: 专业字幕翻译官

## 任务
将外文字幕翻译为中文，严格保留每行的格式。

## 格式要求（极其重要）
每行必须严格保持如下格式，不得改变时间戳和说话人标签：
(HH:MM:SS.mmm) [Speaker 00] 中文译文

## 规则
1. 时间戳 (HH:MM:SS.mmm) 必须原样保留，不得修改数字、格式或符号
2. 说话人标签 [Speaker 00] 必须原样保留，不得翻译、修改编号或改变符号
3. 只翻译说话人标签之后的正文内容为中文
4. 每行的时间戳和说话人标签与原文一一对应，不得合并、拆分或调换顺序
5. 正文尽量纯中文，不要中英文夹杂
6. 不要添加任何解释性文字、注释或说明
7. 不要使用 <think> 标签

## 示例
原文:
(00:01:23.456) [Speaker 00] Hello everyone, welcome to the show.
(00:01:30.123) [Speaker 01] Thanks for having me.

译文:
(00:01:23.456) [Speaker 00] 大家好，欢迎来到节目。
(00:01:30.123) [Speaker 01] 谢谢你们的邀请。
"""

# 本地兜底模型专用提示词（不出现 HH:MM:SS.mmm 字面量，避免小模型照抄为通配符）
FALLBACK_SYSTEM_PROMPT = """# Role: 专业字幕翻译官

## 任务
将外文字幕逐行翻译为中文，严格保留每行原有的时间戳和说话人标签。

## 格式要求（极其重要）
每行必须严格保持如下格式（时间戳为 `时:分:秒.毫秒`，三位整数加三位小数）：
(00:00:00.000) [Speaker 00] 中文译文

## 规则
1. 时间戳必须原样保留：括号、小数点、数字位数均与原文一致，例如 (12:34:56.789) → (12:34:56.789)
2. 说话人标签 [Speaker XX] 必须原样保留，不得翻译、修改编号或改变符号
3. 只翻译冒号之后的正文内容为中文，时间戳与说话人标签一字不改
4. 每行的时间戳和说话人标签与原文一一对应，不得合并、拆分或调换顺序
5. 不要添加任何解释、注释、空行或多余的标点

## 示例
原文:
(00:01:23.456) [Speaker 00] Hello everyone, welcome to the show.

译文:
(00:01:23.456) [Speaker 00] 大家好，欢迎来到节目。
"""

# 本地兜底模型配置（API 失败时启用）
FALLBACK_MODEL_REPO = "tencent/Hy-MT2-1.8B-GGUF"
FALLBACK_MODEL_FILE = "Hy-MT2-1.8B-Q4_K_M.gguf"


completed_count = 0
total_count = 0

# 字幕行格式: (HH:MM:SS.mmm) [Speaker 00] 文本
LINE_PATTERN = re.compile(r'^\((\d{2}:\d{2}:\d{2}\.\d{3})\) \[([^\]]+)\] (.+)$')
TS_PATTERN = re.compile(r'[\(（]\d{1,2}:\d{2}:\d{2}\.\d{1,3}[\)）]')


def read_lines(file_path):
    """读取字幕行。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def segment_lines(lines, segment_size=10, max_chars=2000):
    """将字幕行分组为段落，供 LLM 一次翻译多行。"""
    segments = []
    current = []
    current_chars = 0
    for line in lines:
        line_chars = len(line)
        if (len(current) >= segment_size) or \
           (current and current_chars + line_chars > max_chars):
            segments.append("\n".join(current))
            current = [line]
            current_chars = line_chars
        else:
            current.append(line)
            current_chars += line_chars
    if current:
        segments.append("\n".join(current))
    return segments


def contains_chinese(text):
    return bool(re.search(r"[\u4E00-\u9FFF\uF900-\uFAFF\u3400-\u4DBF]", text))


def normalize_translation(text):
    """纠错：把 [Speaker XX) 这种开口[闭口)的不匹配括号统一修复为 [Speaker XX]。
    只处理这一种已知错误，其他情况不乱动。
    """
    if not text:
        return text
    # 修复 [xxx) → [xxx]
    text = re.sub(r'\[([^\]\n]+?)\)', r'[\1]', text)
    return text


def _extract_text_after_speaker(line):
    """从合规行 '(ts) [Speaker XX] text' 中提取 text 部分。"""
    m = LINE_PATTERN.match(line.strip())
    if m:
        return m.group(3)
    return ""


def is_valid_translation_format(text):
    """校验翻译结果格式：每行符合 (HH:MM:SS.mmm) [Speaker XX] 中文，且时间戳递增。"""
    if not text or not text.strip():
        return False, "文本为空"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False, "没有有效行"
    prev_ts = None
    for i, line in enumerate(lines, 1):
        if not LINE_PATTERN.match(line):
            return False, f"第{i}行格式不正确: {line[:80]}"
        # 提取时间戳，检查严格递增
        ts_match = re.search(r'\((\d{2}:\d{2}:\d{2}\.\d{3})\)', line)
        if ts_match:
            ts = ts_match.group(1)
            if prev_ts is not None and ts <= prev_ts:
                return False, f"第{i}行时间戳未递增: {ts} <= {prev_ts}"
            prev_ts = ts
    return True, "格式正确"


def filter_think_tags(text):
    """过滤 <think></think> 标签及其内容。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def clean_content(content):
    """清理多余字符。"""
    content = (content.replace("&gt;", "").replace(">>", "").replace("> ", "")
               .replace("&nbsp;", "").replace("＞", ""))
    content = " ".join(content.split())
    return content


# ----------------------------------------------------------------------
# 本地 llama-cpp-python 兜底翻译
# ----------------------------------------------------------------------
_local_llm = None


def get_local_llm():
    """懒加载本地 llama-cpp-python 模型，仅在 API 失败时才加载。"""
    global _local_llm
    if _local_llm is not None:
        return _local_llm
    try:
        from llama_cpp import Llama
    except ImportError:
        print("  [兜底] llama-cpp-python 未安装，无法本地兜底")
        return None

    # 优先使用 HF_HOME / HF_HUB_CACHE 等环境变量定位缓存
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE") or None
    print(f"  [兜底] 加载本地模型 {FALLBACK_MODEL_REPO}/{FALLBACK_MODEL_FILE} ...")
    try:
        _local_llm = Llama.from_pretrained(
            repo_id=FALLBACK_MODEL_REPO,
            filename=FALLBACK_MODEL_FILE,
            cache_dir=cache_dir,
            n_ctx=4096,
            n_threads=4,
            verbose=False,
        )
        print("  [兜底] 本地模型加载完成")
    except Exception as e:
        print(f"  [兜底] 本地模型加载失败: {e}")
        _local_llm = None
    return _local_llm


def translate_segment_local(segment_text):
    """用本地 llama 模型翻译一个段落，返回译文或 None。"""
    llm = get_local_llm()
    if llm is None:
        return None
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": segment_text},
            ],
            max_tokens=4000,
            temperature=0.3,
            top_p=0.7,
        )
        translated = resp["choices"][0]["message"]["content"]
        translated = filter_think_tags(translated or "")
        if not translated or not contains_chinese(translated):
            print("  [兜底] 本地翻译未含中文")
            return None
        ok, err = is_valid_translation_format(translated)
        if not ok:
            print(f"  [兜底] 格式校验失败: {err}")
            return None
        return translated
    except Exception as e:
        print(f"  [兜底] 本地翻译异常: {e}")
        return None


def translate_segment(segment_text, api_config, max_retries=5):
    """翻译一个段落，返回 (译文, normalized_flag, server_error)。

    - 成功：(译文, False/True, False)
    - 全部重试 + 纠错都失败：返回最后一次文本（让调用方逐行兜底）
    - API 服务端持续错误（5xx 贯穿 max_retries + 纠错阶段）：返回 (None, False, True) 让调用方走本地模型兜底
    """
    global completed_count
    url = api_config["url"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_config.get('api_key', '')}",
    }
    data = {
        "model": api_config.get("model_name", "THUDM/GLM-4-9B-0414"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": segment_text},
        ],
        "stream": False,
        "max_tokens": 4000,
        "temperature": 0.3,
        "top_p": 0.7,
    }

    last_translated = None  # 记录最后一次原始返回，用于纠错阶段
    last_normalized = None  # 记录纠错后的文本
    saw_server_error = False  # 记录是否遇到 5xx，全部失败后用于决定走兜底

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = 1 * (2 ** (attempt - 1)) + random() * 0.5
                time.sleep(delay)
            response = requests.post(url, json=data, headers=headers, timeout=60)
            # 5xx 服务端错误：不立即返回，继续重试；累计到 max_retries 都失败后再走兜底
            if 500 <= response.status_code < 600:
                saw_server_error = True
                print(f"  服务端错误 {response.status_code} (尝试 {attempt + 1}/{max_retries})，将继续重试")
                continue
            response.raise_for_status()
            result = response.json()

            translated = None
            try:
                translated = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                pass

            translated = filter_think_tags(translated or "")
            last_translated = translated

            if not translated or not contains_chinese(translated):
                print(f"  翻译未含中文 (尝试 {attempt + 1}/{max_retries})")
                continue

            ok, err = is_valid_translation_format(translated)
            if not ok:
                print(f"  格式校验失败 (尝试 {attempt + 1}/{max_retries}): {err}")
                continue

            return translated, False, False

        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            if status is not None and 500 <= status < 600:
                saw_server_error = True
                print(f"  请求错误 {status} (尝试 {attempt + 1}/{max_retries}): {e}，将继续重试")
                continue
            print(f"  请求错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            continue
        except Exception as e:
            print(f"  其他错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            continue

    # 5 次都失败：进入纠错阶段
    # 如果主阶段每次都是 5xx（last_translated 为空），跳过纠错阶段直接走本地兜底
    if saw_server_error and not last_translated:
        print(f"  重试 {max_retries} 次均为服务端错误，跳过纠错阶段，直接走本地兜底")
        return None, False, True

    # 先对最后一次返回纠错
    if last_translated:
        normalized = normalize_translation(last_translated)
        if normalized != last_translated:
            last_normalized = normalized
            ok, err = is_valid_translation_format(normalized)
            if ok:
                print(f"  纠错后格式通过（无需再请求）")
                return normalized, True, False
            print(f"  纠错后仍不通过: {err}，再请求 1 次让模型自修")
        else:
            print(f"  5 次重试后仍未通过且无可纠错项，再请求 1 次")

    # 再请求 1 次，让模型自己修复
    try:
        time.sleep(1)
        response = requests.post(url, json=data, headers=headers, timeout=60)
        # 纠错阶段也遇到 5xx：记下来走本地兜底（前面主阶段已重试 max_retries 次）
        if 500 <= response.status_code < 600:
            saw_server_error = True
            print(f"  纠错阶段服务端错误 {response.status_code}，将尝试本地兜底")
        else:
            response.raise_for_status()
            result = response.json()
            translated = None
            try:
                translated = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                pass
            translated = filter_think_tags(translated or "")
            last_translated = translated

            if translated and contains_chinese(translated):
                ok, err = is_valid_translation_format(translated)
                if ok:
                    print(f"  纠错阶段成功（模型自修）")
                    return translated, True, False
                # 再纠错一次
                normalized = normalize_translation(translated)
                ok2, _ = is_valid_translation_format(normalized)
                if ok2:
                    print(f"  纠错阶段成功（正则修复+模型自修）")
                    return normalized, True, False
                last_normalized = normalized
    except requests.exceptions.RequestException as e:
        status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
        if status is not None and 500 <= status < 600:
            saw_server_error = True
            print(f"  纠错阶段请求错误 {status}，将尝试本地兜底")
        else:
            print(f"  纠错阶段请求出错: {e}")
    except Exception as e:
        print(f"  纠错阶段请求出错: {e}")

    # 5xx 全部失败：跳过逐行兜底，直接走本地模型兜底
    if saw_server_error and not last_normalized:
        return None, False, True

    # 全部失败：返回最后一次纠错后的文本（让 main() 逐行兜底）
    if last_normalized:
        return last_normalized, True, False
    return last_translated, True, False  # 可能为 None，但保持三元组契约


def translate_worker(task, api_config):
    """串行翻译工作函数。返回 (idx, translated, original_seg, normalized_flag)。

    API 服务端错误（5xx）或重试耗尽仍无结果时，用本地 llama 兜底翻译。
    """
    global completed_count, total_count
    idx, text = task
    translated, normalized, server_error = translate_segment(text, api_config)

    # API 服务端错误或重试耗尽仍无译文 → 尝试本地 llama 兜底
    if translated is None and server_error:
        print(f"  段落 {idx + 1} API 服务端错误，尝试本地 llama 兜底...")
        translated = translate_segment_local(text)
        if translated is not None:
            normalized = True  # 本地兜底结果视为已纠错

    completed_count += 1
    status = "成功" if translated else "失败"
    flag = "（已纠错/兜底）" if normalized else ""
    print(f"[进度 {completed_count}/{total_count}] 段落 {idx + 1} {status}{flag}")
    return idx, translated, text, normalized


# ----------------------------------------------------------------------
# 语句合并降速（从 actionsautopython 项目移植，增加同说话人约束）
# ----------------------------------------------------------------------

MAX_SPEAKING_WPM = 440  # 语速阈值（字/分钟），超过则尝试与相邻较慢行合并


def _ts_to_seconds(ts):
    """把时间戳字符串转成秒数。

    支持 HH:MM:SS.mmm / MM:SS.mmm / MM:SS 等任意段数。
    例：'1:23:45.678' -> 5025.678；'00:12.500' -> 12.5
    """
    parts = ts.split(':')
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + float(p)
    return seconds


def _extract_ts(line):
    """从合规行 '(ts) [Speaker XX] text' 中提取 ts 字符串（不含括号）。"""
    m = LINE_PATTERN.match(line.strip())
    if m:
        return m.group(1)
    return None


def _extract_speaker(line):
    """从合规行 '(ts) [Speaker XX] text' 中提取说话人标签。"""
    m = LINE_PATTERN.match(line.strip())
    if m:
        return m.group(2)
    return None


def _merge_two_lines(line_a, line_b):
    """合并两行：保留 line_a 的时间戳和说话人标签，拼接两行文本。

    line_a 是吸收者（时间戳较早），line_b 被吸收。
    调用方已保证两行说话人相同。
    """
    m_a = LINE_PATTERN.match(line_a.strip())
    m_b = LINE_PATTERN.match(line_b.strip())
    if not m_a or not m_b:
        return f"{line_a} {line_b}".strip()
    ts_a = m_a.group(1)
    speaker_a = m_a.group(2)
    text_a = m_a.group(3)
    text_b = m_b.group(3)
    merged_text = f"{text_a} {text_b}".strip()
    return f"({ts_a}) [{speaker_a}] {merged_text}"


def merge_fast_speaking_lines(lines, max_wpm=MAX_SPEAKING_WPM):
    """合并语速超过 max_wpm 的行，与相邻语速较慢者合并以降速。

    与 actionsautopython 原版相比，增加说话人约束：
    只在相同说话人的相邻行之间合并，不同说话人的行即使语速过快也不合并。

    策略：
    - 逐行计算语速 = 字数 / (下一行时间戳 - 本行时间戳) * 60
    - 找到第一行超阈值的行 i，检查其左右邻居的语速
    - 选邻居中语速较慢（且比自己慢）且**同说话人**的方向合并：
        - 合并左邻居：i 吸收进 i-1，保留 i-1 的时间戳和说话人
        - 合并右邻居：i 吸收 i+1，保留 i 的时间戳和说话人
    - 合并后重新计算所有行语速（因为时长结构变了），重复
    - 若某行超阈值但两邻居都不比自己慢或说话人不同，标记跳过，继续找下一个

    无时间戳的行无法计算语速，自然跳过。
    每次合并让总行数减 1，因此循环必然终止。

    返回 (合并后的新行列表, 合并标记列表)，均不修改原列表。
    合并标记列表与结果行一一对应，True 表示该行由合并产生。
    """
    lines = list(lines)
    is_merged = [False] * len(lines)
    merge_count = 0
    skipped = set()

    while True:
        n = len(lines)
        # 解析每行的时间戳（秒）、说话人、字数
        ts_secs = []
        speakers = []
        chars = []
        for line in lines:
            ts = _extract_ts(line)
            ts_sec = None
            if ts:
                try:
                    ts_sec = _ts_to_seconds(ts)
                except (ValueError, IndexError):
                    ts_sec = None
            ts_secs.append(ts_sec)
            speakers.append(_extract_speaker(line))
            text_only = _extract_text_after_speaker(line)
            if not text_only:
                text_only = TS_PATTERN.sub('', line)
            chars.append(len(re.sub(r'\s', '', text_only)))

        # 计算每行语速（最后一行无下一行，不计）
        wpms = [None] * n
        for i in range(n - 1):
            if ts_secs[i] is not None and ts_secs[i + 1] is not None:
                dur = ts_secs[i + 1] - ts_secs[i]
                if dur > 0 and chars[i] > 0:
                    wpms[i] = chars[i] / dur * 60

        # 找第一个可处理的超阈值行
        target = None
        merge_dir = None
        for i in range(n):
            if i in skipped or wpms[i] is None or wpms[i] <= max_wpm:
                continue
            left_wpm = wpms[i - 1] if i > 0 else None
            right_wpm = wpms[i + 1] if i < n - 1 else None
            left_same_spk = (i > 0 and speakers[i] == speakers[i - 1])
            right_same_spk = (i < n - 1 and speakers[i] == speakers[i + 1])
            # 在比自己慢且同说话人的邻居中选最慢的
            best = None
            if left_wpm is not None and left_wpm < wpms[i] and left_same_spk:
                best = ('left', left_wpm)
            if right_wpm is not None and right_wpm < wpms[i] and right_same_spk:
                if best is None or right_wpm < best[1]:
                    best = ('right', right_wpm)
            if best is not None:
                target = i
                merge_dir = best[0]
                break
            else:
                # 两邻居都不比自己慢或说话人不同，合并无法降速，跳过
                skipped.add(i)

        if target is None:
            break

        # 执行合并，行号变化后重新评估所有行
        skipped.clear()
        if merge_dir == 'left':
            lines[target - 1] = _merge_two_lines(lines[target - 1], lines[target])
            del lines[target]
            del is_merged[target]
            is_merged[target - 1] = True
        else:
            lines[target] = _merge_two_lines(lines[target], lines[target + 1])
            del lines[target + 1]
            del is_merged[target + 1]
            is_merged[target] = True
        merge_count += 1

    if merge_count > 0:
        print(f"[语速优化] 合并了 {merge_count} 个语速过快的行（阈值 {max_wpm} 字/分，仅同说话人合并）")
    return lines, is_merged


def estimate_speaking_rates(lines, merged_flags=None):
    """根据每行时间戳与下一行时间戳，估算每行语速（字/分钟）。

    语速 = 本句字数 / (下一句时间戳 - 本句时间戳) * 60。
    最后一行没有"下一句"，不计语速；无时间戳的行也跳过。
    字数按去掉时间戳和说话人标签后的非空白字符数计算。

    若传入 merged_flags（与 lines 等长的布尔列表），对标记为合并产生的行
    追加 '已合并' 标注，与语速一起显示在行末 '[...]' 中。

    返回新的行列表，在每行末尾追加 '  [语速: X字/分]' 等标注。
    """
    ts_seconds = []
    for line in lines:
        ts = _extract_ts(line)
        if ts:
            try:
                ts_seconds.append(_ts_to_seconds(ts))
            except (ValueError, IndexError):
                ts_seconds.append(None)
        else:
            ts_seconds.append(None)

    out_lines = []
    n = len(lines)
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        text_only = _extract_text_after_speaker(line)
        if not text_only:
            text_only = TS_PATTERN.sub('', line)
        char_count = len(re.sub(r'\s', '', text_only))

        merged = bool(merged_flags[i]) if merged_flags else False
        tags = []
        if i < n - 1 and ts_seconds[i] is not None and ts_seconds[i + 1] is not None:
            duration = ts_seconds[i + 1] - ts_seconds[i]
            if duration > 0 and char_count > 0:
                wpm = char_count / duration * 60
                tags.append(f"语速: {wpm:.0f}字/分")
        if merged:
            tags.append("已合并")
        if tags:
            out_lines.append(f"{line_stripped}  [{' | '.join(tags)}]")
        else:
            out_lines.append(line_stripped)
    return out_lines


def main():
    parser = argparse.ArgumentParser(description="LLM 并行字幕翻译")
    parser.add_argument("--input", required=True, help="输入字幕文件")
    parser.add_argument("--output", required=True, help="输出翻译字幕文件")
    parser.add_argument("--api_url", default=os.environ.get("AI_API_URL", ""), help="LLM API URL")
    parser.add_argument("--api_key", default=os.environ.get("AI_API_KEY", ""), help="LLM API Key")
    parser.add_argument("--model", default=os.environ.get("AI_MODEL", "THUDM/GLM-4-9B-0414"), help="模型名")
    parser.add_argument("--segment_size", type=int, default=10, help="每段最大行数")
    parser.add_argument("--max_workers", type=int, default=5, help="并行线程数")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("错误: 缺少 --api_url 或 --api_key (或 AI_API_URL/AI_API_KEY 环境变量)")
        return 1

    api_config = {
        "url": args.api_url,
        "api_key": args.api_key,
        "model_name": args.model,
    }

    lines = read_lines(args.input)
    print(f"读取 {len(lines)} 行字幕")

    segments = segment_lines(lines, segment_size=args.segment_size)
    print(f"分为 {len(segments)} 段，串行顺序翻译")

    global completed_count, total_count
    completed_count = 0
    total_count = len(segments)

    results = {}
    start_time = time.time()

    # 串行顺序翻译（不并发请求 API）
    for i, seg in enumerate(segments):
        idx, translated, original_seg, normalized = translate_worker((i, seg), api_config)
        results[idx] = (translated, original_seg, normalized)

    elapsed = time.time() - start_time

    # 按顺序合并：逐行兜底，确保行数与原文段完全一致
    translated_lines = []
    total_fallback_lines = 0
    for i in range(len(segments)):
        translated, original_seg, normalized = results.get(i, (None, segments[i], False))
        original_lines = [l.strip() for l in original_seg.splitlines() if l.strip()]

        if not translated:
            # 段落整体失败：每行用原文兜底
            total_fallback_lines += len(original_lines)
            for line in original_lines:
                translated_lines.append(line)
            continue

        trans_lines = [l.strip() for l in translated.splitlines() if l.strip()]

        # 校对行数：译文行数 ∈ [原文-1, 原文] 视为正常（最多允许模型合并一句），缺行不兜底
        if len(trans_lines) <= len(original_lines) and len(trans_lines) >= len(original_lines) - 1:
            if len(trans_lines) < len(original_lines):
                # 译文少 1 行：模型合并翻译，缺行不补
                print(f"  [译文合并] 段落 {i + 1}：译文{len(trans_lines)}行/原{len(original_lines)}行（合并翻译，跳过缺行）")
            for j, (tl, ol) in enumerate(zip(trans_lines, original_lines)):
                ok, _ = is_valid_translation_format(tl)
                if ok and contains_chinese(_extract_text_after_speaker(tl)):
                    translated_lines.append(tl)
                else:
                    # 单行校验失败：兜底用原文
                    total_fallback_lines += 1
                    print(f"  [翻译回退] 段落 {i + 1} 行: {ol[:60]}")
                    translated_lines.append(ol)
        else:
            # 行数偏差（译文多行 / 少 ≥2 行）：整段用原文兜底
            print(f"  [翻译回退整段] 段落 {i + 1}（译{len(trans_lines)}行/原{len(original_lines)}行）")
            total_fallback_lines += len(original_lines)
            for line in original_lines:
                translated_lines.append(line)

    # 合并语速过快的行（仅同说话人之间合并，阈值 440 字/分）
    original_line_count = len(translated_lines)
    translated_lines, merged_flags = merge_fast_speaking_lines(translated_lines)
    merged_line_count = original_line_count - len(translated_lines)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in translated_lines:
            f.write(line + "\n")

    print(f"\n=== 翻译完成 ===")
    print(f"总段数: {len(segments)}, 兜底行数: {total_fallback_lines}, 耗时: {elapsed:.1f}s")
    if merged_line_count > 0:
        print(f"语速合并: {original_line_count} 行 → {len(translated_lines)} 行（合并 {merged_line_count} 行）")
    print(f"输出: {args.output}")
    print("\n========== 合并降速后字幕 ==========")
    rate_lines = estimate_speaking_rates(translated_lines, merged_flags)
    print('\n'.join(rate_lines))
    print("====================================\n")
    return 0  # 兜底后必有输出，便于后续流程


if __name__ == "__main__":
    sys.exit(main())
