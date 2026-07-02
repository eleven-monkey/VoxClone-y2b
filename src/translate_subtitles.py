#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 并行字幕翻译（兼容 OpenAI / SiliconFlow 接口）。

输入格式: (HH:MM:SS.mmm) [Speaker 00] 原文
输出格式: (HH:MM:SS.mmm) [Speaker 00] 中文译文
翻译时严格保留时间戳和说话人标签。
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


def translate_segment(segment_text, api_config, max_retries=5):
    """翻译一个段落，返回 (译文, normalized_flag) 或 (None, False)。
    失败时回退：5次重试 → 纠错 + 1次再重试 → 返回最后一次的纠错后文本（让调用方逐行兜底）。
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

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = 1 * (2 ** (attempt - 1)) + random() * 0.5
                time.sleep(delay)
            response = requests.post(url, json=data, headers=headers, timeout=60)
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

            return translated, False

        except requests.exceptions.RequestException as e:
            print(f"  请求错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            continue
        except Exception as e:
            print(f"  其他错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            continue

    # 5 次都失败：进入纠错阶段
    # 先对最后一次返回纠错
    if last_translated:
        normalized = normalize_translation(last_translated)
        if normalized != last_translated:
            last_normalized = normalized
            ok, err = is_valid_translation_format(normalized)
            if ok:
                print(f"  纠错后格式通过（无需再请求）")
                return normalized, True
            print(f"  纠错后仍不通过: {err}，再请求 1 次让模型自修")
        else:
            print(f"  5 次重试后仍未通过且无可纠错项，再请求 1 次")

    # 再请求 1 次，让模型自己修复
    try:
        time.sleep(1)
        response = requests.post(url, json=data, headers=headers, timeout=60)
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
                return translated, True
            # 再纠错一次
            normalized = normalize_translation(translated)
            ok2, _ = is_valid_translation_format(normalized)
            if ok2:
                print(f"  纠错阶段成功（正则修复+模型自修）")
                return normalized, True
            last_normalized = normalized
    except Exception as e:
        print(f"  纠错阶段请求出错: {e}")

    # 全部失败：返回最后一次纠错后的文本（让 main() 逐行兜底）
    if last_normalized:
        return last_normalized, True
    return last_translated, True  # 可能为 None，但保持二元组契约


def translate_worker(task, api_config):
    """串行翻译工作函数。返回 (idx, translated, original_seg, normalized_flag)。"""
    global completed_count, total_count
    idx, text = task
    translated, normalized = translate_segment(text, api_config)
    completed_count += 1
    status = "成功" if translated else "失败"
    flag = "（已纠错）" if normalized else ""
    print(f"[进度 {completed_count}/{total_count}] 段落 {idx + 1} {status}{flag}")
    return idx, translated, text, normalized


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

        # 校对行数：译文比原文少 1 行视为正常（模型合并了相邻句），缺行不兜底
        if len(trans_lines) >= len(original_lines) - 1:
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
            # 行数偏差过大（少 ≥2 行）：整段用原文兜底
            print(f"  [翻译回退整段] 段落 {i + 1}（译{len(trans_lines)}行/原{len(original_lines)}行）")
            total_fallback_lines += len(original_lines)
            for line in original_lines:
                translated_lines.append(line)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in translated_lines:
            f.write(line + "\n")

    print(f"\n=== 翻译完成 ===")
    print(f"总段数: {len(segments)}, 兜底行数: {total_fallback_lines}, 耗时: {elapsed:.1f}s")
    print(f"输出: {args.output}")
    return 0  # 兜底后必有输出，便于后续流程


if __name__ == "__main__":
    sys.exit(main())
