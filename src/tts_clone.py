#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""moss-tts 语音克隆配音（核心创新模块）。

流程:
  解析翻译字幕 (timestamp + speaker + text)
  → 按说话人匹配参考音
  → 单进程加载 moss-tts ONNX 模型，循环推理所有句子（切换参考音）
  → audio_utils 变速防重叠 + numpy 共享内存混音
  → 输出 dubbed_audio.mp3

输入字幕格式: (HH:MM:SS.mmm) [Speaker 00] 中文译文
"""

import os
import sys
import json
import argparse
import traceback

# 同目录导入 audio_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import split_text_by_timestamp, parse_timestamp, mix_segments


def load_speaker_refs(ref_map_path):
    """加载说话人→参考音映射。"""
    with open(ref_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_subtitle(file_path):
    """解析翻译字幕，返回 [(timestamp_str, speaker, text), ...]。"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return split_text_by_timestamp(content)


def build_moss_runtime(moss_tts_dir):
    """加载 moss-tts ONNX runtime（只加载一次）。"""
    if moss_tts_dir and os.path.isdir(moss_tts_dir):
        sys.path.insert(0, os.path.abspath(moss_tts_dir))
    try:
        from onnx_tts_runtime import OnnxTtsRuntime
    except ImportError:
        # editable 安装后模块名可能不同，尝试备用导入
        try:
            from moss_tts.onnx_tts_runtime import OnnxTtsRuntime
        except ImportError as e:
            raise ImportError(
                f"无法导入 OnnxTtsRuntime，请确认 moss-tts 已安装。"
                f"moss_tts_dir={moss_tts_dir}。错误: {e}"
            )

    print("加载 moss-tts ONNX runtime (CPU)...")
    runtime = OnnxTtsRuntime(
        model_dir=None,          # None → 使用默认目录，缺失时自动下载
        thread_count=4,
        max_new_frames=375,
        do_sample=True,
        sample_mode="fixed",
        execution_provider="cpu",
    )
    return runtime


def clone_dub_segments(segments, speaker_ref_map, runtime, output_dir):
    """单进程循环推理所有句子，按说话人切换参考音。

    segments: [(timestamp_str, speaker, text), ...]
    speaker_ref_map: {"Speaker 00": "ref.wav", ...}
    返回: [(start_ms, output_wav_path), ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    fallback_ref = next(iter(speaker_ref_map.values()), None)

    total = len(segments)
    for i, (ts_str, speaker, text) in enumerate(segments):
        start_ms = parse_timestamp(f"({ts_str})")
        out_wav = os.path.join(output_dir, f"seg_{i:05d}.wav")

        # 匹配参考音
        ref_path = speaker_ref_map.get(speaker) or fallback_ref
        if not ref_path or not os.path.exists(ref_path):
            print(f"[{i+1}/{total}] 无参考音 for {speaker}，跳过: {text[:30]}")
            continue

        print(f"[{i+1}/{total}] {ts_str} [{speaker}] {text[:40]}...")
        try:
            result = runtime.synthesize(
                text=text,
                voice="Junhao",
                prompt_audio_path=ref_path,
                output_audio_path=out_wav,
                sample_mode="fixed",
                do_sample=True,
                streaming=True,
                max_new_frames=375,
                voice_clone_max_text_tokens=75,
                enable_wetext=True,
                enable_normalize_tts_text=True,
            )
            saved = result.get("audio_path", out_wav)
            if os.path.exists(saved):
                results.append((start_ms, saved))
            else:
                print(f"  警告: 输出文件不存在 {saved}")
        except Exception as e:
            print(f"  克隆失败: {e}")
            traceback.print_exc()
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="moss-tts 语音克隆配音")
    parser.add_argument("--input", required=True, help="翻译字幕文件")
    parser.add_argument("--output", required=True, help="输出配音 mp3 路径")
    parser.add_argument("--ref_map", required=True, help="说话人参考音映射 JSON")
    parser.add_argument("--moss_tts_dir", default="", help="MOSS-TTS-Nano 目录路径")
    parser.add_argument("--work_dir", default="work_tts", help="临时工作目录")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    # 1. 解析字幕
    segments = parse_subtitle(args.input)
    if not segments:
        print("错误: 未解析到任何字幕段落")
        return 1
    print(f"解析到 {len(segments)} 条字幕")

    # 2. 加载参考音映射
    speaker_ref_map = load_speaker_refs(args.ref_map)
    print(f"说话人参考音: {speaker_ref_map}")

    # 3. 加载 moss-tts 模型
    runtime = build_moss_runtime(args.moss_tts_dir)

    # 4. 逐句克隆推理
    seg_dir = os.path.join(args.work_dir, "segments")
    audio_entries = clone_dub_segments(segments, speaker_ref_map, runtime, seg_dir)
    print(f"成功生成 {len(audio_entries)} 条配音")

    if not audio_entries:
        print("错误: 没有成功生成任何配音")
        return 1

    # 按起始时间排序
    audio_entries.sort(key=lambda x: x[0])

    # 5. 变速防重叠 + 混音
    print("开始混音（含变速防重叠）...")
    final_audio = mix_segments(audio_entries)

    # 6. 导出 mp3
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    final_audio.export(args.output, format="mp3")
    print(f"\n=== 配音完成 ===")
    print(f"输出: {args.output}")
    print(f"时长: {len(final_audio) / 1000:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
