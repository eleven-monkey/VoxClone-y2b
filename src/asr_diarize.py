#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WhisperX ASR + Pyannote 说话人分离 + 自动截取说话人参考音。

流程: yt-dlp 下载音频 → WhisperX(small, int8, CPU) 转写 → 音素对齐
     → Pyannote 说话人分离 → assign_word_speakers 对齐
     → 输出 word_level.txt (带时间戳和说话人标签)
     → 从原音频为每位说话人截取 5-15 秒参考音到 speaker_ref/

输出字幕格式: (HH:MM:SS.mmm) [Speaker 00] 原文
"""

import os
import re
import sys
import gc
import json
import argparse
import subprocess
from pathlib import Path


def download_audio(url, output_dir, cookies_file=None):
    """用 yt-dlp 下载音频为 wav，返回文件路径。"""
    os.makedirs(output_dir, exist_ok=True)
    out_tmpl = os.path.join(output_dir, "original_audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav", "--audio-quality", "0",
        "--extractor-args", "youtube:player_client=default,-web_safari",
        "--remote-components", "ejs:github",
        "--no-playlist",
        "-o", out_tmpl,
        url,
    ]
    if cookies_file and os.path.exists(cookies_file):
        cmd.extend(["--cookies", cookies_file])

    print(f"下载音频: {url}")
    subprocess.run(cmd, check=True)

    # 查找下载的 wav
    for f in os.listdir(output_dir):
        if f.startswith("original_audio") and f.endswith(".wav"):
            return os.path.join(output_dir, f)
    # 兜底：取目录下任意 wav
    wavs = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
    if wavs:
        return os.path.join(output_dir, wavs[0])
    raise FileNotFoundError("音频下载失败：未找到 wav 文件")


def format_timestamp(seconds):
    """秒数 → (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milli = int(round((seconds % 1) * 1000))
    if milli == 1000:
        secs += 1
        milli = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milli:03d}"


def run_asr_diarize(audio_path, hf_token, output_dir, max_speakers=None):
    """执行 WhisperX 转写 + 说话人分离，返回 (word_level路径, diarize_segments, audio)。"""
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    device = "cpu"
    # CPU 下用 int8 加速
    compute_type = "int8"
    print(f"设备: {device}, 计算精度: {compute_type}")

    # 1. 加载 WhisperX 模型
    print("加载 WhisperX 转写模型 (small)...")
    model = whisperx.load_model("small", device, compute_type=compute_type)

    # 2. 加载音频
    print(f"加载音频: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    # 3. 转写
    print("ASR 转写中（CPU，请耐心等待）...")
    result = model.transcribe(audio, batch_size=1)
    print(f"转写完成，检测语言: {result['language']}")

    # 释放转写模型内存
    del model
    gc.collect()

    # 4. 音素对齐
    print("加载对齐模型，提取词级时间戳...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False
    )
    del model_a
    gc.collect()

    # 5. 说话人分离
    print("初始化 Pyannote 说话人分离流水线...")
    try:
        diarize_pipeline = DiarizationPipeline(access_token=hf_token, device=device)
    except TypeError:
        diarize_pipeline = DiarizationPipeline(access_token=hf_token, device=device)

    print("说话人分离中（CPU，较慢）...")
    diarize_kwargs = {}
    if max_speakers:
        diarize_kwargs["max_speakers"] = max_speakers
    try:
        diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
    except TypeError:
        diarize_segments = diarize_pipeline(audio)

    # 6. 对齐说话人到词
    print("对齐说话人标签...")
    final_result = whisperx.assign_word_speakers(diarize_segments, result)

    # 7. 写出 word_level.txt
    os.makedirs(output_dir, exist_ok=True)
    word_level_path = os.path.join(output_dir, "word_level.txt")
    with open(word_level_path, "w", encoding="utf-8") as f:
        for segment in final_result["segments"]:
            start_time = segment.get("start", 0)
            speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
            text = segment.get("text", "").strip()
            if not text:
                continue
            # 规范化说话人标签: SPEAKER_00 → Speaker 00
            spk_clean = re.sub(r"SPEAKER_0*(\d+)", r"Speaker \1", speaker)
            spk_clean = re.sub(r"Speaker 0*(\d+)", r"Speaker \1", spk_clean)
            ts_str = format_timestamp(start_time)
            line = f"({ts_str}) [{spk_clean}] {text}"
            print(line)
            f.write(line + "\n")
    print(f"字幕已保存: {word_level_path}")

    return word_level_path, diarize_segments, audio


def extract_speaker_reference_audio(audio_path, diarize_segments, output_dir,
                                    min_duration=5.0, max_duration=15.0):
    """为每位说话人从原音频截取参考音片段。

    策略: 遍历该说话人所有片段，按时长降序排列；
    取最长段，若 >= min_duration 则裁剪到 max_duration 使用；
    若不足，拼接同说话人相邻段直至满足 min_duration。

    返回: {"Speaker 00": "speaker_ref/Speaker_00.wav", ...}
    """
    os.makedirs(output_dir, exist_ok=True)

    # 按说话人聚合片段
    speaker_segs = {}
    try:
        for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
            dur = turn.end - turn.start
            speaker_segs.setdefault(speaker, []).append((turn.start, turn.end, dur))
    except Exception:
        # 兼容部分版本返回 DataFrame
        for _, row in diarize_segments.iterrows():
            speaker = row.get("speaker", "SPEAKER_00")
            speaker_segs.setdefault(speaker, []).append(
                (row["start"], row["end"], row["end"] - row["start"])
            )

    ref_map = {}
    for speaker, segs in speaker_segs.items():
        # 规范化说话人名
        spk_clean = re.sub(r"SPEAKER_0*(\d+)", r"Speaker \1", speaker)
        spk_clean = re.sub(r"Speaker 0*(\d+)", r"Speaker \1", spk_clean)
        # 文件名安全化
        safe_name = spk_clean.replace(" ", "_")
        out_path = os.path.join(output_dir, f"{safe_name}.wav")

        # 按时长降序
        segs_sorted = sorted(segs, key=lambda x: x[2], reverse=True)
        longest = segs_sorted[0]
        start, end, dur = longest

        if dur >= min_duration:
            # 单段足够，裁剪到 max_duration
            clip_dur = min(dur, max_duration)
            _ffmpeg_cut(audio_path, start, clip_dur, out_path)
        else:
            # 拼接多段直至达到 min_duration
            collected = []
            total = 0.0
            for s, e, d in segs_sorted:
                collected.append((s, e))
                total += d
                if total >= min_duration:
                    break
            _ffmpeg_concat(audio_path, collected, out_path, max_duration)

        if os.path.exists(out_path):
            ref_map[spk_clean] = out_path
            print(f"说话人 {spk_clean} 参考音: {out_path}")
        else:
            print(f"警告: 说话人 {spk_clean} 参考音截取失败")

    return ref_map


def _ffmpeg_cut(audio_path, start, duration, out_path):
    """用 ffmpeg 截取单段音频。"""
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", audio_path, "-ac", "1", "-ar", "24000",
        "-c:a", "pcm_s16le", out_path
    ]
    subprocess.run(cmd, capture_output=True, check=False)


def _ffmpeg_concat(audio_path, segments, out_path, max_duration):
    """用 ffmpeg concat 拼接多段音频。"""
    if not segments:
        return
    list_file = out_path + ".list.txt"
    temp_files = []
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for i, (s, e) in enumerate(segments):
                tmp = f"{out_path}.part{i}.wav"
                cmd = [
                    "ffmpeg", "-y", "-ss", f"{s:.3f}", "-t", f"{(e - s):.3f}",
                    "-i", audio_path, "-ac", "1", "-ar", "24000",
                    "-c:a", "pcm_s16le", tmp
                ]
                subprocess.run(cmd, capture_output=True, check=False)
                if os.path.exists(tmp):
                    f.write(f"file '{os.path.abspath(tmp)}'\n")
                    temp_files.append(tmp)
        # concat
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-t", f"{max_duration:.3f}",
            "-c:a", "pcm_s16le", out_path
        ]
        subprocess.run(cmd, capture_output=True, check=False)
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)
        for tmp in temp_files:
            if os.path.exists(tmp):
                os.remove(tmp)


def main():
    parser = argparse.ArgumentParser(description="WhisperX ASR + 说话人分离 + 参考音截取")
    parser.add_argument("--url", required=True, help="YouTube 视频 URL")
    parser.add_argument("--output_dir", default="work", help="输出目录")
    parser.add_argument("--cookies", help="YouTube cookies 文件路径")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""), help="HuggingFace token")
    parser.add_argument("--max_speakers", type=int, default=None, help="最大说话人数量")
    parser.add_argument("--audio_path", default=None, help="已有音频路径(跳过下载)")
    args = parser.parse_args()

    if not args.hf_token:
        print("错误: 未提供 HF_TOKEN，Pyannote 模型需要授权")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 下载音频
    if args.audio_path and os.path.exists(args.audio_path):
        audio_path = args.audio_path
        print(f"使用已有音频: {audio_path}")
    else:
        audio_path = download_audio(args.url, args.output_dir, args.cookies)
    print(f"音频路径: {audio_path}")

    # 2. ASR + 说话人分离
    word_level_path, diarize_segments, _ = run_asr_diarize(
        audio_path, args.hf_token, args.output_dir, args.max_speakers
    )

    # 3. 截取说话人参考音
    ref_dir = os.path.join(args.output_dir, "speaker_ref")
    ref_map = extract_speaker_reference_audio(audio_path, diarize_segments, ref_dir)

    # 4. 保存说话人→参考音映射
    ref_map_path = os.path.join(args.output_dir, "speaker_ref_map.json")
    with open(ref_map_path, "w", encoding="utf-8") as f:
        json.dump(ref_map, f, ensure_ascii=False, indent=2)
    print(f"参考音映射已保存: {ref_map_path}")

    print("\n=== ASR + 说话人分离完成 ===")
    print(f"字幕: {word_level_path}")
    print(f"参考音目录: {ref_dir}")
    print(f"参考音映射: {ref_map_path}")
    print(f"说话人数: {len(ref_map)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
