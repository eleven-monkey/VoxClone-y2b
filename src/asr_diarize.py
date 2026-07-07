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


def _resegment_with_pysbd(final_result, language="en"):
    """用 PySBD 对 WhisperX segments 重新分句，词级时间戳回填。

    WhisperX 的 segment 边界依赖 Whisper 自动生成的标点，对含点缩写
    （Mr./U.S./p.m./L.I./Ph.D./No./St. 等）会误切成两段。本函数后处理：

    1. 拍平所有 segments 的 words 为全局词列表 + 大文本，记录每个词
       在大文本中的字符偏移
    2. PySBD 对大文本重新切句
    3. 按字符范围把词映射回各句
    4. 每句 start=首词.start、end=末词.end、speaker=多数票
    5. 替换 final_result["segments"]

    若 PySBD 未安装、词级时间戳缺失、或 PySBD 输出长度与原文不符
    （说明切句改写了字符），原样返回 final_result，保证流程不中断。

    Args:
        final_result: whisperx.assign_word_speakers 的返回值
        language: 语言代码（en/zh/ja 等），用于 PySBD 选择规则集

    Returns:
        final_result（可能 segments 已重写）
    """
    try:
        import pysbd
    except ImportError:
        print("pysbd 未安装，跳过重分句（pip install pysbd 可启用）")
        return final_result

    segments = final_result.get("segments", [])
    if not segments:
        return final_result

    # 1. 拍平所有 words，构造大文本 + 字符偏移
    global_words = []
    text_parts = []
    cursor = 0
    for seg in segments:
        words = seg.get("words")
        if not words:
            # 任意段缺词级时间戳，无法回填，跳过重分句
            print("部分 segment 缺失词级时间戳，跳过重分句")
            return final_result
        for w in words:
            wtext = w.get("word", "")
            if not wtext:
                continue
            if text_parts:
                text_parts.append(" ")
                cursor += 1
            cs = cursor
            ce = cursor + len(wtext)
            global_words.append({
                "word": wtext,
                "start": w.get("start"),
                "end": w.get("end"),
                "speaker": w.get("speaker"),
                "char_start": cs,
                "char_end": ce,
            })
            text_parts.append(wtext)
            cursor = ce

    if not global_words:
        return final_result

    full_text = "".join(text_parts)

    # 2. PySBD 分句（clean=False 保留原文标点不清洗）
    try:
        segmenter = pysbd.Segmenter(language=language, clean=False)
    except (ValueError, KeyError):
        # 不支持的语言，回退 en
        print(f"PySBD 不支持语言 {language}，回退 en")
        segmenter = pysbd.Segmenter(language="en", clean=False)

    sent_texts = segmenter.segment(full_text)

    # 3. 累积定位每句字符范围，校验长度一致
    #    PySBD 不增删字符，只切边界，累积长度应等于原文长度
    sent_ranges = []
    pos = 0
    for st in sent_texts:
        sent_ranges.append((pos, pos + len(st)))
        pos += len(st)
    if pos != len(full_text):
        print(f"PySBD 输出长度 {pos} 与原文 {len(full_text)} 不符，跳过重分句")
        return final_result

    # 4. 字符范围 → 词列表 → 时间戳与说话人
    new_segments = []
    word_idx = 0
    n_words = len(global_words)
    for s, e in sent_ranges:
        # 双指针：词与句都按文本顺序排列，推进直到词起点 >= 句末
        sent_words = []
        while word_idx < n_words and global_words[word_idx]["char_start"] < e:
            w = global_words[word_idx]
            if w["char_end"] > s:
                sent_words.append(w)
            word_idx += 1

        if not sent_words:
            continue

        starts = [w["start"] for w in sent_words if w["start"] is not None]
        ends = [w["end"] for w in sent_words if w["end"] is not None]
        if not starts or not ends:
            # 词级时间戳缺失，跳过此句
            continue

        # 说话人多数票
        speakers = [w["speaker"] for w in sent_words if w.get("speaker")]
        if speakers:
            speaker = max(set(speakers), key=speakers.count)
        else:
            speaker = "SPEAKER_UNKNOWN"

        sent_text = full_text[s:e].strip()
        if not sent_text:
            continue

        new_segments.append({
            "start": starts[0],
            "end": ends[-1],
            "text": sent_text,
            "speaker": speaker,
            "words": [
                {
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                    "speaker": w.get("speaker"),
                }
                for w in sent_words
            ],
        })

    if not new_segments:
        print("PySBD 重分句未产出有效段，保留原 segments")
        return final_result

    print(f"PySBD 重分句: 原 {len(segments)} 段 → 新 {len(new_segments)} 段")
    final_result["segments"] = new_segments
    return final_result


def run_asr_diarize(audio_path, hf_token, output_dir, max_speakers=None, language=None):
    """执行 WhisperX 转写 + 说话人分离，返回 (word_level路径, diarize_segments, audio)。

    language: ISO 语言代码（如 en/zh/ja）。指定则跳过自动检测，避免误判
              （WhisperX 自动检测常把英语误判为 cy 等小语种，导致无对齐模型）。
    """
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
    transcribe_kwargs = {"batch_size": 1}
    if language:
        transcribe_kwargs["language"] = language
        print(f"使用指定语言: {language}（跳过自动检测）")
    result = model.transcribe(audio, **transcribe_kwargs)
    detected_lang = result["language"]
    print(f"转写完成，语言: {detected_lang}")

    # 释放转写模型内存
    del model
    gc.collect()

    # 4. 音素对齐
    # 兜底：WhisperX 自带的自动检测偶尔会把英语误判成 cy（威尔士语）等
    # 小语种，而这些语种没有对齐模型，会直接抛 ValueError 中断流程。
    # 若检测出的语言没有对齐模型，回退到 en 并警告（用户可用 --language 修正）。
    print(f"加载对齐模型（语言: {detected_lang}）...")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
    except ValueError as e:
        print(f"警告: 语言 {detected_lang} 无对齐模型: {e}")
        if language:
            print(f"指定的语言 {language} 不被对齐模型支持，回退到 en。")
        else:
            print("自动检测的语言不被对齐模型支持，回退到 en。")
            print("建议下次用 --language en（或其他支持的语言）显式指定以获得更好对齐质量。")
        detected_lang = "en"
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
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
        diarize_pipeline = DiarizationPipeline(token=hf_token, device=device)
    except TypeError:
        diarize_pipeline = DiarizationPipeline(use_auth_token=hf_token, device=device)

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

    # 6.5 PySBD 重分句，修正 Mr./U.S./p.m./L.I. 等含点缩写导致的误切
    final_result = _resegment_with_pysbd(final_result, language=detected_lang)

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
    parser.add_argument("--language", default="en",
                        help="ASR 语言代码（默认 en）。绝大部分翻译源视频是英语，"
                             "默认指定以避免 WhisperX 自动检测把英语误判成 cy 等小语种"
                             "导致对齐失败。其他语言如 zh/ja 按需传入。")
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
        audio_path, args.hf_token, args.output_dir, args.max_speakers,
        language=args.language
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
